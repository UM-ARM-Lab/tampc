import copy
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import sklearn.preprocessing as skpre
import torch
from arm_pytorch_utilities import preprocess, math_utils
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from tensorboardX import SummaryWriter

from meta_contact import cfg, invariant
from meta_contact import model
from meta_contact import online_model
from meta_contact import prior
from meta_contact.controller import controller
from meta_contact.controller import global_controller
from meta_contact.controller import online_controller
from meta_contact.env import block_push

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def random_touching_start(w=block_push.DIST_FOR_JUST_TOUCHING):
    init_block_pos = (np.random.random((2,)) - 0.5)
    init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
    # randomly initialize pusher adjacent to block
    # choose which face we will be next to
    along_face = (np.random.random() - 0.5) * 2 * w  # each face has 1 fixed value and 1 free value
    face = np.random.randint(0, 4)
    # for sticky environment, only have to give how far along the face
    init_pusher = np.random.uniform(-block_push.MAX_ALONG, block_push.MAX_ALONG)
    # init_pusher = interactive_block_pushing.pusher_pos_for_touching(init_block_pos, init_block_yaw, from_center=w,
    #                                                                 face=face,
    #                                                                 along_face=along_face)
    return init_block_pos, init_block_yaw, init_pusher


def get_control_bounds():
    # depends on the environment; these are the limits for StickyEnv
    u_min = np.array([-0.02, 0])
    u_max = np.array([0.02, 0.03])
    return u_min, u_max


def collect_touching_freespace_data(trials=20, trial_length=40, level=0):
    env = get_easy_env(p.DIRECT, level)
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching{}'.format(level)
    sim = block_push.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                     save_dir=save_dir)
    rand.seed(4)
    # deterministically spread out the data
    # init_block_pos = (0, 0)
    # for init_block_yaw in np.linspace(-2., 2., 10):
    #     for init_pusher in np.linspace(-interactive_block_pushing.MAX_ALONG, interactive_block_pushing.MAX_ALONG, 10):
    #         seed = rand.seed()
    #         # start at fixed location
    #         env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    #         ctrl = controller.FullRandomController(2, (-0.01, 0), (0.01, 0.03))
    #         sim.ctrl = ctrl
    #         sim.run(seed)

    # randomly distribute data
    for _ in range(trials):
        seed = rand.seed()
        # start at fixed location
        init_block_pos, init_block_yaw, init_pusher = random_touching_start()
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def get_data_dir(level=0):
    return 'pushing/touching{}.mat'.format(level)


def get_easy_env(mode=p.GUI, level=0, log_video=False):
    init_block_pos = [0, 0]
    init_block_yaw = 0
    # init_pusher = [-0.095, 0]
    init_pusher = 0
    # goal_pos = [1.1, 0.5]
    # goal_pos = [0.8, 0.2]
    goal_pos = [0.5, 0.5]
    # env = interactive_block_pushing.PushAgainstWallEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher,
    #                                                    init_block=init_block_pos, init_yaw=init_block_yaw,
    #                                                    environment_level=level)
    env = block_push.PushAgainstWallStickyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher, log_video=log_video,
                                              init_block=init_block_pos, init_yaw=init_block_yaw,
                                              environment_level=level)
    return env


def compare_to_goal_factory(env):
    def compare_to_goal(*args):
        return torch.from_numpy(env.compare_to_goal(*args))

    return compare_to_goal


def constrain_state(state):
    # yaw gets normalized
    state[:, 2] = math_utils.angle_normalize(state[:, 2])
    # along gets constrained
    state[:, 3] = math_utils.clip(state[:, 3], -torch.tensor(block_push.MAX_ALONG, dtype=torch.double),
                                  torch.tensor(block_push.MAX_ALONG, dtype=torch.double))
    return state


class WorldBodyFrameTransform(invariant.InvariantTransform):
    """
    Specific to StickyEnv formulation! (expects the states to be block pose and pusher along)

    Transforms world frame coordinates to input required for body frame dynamics
    (along, d_along, and push magnitude) = z_i and predicts (dx, dy, dtheta) of block in previous block frame = z_o
    separate latent space for input and output (z_i, z_o)
    this is actually h and h^{-1} combined into 1, h(x,u) = z_i, learned dynamics hat{f}(z_i) = z_o, h^{-1}(z_o) = dx
    """

    def __init__(self, ds, **kwargs):
        # need along, d_along, and push magnitude; don't need block position or yaw
        self.nzo = 4
        super().__init__(ds, 3, **kwargs)
        self.name = 'coord_{}'.format(self.name)

    @staticmethod
    def supports_only_direct_zi_to_dx():
        # converts z to dx in body frame, then needs to bring back to world frame
        return False

    def xu_to_zi(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # TODO might be able to remove push magnitude (and just directly multiply by that)
        # (along, d_along, push magnitude)
        z = torch.from_numpy(np.column_stack((state[:, -1], action)))
        return z

    def dx_to_zo(self, x, dx):
        if len(x.shape) == 1:
            x = x.view(1, -1)
            dx = dx.view(1, -1)
        N = dx.shape[0]
        z_o = torch.zeros((N, self.nzo), dtype=dx.dtype, device=dx.device)
        # convert world frame to body frame
        z_o[:, :2] = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
        # second last element is dyaw, which also gets passed along directly
        z_o[:, 2] = dx[:, 2]
        # last element is d_along, which gets passed along directly
        z_o[:, 3] = dx[:, 3]
        return z_o

    def zo_to_dx(self, x, z_o):
        if len(x.shape) == 1:
            x = x.view(1, -1)
            z_o = z_o.view(1, -1)
        N = z_o.shape[0]
        dx = torch.zeros((N, 4), dtype=z_o.dtype, device=z_o.device)
        # convert (dx, dy) from body frame back to world frame
        dx[:, :2] = math_utils.batch_rotate_wrt_origin(z_o[:, :2], x[:, 2])
        # second last element is dyaw, which also gets passed along directly
        dx[:, 2] = z_o[:, 2]
        # last element is d_along, which gets passed along directly
        dx[:, 3] = z_o[:, 3]
        return dx

    def parameters(self):
        return [torch.zeros(1)]

    def _model_state_dict(self):
        return None

    def _load_model_state_dict(self, saved_state_dict):
        pass

    def learn_model(self, max_epoch, batch_N=500):
        pass


def verify_coordinate_transform():
    def get_dx(px, cx):
        dpos = cx[:2] - px[:2]
        dyaw = math_utils.angular_diff(cx[2], px[2])
        dalong = cx[3] - px[3]
        dx = torch.from_numpy(np.r_[dpos, dyaw, dalong])
        return dx

    # comparison tolerance
    tol = 2e-4
    env = get_easy_env(p.GUI)
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config)

    tsf = WorldBodyFrameTransform(ds)

    along = block_push.MAX_ALONG / 1.5
    init_block_pos = [0, 0]
    init_block_yaw = 0
    env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=along)
    action = np.array([0, 0.02])
    # push with original yaw (try this N times to confirm that all pushes are consistent)
    N = 10
    dxes = torch.zeros((N, env.ny))
    for i in range(N):
        px = env.reset()
        cx, _, _, _ = env.step(action)
        # get actual difference dx
        dx = get_dx(px, cx)
        dxes[i] = dx
    assert torch.allclose(dxes.std(0), torch.zeros(env.ny))
    # get input in latent space
    px = torch.from_numpy(px)
    z_i = tsf.xu_to_zi(px, action)
    # try inverting the transforms
    z_o_1 = tsf.dx_to_zo(px, dx)
    dx_inverted = tsf.zo_to_dx(px, z_o_1)
    assert torch.allclose(dx, dx_inverted)
    # same push with yaw, should result in the same z_i and the dx should give the same z_o but different dx

    # TODO fit linear model in z space; should get the same parameters
    N = 16
    dxes = torch.zeros((N, env.ny))
    z_os = torch.zeros((N, env.ny))
    # for i, yaw_shift in enumerate(np.linspace(0, math.pi*2, 4)):
    for i, yaw_shift in enumerate(np.linspace(0, math.pi * 2, N)):
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw + yaw_shift, init_pusher=along)
        px = env.reset()
        cx, _, _, _ = env.step(action)
        # get actual difference dx
        dx = get_dx(px, cx)
        px = torch.from_numpy(px)
        z_i_2 = tsf.xu_to_zi(px, action)
        assert torch.allclose(z_i, z_i_2, atol=tol / 10)
        z_o_2 = tsf.dx_to_zo(px, dx)
        z_os[i] = z_o_2
        dxes[i] = dx
        dx_inverted_2 = tsf.zo_to_dx(px, z_o_2)
        assert torch.allclose(dx, dx_inverted_2)
    # change in body frame should be exactly the same
    logger.info(z_os)
    # relative standard deviation
    logger.info(z_os.std(0) / torch.abs(z_os.mean(0)))
    assert torch.allclose(z_os.std(0), torch.zeros(4), atol=tol)
    # actual dx should be different since we have yaw
    assert not torch.allclose(dxes.std(0), torch.zeros(4), atol=tol)


class UseTransform:
    NO_TRANSFORM = 0
    COORDINATE_TRANSFORM = 1


def test_local_dynamics(level=0):
    num_frames = 70
    seed = 1
    use_tsf = UseTransform.NO_TRANSFORM

    env = get_easy_env(p.DIRECT, level=level)
    # TODO preprocessor in online dynamics not yet supported
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), preprocessor=preprocessor,
                                   validation_ratio=0.1, config=config)
    untransformed_config = copy.deepcopy(config)

    logger.info("initial random seed %d", rand.seed(seed))

    # add in invariant transform here
    base_name = 'push_s{}'.format(seed)
    transforms = {UseTransform.NO_TRANSFORM: None,
                  UseTransform.COORDINATE_TRANSFORM: WorldBodyFrameTransform(ds, name=base_name)}
    transform_names = {UseTransform.NO_TRANSFORM: 'none', UseTransform.COORDINATE_TRANSFORM: 'coord'}
    invariant_tsf = transforms[use_tsf]

    if invariant_tsf:
        training_epochs = 40
        # either load or learn the transform
        if use_tsf is not UseTransform.COORDINATE_TRANSFORM and not invariant_tsf.load(
                invariant_tsf.get_last_checkpoint()):
            invariant_tsf.learn_model(training_epochs, 5)

        # wrap the transform as a data preprocessor
        preprocessor = invariant.InvariantPreprocessor(invariant_tsf)
        # update the datasource to use transformed data
        ds.update_preprocessor(preprocessor)

    prior_name = '{}_prior'.format(transform_names[use_tsf])

    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds, name=prior_name)
    pm = prior.NNPrior.from_data(mw, checkpoint=None, train_epochs=200)
    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    u_min, u_max = get_control_bounds()
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, untransformed_config, sigreg=1e-10)
    Q = torch.diag(torch.tensor([1, 1, 0, 0.01], dtype=torch.double))
    R = 1

    ctrl = online_controller.OnlineCEM(dynamics, untransformed_config, Q=Q.numpy(), R=R, u_min=u_min, u_max=u_max,
                                       compare_to_goal=env.compare_to_goal,
                                       constrain_state=constrain_state, mpc_opts={'init_cov_diag': 0.002})
    # ctrl = online_controller.OnlineMPPI(dynamics, untransformed_config, Q=Q.numpy(), R=R, u_min=u_min, u_max=u_max,
    #                                     compare_to_goal=env.compare_to_goal,
    #                                     constrain_state=constrain_state,
    #                                     mpc_opts={'num_samples': 10000,
    #                                               'noise_sigma': torch.eye(env.nu, dtype=torch.double) * 0.001})

    name = pm.dyn_net.name if isinstance(pm, prior.NNPrior) else pm.__class__.__name__
    # expensive evaluation
    evaluate_controller(env, ctrl, name)

    # sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=True, save=False)
    # seed = rand.seed()
    # sim.run(seed)
    # logger.info("last run cost %f", sim.last_run_cost)
    # plt.ioff()
    # plt.show()


def evaluate_controller(env, ctrl, name, tasks=10, tries=10, start_seed=0):
    """Fixed set of benchmark tasks to do control over, with the total reward for each task collected and reported"""
    num_frames = 100
    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=False, save=False)

    name = "{}_{}".format(ctrl.__class__.__name__, name)
    writer = SummaryWriter(flush_secs=20, comment=name)

    seed = rand.seed(start_seed)
    logger.info("evaluation seed %d tasks %d tries %d", seed, tasks, tries)

    total_costs = np.zeros((tasks, tries))
    for t in range(tasks):
        task_seed = rand.seed()
        # configure init and goal for task
        init_block_pos, init_block_yaw, init_pusher = random_touching_start()
        goal_pos = np.random.uniform(-0.6, 0.6, 2)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher, goal=goal_pos)
        logger.info("task %d init block %s goal %s", task_seed, init_block_pos, goal_pos)

        task_costs = np.zeros((num_frames, tries))

        for i in range(tries):
            try_seed = rand.seed()
            sim.run(try_seed)
            logger.info("task %d try %d run cost %f", task_seed, try_seed, sum(sim.last_run_cost))
            total_costs[t, i] = sum(sim.last_run_cost)
            task_costs[:, i] = sim.last_run_cost

        for step, costs in enumerate(task_costs):
            writer.add_histogram('ctrl_eval/task_{}'.format(task_seed), costs, step)

        task_mean_cost = np.mean(total_costs[t])
        writer.add_scalar('ctrl_eval/task_{}'.format(task_seed), task_mean_cost, 0)
        logger.info("task %d cost: %f std %f", task_seed, task_mean_cost, np.std(total_costs[t]))

    # summarize stats
    mean_cost = np.mean(total_costs)
    logger.info("total cost: %f std %f", mean_cost, np.std(total_costs))
    writer.add_scalar('ctrl_eval/total', mean_cost, 0)
    return total_costs


def test_global_linear_dynamics(level=0):
    env = get_easy_env(p.GUI, level)
    config = load_data.DataConfig(predict_difference=False, predict_all_dims=True)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), validation_ratio=0.01, config=config)

    u_min, u_max = get_control_bounds()
    ctrl = global_controller.GlobalLQRController(ds, R=100, u_min=u_min, u_max=u_max)
    sim = block_push.InteractivePush(env, ctrl, num_frames=50, plot=True, save=False)

    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_global_qr_cost_optimal_controller(controller, level=0, **kwargs):
    env = get_easy_env(p.GUI, level=level)
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), validation_ratio=0.1,
                                   config=config, preprocessor=preprocessor)
    pml = model.LinearModelTorch(ds)
    pm = model.NetworkModelWrapper(
        model.DeterministicUser(
            make.make_sequential_network(config, activation_factory=torch.nn.LeakyReLU, h_units=(32, 32))), ds)

    if not pm.load(pm.get_last_checkpoint()):
        pm.learn_model(200, batch_N=10000)

    pm.freeze()

    Q = torch.diag(torch.tensor([1, 1, 0, 0.01], dtype=torch.double))
    u_min, u_max = get_control_bounds()
    u_min = torch.tensor(u_min, dtype=torch.double)
    u_max = torch.tensor(u_max, dtype=torch.double)
    ctrl = controller(pm, ds, Q=Q, u_min=u_min, u_max=u_max, compare_to_goal=compare_to_goal_factory(env), **kwargs)
    sim = block_push.InteractivePush(env, ctrl, num_frames=200, plot=True, save=False)

    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # collect_touching_freespace_data(trials=100, trial_length=50, level=0)
    # ctrl = global_controller.GlobalCEMController
    # test_global_qr_cost_optimal_controller(ctrl, num_samples=1000, horizon=7, num_elite=50, level=0,
    #                                        init_cov_diag=0.002)  # CEM options
    # ctrl = global_controller.GlobalMPPIController
    # test_global_qr_cost_optimal_controller(ctrl, num_samples=1000, horizon=7, level=0, lambda_=0.1,
    #                                        noise_sigma=torch.diag(
    #                                            torch.tensor([0.01, 0.01], dtype=torch.double)))  # MPPI options
    # test_global_linear_dynamics(level=0)
    # test_local_dynamics(0)
    verify_coordinate_transform()
