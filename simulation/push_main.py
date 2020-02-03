import logging
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from tensorboardX import SummaryWriter

from meta_contact import cfg, invariant
from meta_contact import model
from meta_contact import online_model
from meta_contact import prior
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.controller import global_controller
from meta_contact.env import block_push

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def random_touching_start(env, w=block_push.DIST_FOR_JUST_TOUCHING):
    init_block_pos = (np.random.random((2,)) - 0.5)
    init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
    # randomly initialize pusher adjacent to block
    # choose which face we will be next to
    env_type = type(env)
    if env_type == block_push.PushAgainstWallEnv:
        along_face = (np.random.random() - 0.5) * 2
        face = np.random.randint(0, 4)
        init_pusher = block_push.pusher_pos_for_touching(init_block_pos, init_block_yaw, from_center=w,
                                                         face=face,
                                                         along_face=along_face)
    elif env_type == block_push.PushAgainstWallStickyEnv or env_type == block_push.PushWithForceDirectlyEnv:
        init_pusher = np.random.uniform(-1, 1)
    else:
        raise RuntimeError("Unrecognized env type")
    return init_block_pos, init_block_yaw, init_pusher


# have to be set after selecting an environment
env_dir = None


def collect_touching_freespace_data(trials=20, trial_length=40, level=0):
    env = get_easy_env(p.DIRECT, level)
    u_min, u_max = env.get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    # use mode p.GUI to see what the trials look like
    save_dir = '{}{}'.format(env_dir, level)
    sim = block_push.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                     stop_when_done=False, save_dir=save_dir)
    rand.seed(4)
    # randomly distribute data
    for _ in range(trials):
        seed = rand.seed()
        # start at fixed location
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(env)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_easy_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_block_pos = [0, 0]
    init_block_yaw = 0
    init_pusher = 0
    goal_pos = [-0.3, 0.3]
    # env = interactive_block_pushing.PushAgainstWallEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher,
    #                                                    init_block=init_block_pos, init_yaw=init_block_yaw,
    #                                                    environment_level=level)
    # env_dir = 'pushing/no_restriction'
    # env = block_push.PushAgainstWallStickyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher, log_video=log_video,
    #                                           init_block=init_block_pos, init_yaw=init_block_yaw,
    #                                           environment_level=level)
    # env_dir = 'pushing/sticky'
    env = block_push.PushWithForceDirectlyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher, log_video=log_video,
                                              init_block=init_block_pos, init_yaw=init_block_yaw,
                                              environment_level=level)
    env_dir = 'pushing/direct_force'
    return env


def compare_to_goal_factory(env):
    def compare_to_goal(*args):
        return torch.from_numpy(env.compare_to_goal(*args))

    return compare_to_goal


def constrain_state(state):
    # yaw gets normalized
    state[:, 2] = math_utils.angle_normalize(state[:, 2])
    # along gets constrained
    state[:, 3] = math_utils.clip(state[:, 3], torch.tensor(-1, dtype=torch.double, device=state.device),
                                  torch.tensor(1, dtype=torch.double, device=state.device))
    return state


class HandDesignedCoordTransform(invariant.InvariantTransform):
    def __init__(self, ds, nz, **kwargs):
        # z_o is dx, dy, dyaw in body frame and d_along
        super().__init__(ds, nz, 4, **kwargs)
        self.name = 'coord_{}'.format(self.name)

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


class WorldBodyFrameTransformForStickyEnv(HandDesignedCoordTransform):
    """
    Specific to StickyEnv formulation! (expects the states to be block pose and pusher along)

    Transforms world frame coordinates to input required for body frame dynamics
    (along, d_along, and push magnitude) = z_i and predicts (dx, dy, dtheta) of block in previous block frame = z_o
    separate latent space for input and output (z_i, z_o)
    this is actually h and h^{-1} combined into 1, h(x,u) = z_i, learned dynamics hat{f}(z_i) = z_o, h^{-1}(z_o) = dx
    """

    def __init__(self, ds, **kwargs):
        # need along, d_along, and push magnitude; don't need block position or yaw
        super().__init__(ds, 3, **kwargs)

    def xu_to_zi(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # TODO might be able to remove push magnitude (and just directly multiply by that)
        # (along, d_along, push magnitude)
        z = torch.from_numpy(np.column_stack((state[:, -1], action)))
        return z


class ParameterizedCoordTransform(invariant.LearnLinearDynamicsTransform):
    """Parameterize the coordinate transform such that it has to learn something"""

    def __init__(self, ds, device, model_opts=None, **kwargs):
        if model_opts is None:
            model_opts = {}
        # z_o is dx, dy, dyaw in body frame and d_along
        nzo = 4
        nz = 1 + ds.config.nu
        # input is x, output is yaw
        self.yaw_selector = torch.nn.Linear(ds.config.nx, 1, bias=False).to(device=device, dtype=torch.double)
        self.true_yaw_param = torch.zeros(ds.config.nx, device=device, dtype=torch.double)
        self.true_yaw_param[2] = 1
        self.true_yaw_param = self.true_yaw_param.view(1, -1)  # to be consistent with weights
        # try starting at the true parameters
        # self.yaw_selector.weight.data = self.true_yaw_param + torch.randn_like(self.true_yaw_param)
        # self.yaw_selector.weight.requires_grad = False

        # input to local model is z, output is zo
        config = load_data.DataConfig()
        config.nx = nz
        config.ny = nzo * nz  # matrix output
        self.linear_model_producer = model.DeterministicUser(
            make.make_sequential_network(config, **model_opts).to(device=device))
        super().__init__(ds, nz, nzo, **kwargs)
        self.name = 'param_coord_{}'.format(self.name)

    def linear_dynamics(self, zi):
        B = zi.shape[0]
        return self.linear_model_producer.sample(zi).view(B, self.nzo, self.nz)

    def xu_to_zi(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # TODO make more general parameterized versions where we select which components to take
        # (along, d_along, push magnitude)
        z = torch.cat((state[:, -1].view(-1, 1), action), dim=1)
        return z

    def dx_to_zo(self, x, dx):
        raise RuntimeError("Shouldn't have to convert from dx to zo")

    def zo_to_dx(self, x, z_o):
        if len(x.shape) == 1:
            x = x.view(1, -1)
            z_o = z_o.view(1, -1)

        # choose which component of x to take as rotation (should select just theta)
        yaw = self.yaw_selector(x)

        N = z_o.shape[0]
        dx = torch.zeros((N, 4), dtype=z_o.dtype, device=z_o.device)
        # convert (dx, dy) from body frame back to world frame
        dx[:, :2] = math_utils.batch_rotate_wrt_origin(z_o[:, :2], yaw)
        # second last element is dyaw, which also gets passed along directly
        dx[:, 2] = z_o[:, 2]
        # last element is d_along, which gets passed along directly
        dx[:, 3] = z_o[:, 3]
        return dx

    def parameters(self):
        return list(self.yaw_selector.parameters()) + list(self.linear_model_producer.model.parameters())

    def _model_state_dict(self):
        d = {'yaw': self.yaw_selector.state_dict(), 'linear': self.linear_model_producer.model.state_dict()}
        return d

    def _load_model_state_dict(self, saved_state_dict):
        self.yaw_selector.load_state_dict(saved_state_dict['yaw'])
        self.linear_model_producer.model.load_state_dict(saved_state_dict['linear'])

    def _record_metrics(self, writer, losses, **kwargs):
        super()._record_metrics(writer, losses, **kwargs)

    def _evaluate_validation_set(self, writer):
        super(ParameterizedCoordTransform, self)._evaluate_validation_set(writer)
        with torch.no_grad():
            yaw_param = self.yaw_selector.weight.data
            cs = torch.nn.functional.cosine_similarity(yaw_param, self.true_yaw_param).item()
            dist = torch.norm(yaw_param - self.true_yaw_param).item()

            logger.debug("step %d yaw cos sim %f dist %f", self.step, cs, dist)

            writer.add_scalar('cosine_similarity', cs, self.step)
            writer.add_scalar('param_diff', dist, self.step)
            writer.add_scalar('param_norm', yaw_param.norm().item(), self.step)


class WorldBodyFrameTransformForDirectPush(HandDesignedCoordTransform):
    def __init__(self, ds, **kwargs):
        # need along, d_along, push magnitude, and push direction; don't need block position or yaw
        super().__init__(ds, 4, **kwargs)

    def xu_to_zi(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # (along, d_along, push magnitude, push direction)
        # z = torch.from_numpy(np.column_stack((state[:, -1], action)))
        z = torch.cat((state[:, -1].view(-1, 1), action), dim=1)
        return z


def coord_tsf_factory(env, *args, **kwargs):
    tsfs = {block_push.PushAgainstWallStickyEnv: WorldBodyFrameTransformForStickyEnv,
            block_push.PushWithForceDirectlyEnv: WorldBodyFrameTransformForDirectPush}
    tsf_type = tsfs.get(type(env), None)
    if tsf_type is None:
        raise RuntimeError("No tsf specified for env type {}".format(type(env)))
    return tsf_type(*args, **kwargs)


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

    tsf = coord_tsf_factory(env, ds)

    along = 0.7
    init_block_pos = [0, 0]
    init_block_yaw = 0
    env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=along)
    action = np.array([0, 0.4, 0])
    # push with original yaw (try this N times to confirm that all pushes are consistent)
    N = 10
    px, dx = None, None
    dxes = torch.zeros((N, env.ny))
    for i in range(N):
        px = env.reset()
        cx, _, _, _ = env.step(action)
        # get actual difference dx
        dx = get_dx(px, cx)
        dxes[i] = dx
    assert torch.allclose(dxes.std(0), torch.zeros(env.ny))
    assert px is not None
    # get input in latent space
    px = torch.from_numpy(px)
    z_i = tsf.xu_to_zi(px, action)
    # try inverting the transforms
    z_o_1 = tsf.dx_to_zo(px, dx)
    dx_inverted = tsf.zo_to_dx(px, z_o_1)
    assert torch.allclose(dx, dx_inverted)
    # same push with yaw, should result in the same z_i and the dx should give the same z_o but different dx

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
    PARAMETERIZED_1 = 2


def test_dynamics(level=0, use_tsf=UseTransform.COORDINATE_TRANSFORM, relearn_dynamics=False, online_adapt=True):
    seed = 1
    plot_model_error = False
    test_model_rollouts = True
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(p.GUI, level=level, log_video=True)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))

    # add in invariant transform here
    base_name = 'push_s{}'.format(seed)
    transforms = {UseTransform.NO_TRANSFORM: None,
                  UseTransform.COORDINATE_TRANSFORM: coord_tsf_factory(env, ds, name=base_name),
                  UseTransform.PARAMETERIZED_1: ParameterizedCoordTransform(ds, d, model_opts={'h_units': (16, 32)},
                                                                            # TODO currently using fixed tsf
                                                                            too_far_for_neighbour=0.3, name="_s2")}
    transform_names = {UseTransform.NO_TRANSFORM: 'none', UseTransform.COORDINATE_TRANSFORM: 'coord',
                       UseTransform.PARAMETERIZED_1: 'param1'}
    invariant_tsf = transforms[use_tsf]

    if invariant_tsf:
        training_epochs = 40
        # either load or learn the transform
        if use_tsf is not UseTransform.COORDINATE_TRANSFORM and not invariant_tsf.load(
                invariant_tsf.get_last_checkpoint()):
            invariant_tsf.learn_model(training_epochs, 5)

        if isinstance(invariant_tsf, invariant.LearnLinearDynamicsTransform):
            transformer = invariant.LearnLinearInvariantTransformer
        else:
            transformer = invariant.InvariantTransformer
        # wrap the transform as a data preprocessor
        preprocessor = preprocess.Compose(
            [transformer(invariant_tsf),
             preprocess.PytorchTransformer(preprocess.MinMaxScaler())])
    else:
        # use minmax scaling if we're not using an invariant transform (baseline)
        preprocessor = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)

    prior_name = '{}_prior'.format(transform_names[use_tsf])

    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                                   name=prior_name)

    pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                                 train_epochs=600)
    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)

    # test that the model predictions are relatively symmetric for positive and negative along
    if test_model_rollouts and isinstance(env, block_push.PushAgainstWallStickyEnv):
        N = 5
        x_top = torch.tensor([0, 0, 0, 1], dtype=torch.double, device=d).repeat(N, 1)
        x_bot = torch.tensor([0, 0, 0, -1], dtype=torch.double, device=d).repeat(N, 1)
        # push straight
        u = torch.tensor([0, 1, 0], dtype=torch.double, device=d)
        # do rollouts
        for i in range(N - 1):
            x_top[i + 1] = mw.predict(torch.cat((x_top[i], u)).view(1, -1))
            x_bot[i + 1] = mw.predict(torch.cat((x_bot[i], u)).view(1, -1))
        # check sign of the last states
        x = x_top[N - 1]
        assert x[0] > 0
        assert x[2] < 0  # yaw decreased (rotated ccw)
        assert abs(x[3] - x_top[0, 3]) < 0.1  # along hasn't changed much
        x = x_bot[N - 1]
        assert x[0] > 0
        assert x[2] > 0  # yaw increased (rotated cw)
        assert abs(x[3] - x_bot[0, 3]) < 0.1  # along hasn't changed much

    # plot model prediction
    if plot_model_error:
        XU, Y, _ = ds.validation_set()
        Y = Y.cpu().numpy()
        Yhatn = mw.user.sample(XU).cpu().detach().numpy()
        E = Yhatn - Y
        # relative error (compared to the mean magnitude)
        Er = E / np.mean(np.abs(Y), axis=0)
        for i in range(config.ny):
            plt.subplot(4, 2, 2 * i + 1)
            plt.plot(Y[:, i])
            plt.ylabel("$y_{}$".format(i))
            plt.subplot(4, 2, 2 * i + 2)
            plt.plot(Er[:, i])
            # plt.plot(E[:, i])
            plt.ylabel("$e_{}$".format(i))
        plt.show()

    u_min, u_max = env.get_control_bounds()
    Q = torch.diag(torch.tensor([10, 10, 0, 0.01], dtype=torch.double))
    R = 0.01
    # tune this so that we figure out to make u-turns
    mpc_opts = {
        'num_samples': 10000,
        'noise_sigma': torch.diag(torch.ones(env.nu, dtype=torch.double, device=d) * 0.5),
        'noise_mu': torch.tensor([0, 0.5, 0], dtype=torch.double, device=d),
        'lambda_': 1e-2,
        'horizon': 35,
        'u_init': torch.tensor([0, 0.5, 0], dtype=torch.double, device=d),
    }
    if online_adapt:
        dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, local_mix_weight=0.5, sigreg=1e-10)
        ctrl = online_controller.OnlineMPPI(dynamics, untransformed_config, Q=Q.numpy(), R=R, u_min=u_min, u_max=u_max,
                                            compare_to_goal=env.compare_to_goal,
                                            constrain_state=constrain_state,
                                            device=d, mpc_opts=mpc_opts)
    else:
        ctrl = global_controller.GlobalMPPIController(mw, untransformed_config, Q=Q, R=R, u_min=u_min, u_max=u_max,
                                                      compare_to_goal=env.compare_to_goal,
                                                      device=d,
                                                      mpc_opts=mpc_opts)

    name = pm.dyn_net.name if isinstance(pm, prior.NNPrior) else pm.__class__.__name__
    # expensive evaluation
    evaluate_controller(env, ctrl, name, translation=(10, 10), tasks=[885440])
    env.close()

    # sim = block_push.InteractivePush(env, ctrl, num_frames=150, plot=True, save=False, stop_when_done=True)
    # seed = rand.seed()
    # sim.run(seed)
    # logger.info("last run cost %f", np.sum(sim.last_run_cost))
    # plt.ioff()
    # plt.show()


def evaluate_controller(env: block_push.PushAgainstWallStickyEnv, ctrl: controller.Controller, name,
                        tasks: typing.Union[list, int] = 10, tries=10,
                        start_seed=0,
                        translation=(0, 0)):
    """Fixed set of benchmark tasks to do control over, with the total reward for each task collected and reported"""
    num_frames = 150
    env.set_camera_position(translation)
    env.draw_user_text('center {}'.format(translation), 1)
    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=False, save=False)

    name = "{}_{}".format(ctrl.__class__.__name__, name)
    env.draw_user_text(name, 14, left_offset=-1.5)
    writer = SummaryWriter(flush_secs=20, comment=name)

    seed = rand.seed(start_seed)

    if type(tasks) is int:
        tasks = [rand.seed() for _ in range(tasks)]

    logger.info("evaluation seed %d tasks %s tries %d", seed, tasks, tries)

    total_costs = torch.zeros((len(tasks), tries))
    lowest_costs = torch.zeros_like(total_costs)
    successes = torch.zeros_like(total_costs)
    for t in range(len(tasks)):
        task_seed = tasks[t]
        rand.seed(task_seed)
        # configure init and goal for task
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(env)
        init_block_pos = np.add(init_block_pos, translation)
        goal_pos = np.add(np.random.uniform(-0.6, 0.6, 2), translation)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher, goal=goal_pos)
        env.draw_user_text('task {}'.format(task_seed), 2)
        logger.info("task %d init block %s goal %s", task_seed, init_block_pos, goal_pos)

        task_costs = np.zeros((num_frames, tries))

        for i in range(tries):
            try_seed = rand.seed()
            env.draw_user_text('try {}'.format(try_seed), 3)
            env.draw_user_text('success {}/{}'.format(int(torch.sum(successes[t])), tries), 4)
            sim.run(try_seed)
            logger.info("task %d try %d run cost %f", task_seed, try_seed, sum(sim.last_run_cost))
            total_costs[t, i] = sum(sim.last_run_cost)
            lowest_costs[t, i] = min(sim.last_run_cost)
            task_costs[:len(sim.last_run_cost), i] = sim.last_run_cost
            if task_costs[-1, i] == 0:
                successes[t, i] = 1
            ctrl.reset()

        for step, costs in enumerate(task_costs):
            writer.add_histogram('ctrl_eval/task_{}'.format(task_seed), costs, step)

        task_mean_cost = torch.mean(total_costs[t])
        writer.add_scalar('ctrl_eval/task_{}'.format(task_seed), task_mean_cost, 0)
        logger.info("task %d cost: %f std %f", task_seed, task_mean_cost, torch.std(total_costs[t]))
        # clear trajectories of this task
        env.clear_debug_trajectories()

    # summarize stats
    logger.info("accumulated cost")
    logger.info(total_costs)
    logger.info("lowest costs per task and try")
    logger.info(lowest_costs)

    for t in range(len(tasks)):
        logger.info("task %d success %d/%d t cost %.2f (%.2f) l cost %.2f (%.2f)", tasks[t], torch.sum(successes[t]),
                    tries, torch.mean(total_costs[t]), torch.std(total_costs[t]), torch.mean(lowest_costs),
                    torch.std(lowest_costs))
    logger.info("total cost: %f (%f)", torch.mean(total_costs), torch.std(total_costs))
    logger.info("lowest cost: %f (%f)", torch.mean(lowest_costs), torch.std(lowest_costs))
    logger.info("total success: %d/%d", torch.sum(successes), torch.numel(successes))
    return total_costs


def learn_invariant(seed=1, name="", MAX_EPOCH=10, BATCH_SIZE=10):
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(p.DIRECT)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))

    # add in invariant transform here
    invariant_tsf = ParameterizedCoordTransform(ds, d, model_opts={'h_units': (16, 32)}, too_far_for_neighbour=0.3,
                                                name="{}_s{}".format(name, seed))
    invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)


def test_online_model():
    seed = 1
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(p.DIRECT, level=0)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))

    base_name = 'push_s{}'.format(seed)
    invariant_tsf = coord_tsf_factory(env, ds, name=base_name)
    transformer = invariant.InvariantTransformer
    preprocessor = preprocess.Compose(
        [transformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    ds.update_preprocessor(preprocessor)

    prior_name = 'coord_prior'

    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                                   name=prior_name)

    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=600)

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, local_mix_weight=0, sigreg=1e-10)

    # evaluate linearization by comparing error from applying model directly vs applying linearized model
    xuv, yv, _ = ds.original_validation_set()
    xv = xuv[:, :ds.original_config().nx]
    uv = xuv[:, ds.original_config().nx:]
    if ds.original_config().predict_difference:
        yv = yv + xv
    # full model prediction
    yhat1 = pm.dyn_net.predict(xuv)
    # linearized prediction
    yhat2 = dynamics.predict(None, None, xv, uv)

    e1 = torch.norm((yhat1 - yv), dim=1)
    e2 = torch.norm((yhat2 - yv), dim=1)
    assert torch.allclose(yhat1, yhat2)
    logger.info("Full model MSE %f linearized model MSE %f", e1.mean(), e2.mean())

    mix_weights = [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
    errors = torch.zeros(len(mix_weights))
    divergence = torch.zeros_like(errors)
    # debug updating linear model and using non-0 weight (error should increase with distance from update)
    for i, weight in enumerate(mix_weights):
        dynamics.empsig_N = weight
        yhat2 = dynamics.predict(None, None, xv, uv)
        errors[i] = torch.mean(torch.norm((yhat2 - yv), dim=1))
        divergence[i] = torch.mean(torch.norm((yhat2 - yhat1), dim=1))
    logger.info("error with increasing weight %s", errors)
    logger.info("divergence increasing weight %s", divergence)

    # use just a single trajectory
    N = 49  # xv.shape[0]-1
    xv = xv[:N]
    uv = uv[:N]
    yv = yv[:N]

    horizon = 3
    dynamics.gamma = 0.1
    dynamics.empsig_N = 1.0
    compare_against_last_updated = False
    errors = torch.zeros((N, 3))
    GLOBAL = 0
    BEFORE = 1
    AFTER = 2

    yhat2 = dynamics.predict(None, None, xv, uv)
    e2 = torch.norm((yhat2 - yv), dim=1)
    for i in range(N - 1):
        if compare_against_last_updated:
            yhat2 = dynamics.predict(None, None, xv, uv)
            e2 = torch.norm((yhat2 - yv), dim=1)
        dynamics.update(xv[i], uv[i], xv[i + 1])
        # after
        yhat3 = dynamics.predict(None, None, xv, uv)
        e3 = torch.norm((yhat3 - yv), dim=1)

        errors[i, GLOBAL] = e1[i + 1:i + 1 + horizon].mean()
        errors[i, BEFORE] = e2[i + 1:i + 1 + horizon].mean()
        errors[i, AFTER] = e3[i + 1:i + 1 + horizon].mean()
        # when updated with xux' from ground truth, should do better at current location
        if errors[i, AFTER] > errors[i, BEFORE]:
            logger.warning("error increased after update at %d", i)
        # also look at error with global model
        logger.info("global before after %s", errors[i])

    errors = errors[:N - 1]
    # plot these two errors relative to the global model error
    plt.figure()
    plt.plot(errors[:, BEFORE] / errors[:, GLOBAL])
    plt.plot(errors[:, AFTER] / errors[:, GLOBAL])
    plt.title(
        'local error after update for horizon {} gamma {} weight {}'.format(horizon, dynamics.gamma, dynamics.empsig_N))
    plt.xlabel('step')
    plt.ylabel('relative error to global model')
    plt.yscale('log')
    plt.legend(['before update', 'after update'])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # collect_touching_freespace_data(trials=200, trial_length=50, level=0)
    # test_dynamics(0, use_tsf=UseTransform.NO_TRANSFORM, online_adapt=False)
    test_dynamics(0, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=True)
    # verify_coordinate_transform()
    # for seed in range(10):
    #     learn_invariant(seed=seed, name="", MAX_EPOCH=40)
    # test_online_model()
