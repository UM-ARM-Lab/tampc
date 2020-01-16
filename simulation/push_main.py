import logging
import math
import pybullet as p

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as skpre
import torch
from arm_pytorch_utilities import preprocess, math_utils
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from meta_contact import cfg
from meta_contact import model
from meta_contact import prior
from meta_contact.controller import controller
from meta_contact.controller import global_controller
from meta_contact.controller import online_controller
from meta_contact.env import block_push
from tensorboardX import SummaryWriter

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


def test_local_dynamics(level=0):
    num_frames = 70
    env = get_easy_env(p.DIRECT, level=level)
    # TODO preprocessor in online dynamics not yet supported
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), preprocessor=preprocessor,
                                   validation_ratio=0.1, config=config)

    # TODO add in invariant transform here

    m = model.DeterministicUser(make.make_sequential_network(config))
    mw = model.NetworkModelWrapper(m, ds, name='contextual')
    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=200)
    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    u_min, u_max = get_control_bounds()
    # ctrl = online_controller.OnlineLQR(pm, ds=ds, max_timestep=num_frames, R=5, horizon=20, lqr_iter=3,
    #                                    init_gamma=0.1, u_min=u_min, u_max=u_max)

    Q = torch.diag(torch.tensor([1, 1, 0, 0.01], dtype=torch.double))
    R = 1
    ctrl = online_controller.OnlineCEM(pm, ds, Q=Q.numpy(), R=R, u_min=u_min, u_max=u_max,
                                       compare_to_goal=env.compare_to_goal,
                                       constrain_state=constrain_state, mpc_opts={'init_cov_diag': 0.002})  # use seed 7

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
    test_local_dynamics(0)
