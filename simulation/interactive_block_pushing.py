import pybullet as p
import torch
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from arm_pytorch_utilities import preprocess
import sklearn.preprocessing as skpre

from meta_contact import cfg
from arm_pytorch_utilities import rand, load_data

from meta_contact.controller import controller
from meta_contact.controller import global_controller
from meta_contact.controller import online_controller
from meta_contact.experiment import interactive_block_pushing
from meta_contact import prior
from meta_contact import model
from arm_pytorch_utilities.model import make

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def random_touching_start(w=interactive_block_pushing.DIST_FOR_JUST_TOUCHING):
    # w = 0.087 will be touching_wall, anything greater will not be
    init_block_pos = (np.random.random((2,)) - 0.5)
    init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
    # randomly initialize pusher adjacent to block
    # choose which face we will be next to
    along_face = (np.random.random() - 0.5) * 2 * w  # each face has 1 fixed value and 1 free value
    face = np.random.randint(0, 4)
    # for sticky environment, only have to give how far along the face
    init_pusher = np.random.uniform(-interactive_block_pushing.MAX_ALONG, interactive_block_pushing.MAX_ALONG)
    # init_pusher = interactive_block_pushing.pusher_pos_for_touching(init_block_pos, init_block_yaw, from_center=w,
    #                                                                 face=face,
    #                                                                 along_face=along_face)
    return init_block_pos, init_block_yaw, init_pusher


def collect_touching_freespace_data(trials=20, trial_length=40, level=0):
    # use random controller (with varying push direction)
    push_mag = 0.03
    # ctrl = controller.RandomController(push_mag, .02, 1)
    ctrl = controller.FullRandomController(2, (-0.01, 0), (0.01, 0.03))

    env = get_easy_env(p.DIRECT, level)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching{}'.format(level)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                                    save_dir=save_dir)
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
        ctrl = controller.FullRandomController(2, (-0.02, 0), (0.02, 0.03))
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def get_data_dir(level=0):
    return 'pushing/touching{}.mat'.format(level)


def get_easy_env(mode=p.GUI, level=0):
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
    env = interactive_block_pushing.PushAgainstWallStickyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher,
                                                             init_block=init_block_pos, init_yaw=init_block_yaw,
                                                             environment_level=level)
    return env


def test_local_dynamics(level=0):
    num_frames = 100
    # TODO preprocessor in online dynamics not yet supported
    preprocessor = None
    config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, expanded_input=True)
    # config = load_data.DataConfig(predict_difference=True, predict_all_dims=True)
    ds = interactive_block_pushing.PushDataset(data_dir=get_data_dir(level), preprocessor=preprocessor,
                                               validation_ratio=0.01, config=config)

    m = model.DeterministicUser(make.make_sequential_network(config))
    mw = model.NetworkModelWrapper(m, ds, name='contextual')
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/contextual.1000.tar'
    checkpoint = None
    # pm = prior.NNPrior.from_data(mw, checkpoint=checkpoint, train_epochs=200)
    pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    ctrl = online_controller.OnlineController(pm, ds=ds, max_timestep=num_frames, R=5, horizon=20, lqr_iter=3,
                                              init_gamma=0.1, max_ctrl=0.03)

    env = get_easy_env(p.GUI, level=level)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=num_frames, plot=True, save=False)

    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_global_linear_dynamics(level=0):
    config = load_data.DataConfig(predict_difference=False, predict_all_dims=True)
    ds = interactive_block_pushing.PushDataset(data_dir=get_data_dir(level), validation_ratio=0.01, config=config)

    u_min = [-0.02, 0]
    u_max = [0.02, 0.03]
    ctrl = global_controller.GlobalLQRController(ds, R=100, u_min=u_min, u_max=u_max)
    env = get_easy_env(p.GUI, level)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=200, plot=True, save=False)

    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_global_qr_cost_optimal_controller(controller, level=0, **kwargs):
    # preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True)
    ds = interactive_block_pushing.PushDataset(data_dir=get_data_dir(level), validation_ratio=0.1,
                                               config=config, preprocessor=preprocessor)
    pml = model.LinearModelTorch(ds)
    pm = model.NetworkModelWrapper(
        model.DeterministicUser(
            make.make_sequential_network(config, activation_factory=torch.nn.LeakyReLU, h_units=(32, 32))), ds)

    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/.200.tar'
    # checkpoint = None
    if checkpoint and pm.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        pm.learn_model(200, batch_N=10000)

    pm.freeze()

    Q = torch.diag(torch.tensor([1, 1, 0, 0.01], dtype=torch.double))
    u_min = torch.tensor([-0.02, 0], dtype=torch.double)
    u_max = torch.tensor([0.02, 0.03], dtype=torch.double)
    ctrl = controller(pm, ds, Q=Q, u_min=u_min, u_max=u_max, **kwargs)
    env = get_easy_env(p.GUI, level=level)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=200, plot=True, save=False)

    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # collect_touching_freespace_data(trials=50, trial_length=50, level=0)
    ctrl = global_controller.GlobalCEMController
    test_global_qr_cost_optimal_controller(ctrl, num_samples=1000, horizon=7, num_elite=50, level=0,
                                           init_cov_diag=0.002) # CEM options
    # ctrl = global_controller.GlobalMPPIController
    # test_global_qr_cost_optimal_controller(ctrl, num_samples=1000, horizon=7, level=0, lambda_=0.1,
    #                                        noise_sigma=torch.diag(
    #                                            torch.tensor([0.01, 0.01], dtype=torch.double)))  # MPPI options
    # test_global_linear_dynamics(level=0)
    # test_local_dynamics(0)
    # sandbox()
