import pybullet as p
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from hybrid_sysid.experiment import preprocess
import sklearn.preprocessing as skpre

from meta_contact import cfg
from arm_pytorch_utilities import rand, load_data

from meta_contact.controller import controller
from meta_contact.controller import baseline_prior, locally_linear
from meta_contact.experiment import interactive_block_pushing
from meta_contact.util import rotate_wrt_origin
from meta_contact.model import make_mdn_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def random_touching_start(w=0.087):
    # w = 0.087 will be touching_wall, anything greater will not be
    init_block_pos = (np.random.random((2,)) - 0.5)
    init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
    # randomly initialize pusher adjacent to block
    # choose which face we will be next to
    non_fixed_val = (np.random.random() - 0.5) * 2 * w  # each face has 1 fixed value and 1 free value
    face = np.random.randint(0, 4)
    if face == 0:  # right
        dxy = (w, non_fixed_val)
    elif face == 1:  # top
        dxy = (non_fixed_val, w)
    elif face == 2:  # left
        dxy = (-w, non_fixed_val)
    else:
        dxy = (non_fixed_val, -w)
    # rotate by yaw to match (around origin since these are differences)
    dxy = rotate_wrt_origin(dxy, init_block_yaw)
    init_pusher = np.add(init_block_pos, dxy)
    return init_block_pos, init_block_yaw, init_pusher


def collect_touching_freespace_data(trials=20, trial_length=40):
    # use random controller (with varying push direction)
    push_mag = 0.03
    ctrl = controller.RandomController(push_mag, .02, 1)
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.DIRECT)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching'
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        # init_block_pos, init_block_yaw, init_pusher = random_touching_start()
        init_block_pos = [0, 0]
        init_block_yaw = 0
        init_pusher = [-0.5, 0]
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        # ctrl = controller.RandomStraightController(push_mag, .3, init_pusher, init_block_pos)
        # sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def collect_notouch_freespace_data(trials=100, trial_length=10):
    ctrl = controller.FullRandomController(0.03)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/notouch_freespace'
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.DIRECT)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(0.4)
        init_block_pos = [0, 0]
        init_block_yaw = 0
        init_pusher = [-0.25, 0]
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def test_global_linear_dynamics():
    ctrl = baseline_prior.GlobalLQRController(1)
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.GUI)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=100, plot=True, save=False)

    seed = rand.seed(3)

    init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_global_prior_dynamics():
    mdn = make_mdn_model(num_components=3)

    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None
    ctrl = baseline_prior.GlobalNetworkCrossEntropyController(mdn, 'mdn_cem', R=1, preprocessor=preprocessor,
                                                              checkpoint='/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn.6000.tar')
    # ctrl = baseline_prior.GlobalNetworkCrossEntropyController(
    #     feature.SequentialFC(input_dim=2, feature_dim=3, hidden_units=10,
    #                          hidden_layers=3).double(), R=1)
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.GUI)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=100, plot=False, save=False)

    seed = rand.seed(2)
    # init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    init_block_pos = [0, 0]
    init_block_yaw = 0
    init_pusher = [-0.25, 0]
    goal_pos = [1.0, 0]
    env.set_task_config(goal=goal_pos, init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    plt.ioff()
    plt.show()


def get_easy_env(mode=p.GUI):
    init_block_pos = [0, 0]
    init_block_yaw = 0
    init_pusher = [-0.25, 0]
    goal_pos = [1.0, 0]
    env = interactive_block_pushing.PushAgainstWallEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher,
                                                       init_block=init_block_pos, init_yaw=init_block_yaw)
    return env


from meta_contact.prior import LSQPrior, GMMPrior
from meta_contact.controller import online_controller
import torch
from arm_pytorch_utilities import linalg


def test_local_dynamics():
    num_frames = 100

    preprocessor = None
    ds = interactive_block_pushing.PushDataset(data_dir='pushing/touching.mat', preprocessor=preprocessor,
                                               validation_ratio=0.01)
    ds.make_data()
    prior = GMMPrior.from_data(ds)
    ctrl = online_controller.OnlineController(prior, ds=ds, max_timestep=num_frames, R=1, horizon=15, lqr_iter=1)

    env = get_easy_env(p.GUI)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=num_frames, plot=True, save=False)

    seed = rand.seed(4)
    sim.run(seed)
    plt.ioff()
    plt.show()


def sandbox():
    ctrl = controller.FullRandomController(0.05)
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.GUI)
    sim = interactive_block_pushing.InteractivePush(env, ctrl, num_frames=1000, plot=False, save=False)
    sim.run()


if __name__ == "__main__":
    # collect_touching_freespace_data(trials=100, trial_length=70)
    # collect_notouch_freespace_data()
    # test_global_prior_dynamics()
    # test_global_linear_dynamics()
    test_local_dynamics()
    # sandbox()
