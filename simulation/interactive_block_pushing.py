import pybullet as p
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from hybrid_sysid.experiment import preprocess
import sklearn.preprocessing as skpre

from hybrid_sysid.model import feature
from meta_contact import cfg
from arm_pytorch_utilities import rand, load_data
from hybrid_system_with_mixtures.mdn.model import MixtureDensityNetwork

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
    # w = 0.087 will be touching, anything greater will not be
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
    ctrl = controller.RandomController(push_mag, .3, 1)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching_freespace'
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=trial_length, mode=p.DIRECT, plot=False, save=True,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        # init_block_pos, init_block_yaw, init_pusher = random_touching_start()
        init_block_pos = [0,0]
        init_block_yaw = 0
        init_pusher = [-0.5, 0]
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def collect_notouch_freespace_data(trials=100, trial_length=10):
    ctrl = controller.FullRandomController(0.03)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/notouch_freespace'
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=trial_length, mode=p.DIRECT, plot=False, save=True,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(0.4)
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def test_global_linear_dynamics():
    ctrl = baseline_prior.GlobalLQRController(1)
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=100, mode=p.GUI, plot=True, save=False)

    seed = rand.seed(3)
    init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_global_prior_dynamics():
    mdn = make_mdn_model()

    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    preprocessor = None
    ctrl = baseline_prior.GlobalNetworkCrossEntropyController(mdn, 'mdn_cem', R=1, preprocessor=preprocessor,
                                                              checkpoint='/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_quasistatic_unnormalized.800.tar')
    # ctrl = baseline_prior.GlobalNetworkCrossEntropyController(
    #     feature.SequentialFC(input_dim=2, feature_dim=3, hidden_units=10,
    #                          hidden_layers=3).double(), R=1)
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=100, mode=p.GUI, plot=True, save=True)

    seed = rand.seed(4)
    init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    plt.ioff()
    plt.show()


def test_local_dynamics():
    ctrl = locally_linear.LocallyLinearLQRController()
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=1000, mode=p.GUI, plot=True, save=False)
    seed = rand.seed(4)
    init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    plt.ioff()
    plt.show()


def sandbox():
    ctrl = controller.FullRandomController(0.05)
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=1000, mode=p.GUI, plot=False, save=False)
    sim.run()


if __name__ == "__main__":
    collect_touching_freespace_data(trials=50, trial_length=100)
    # collect_notouch_freespace_data()
    # test_global_prior_dynamics()
    # test_global_linear_dynamics()
    # test_local_dynamics()
    # sandbox()
