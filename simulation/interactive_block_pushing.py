import pybullet as p
import math
import numpy as np
import logging

from meta_contact import cfg
from arm_pytorch_utilities import rand, load_data

from meta_contact.controller import controller
from meta_contact.controller import baseline_prior
from meta_contact.experiment import interactive_block_pushing
from meta_contact.util import rotate_wrt_origin

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
    ctrl = controller.RandomController(0.03, .3, 1)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching_freespace'
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=trial_length, mode=p.DIRECT, plot=True, save=True,
                                                    config=cfg,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        init_block_pos, init_block_yaw, init_pusher = random_touching_start()
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    input('enter to finish')


def collect_notouch_freespace_data(trials=100, trial_length=10):
    ctrl = controller.FullRandomController(0.04)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/notouch_freespace'
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=trial_length, mode=p.DIRECT
                                                    , plot=True, save=True,
                                                    config=cfg,
                                                    save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(0.4)
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    input('enter to finish')


def test_global_linear_dynamics():
    ctrl = baseline_prior.GlobalLQRController(1)
    sim = interactive_block_pushing.InteractivePush(ctrl, num_frames=100, mode=p.GUI, plot=True, save=False)

    seed = rand.seed(3)
    init_block_pos, init_block_yaw, init_pusher = random_touching_start()
    sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(seed)
    input('enter to finish')


if __name__ == "__main__":
    # collect_touching_freespace_data(trial_length=50)
    # collect_notouch_freespace_data()
    test_global_linear_dynamics()
