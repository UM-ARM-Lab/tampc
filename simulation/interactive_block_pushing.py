import pybullet as p
import math
import numpy as np
import logging

from meta_contact import cfg
from arm_pytorch_utilities import rand, load_data

from meta_contact.controller.controller import RandomController
from meta_contact.experiment.interactive_block_pushing import InteractivePush
from meta_contact.util import rotate_wrt_origin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def collect_touching_freespace_data(trials=20, trial_length=40):
    # use random controller (with varying push direction)
    ctrl = RandomController(0.03, .3, 1)
    # use mode p.GUI to see what the trials look like
    save_dir = 'pushing/touching_freespace'
    sim = InteractivePush(ctrl, num_frames=trial_length, mode=p.DIRECT, plot=True, save=True, config=cfg,
                          save_dir=save_dir)
    for _ in range(trials):
        seed = rand.seed()
        init_block_pos = (np.random.random((2,)) - 0.5)
        init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
        # randomly initialize pusher adjacent to block
        # choose which face we will be next to
        w = 0.087
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
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
    load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    input('enter to finish')


if __name__ == "__main__":
    collect_touching_freespace_data(trial_length=50)
