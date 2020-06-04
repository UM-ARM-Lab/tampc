import enum
import math
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

from arm_pytorch_utilities import rand, load_data

from meta_contact import cfg
from meta_contact.env import peg_in_hole
from meta_contact.controller import controller

env_dir = None


# --- SHARED GETTERS
def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_peg = [-0.2, 0]
    hole_pos = [0.3, 0.3]

    env_opts = {
        'mode': mode,
        'hole': hole_pos,
        'init_peg': init_peg,
        'log_video': log_video,
        'environment_level': level,
    }
    env = peg_in_hole.PegFloatingGripperEnv(**env_opts)
    env_dir = 'peg/floating'
    return env


class OfflineDataCollection:
    @staticmethod
    def random_config(env):
        hole = (np.random.random((2,)) - 0.5)
        init_peg = (np.random.random((2,)) - 0.5)
        return hole, init_peg

    @staticmethod
    def freespace(trials=200, trial_length=50, mode=p.DIRECT):
        env = get_env(mode, 0)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(env_dir, level)
        sim = peg_in_hole.PegInHole(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                    stop_when_done=False, save_dir=save_dir)
        rand.seed(4)
        # randomly distribute data
        for _ in range(trials):
            seed = rand.seed()
            # start at fixed location
            hole, init_peg = OfflineDataCollection.random_config(env)
            env.set_task_config(hole=hole, init_peg=init_peg)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()


class UseTsf(enum.Enum):
    NO_TRANSFORM = 0
    COORD = 1


if __name__ == "__main__":
    level = 0
    ut = UseTsf.COORD
    OfflineDataCollection.freespace(trials=200, trial_length=50, mode=p.GUI)
