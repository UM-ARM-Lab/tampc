import numpy as np
import logging
import time
from meta_contact.env import myenv
from meta_contact.env import linear
from meta_contact.controller import controller
from arm_pytorch_utilities import rand

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def test_env_control():
    init_state = [-1.5, 1.5]
    goal = [2, -2]
    noise = (0.04, 0.04)
    env = linear.WaterWorld(init_state, goal, mode=myenv.Mode.GUI, process_noise=noise, max_move_step=0.01)
    max_mag = 0.3
    ctrl = controller.FullRandomController(env.nu, (-max_mag, -max_mag), (max_mag, max_mag))
    sim = linear.LinearSim(env, ctrl, num_frames=100, plot=False, save=False)
    seed = rand.seed()
    sim.run(seed)


if __name__ == "__main__":
    test_env_control()
