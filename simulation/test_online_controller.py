import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from meta_contact.env import myenv
from meta_contact.env import linear
from meta_contact.controller import controller
from arm_pytorch_utilities import rand, load_data
from meta_contact import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def get_control_bounds():
    max_mag = 0.3
    u_min = np.array([-max_mag, -max_mag])
    u_max = np.array([max_mag, max_mag])
    return u_min, u_max


def get_env(mode=myenv.Mode.GUI):
    init_state = [-1.5, 1.5]
    goal = [2, -2]
    noise = (0.04, 0.04)
    env = linear.WaterWorld(init_state, goal, mode=mode, process_noise=noise, max_move_step=0.01)
    return env


def test_env_control():
    env = get_env(myenv.Mode.GUI)
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    sim = linear.LinearSim(env, ctrl, num_frames=100, plot=False, save=False)
    seed = rand.seed()
    sim.run(seed)


def collect_data(trials=20, trial_length=40):
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(2, u_min, u_max)

    env = get_env(myenv.Mode.DIRECT)
    save_dir = 'linear/linear0'
    sim = linear.LinearSim(env, ctrl, num_frames=trial_length, plot=False, save=True, save_dir=save_dir)

    # randomly distribute data
    min_allowed_y = 0
    for _ in range(trials):
        seed = rand.seed()
        # randomize so that our prior is accurate in one mode but not the other
        init_state = np.random.uniform((-3, min_allowed_y), (3, 3))
        env.set_task_config(init_state=init_state)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # test_env_control()
    collect_data(50, 50)
