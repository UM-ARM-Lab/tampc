import pybullet as p
import numpy as np
import logging
import time
from meta_contact.env import block_push
from meta_contact.controller import controller
from arm_pytorch_utilities import rand
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def test_pusher_placement_inverse():
    init_block_pos = [0.2, 0.3]
    init_block_yaw = -0.4
    face = block_push.BlockFace.LEFT
    along_face = 0.075
    from_center = 0.096
    init_pusher = block_push.pusher_pos_for_touching(init_block_pos, init_block_yaw,
                                                     face=face, from_center=from_center,
                                                     along_face=along_face)
    # initializing env to visually confirm the first function works
    env = block_push.PushAgainstWallEnv(mode=p.GUI, init_pusher=init_pusher,
                                        init_block=init_block_pos, init_yaw=init_block_yaw)
    pusher_pos = env._observe_pusher()
    init_block = np.array((*init_block_pos, init_block_yaw))
    predicted_along_face, from_center = block_push.pusher_pos_along_face(init_block_pos, init_block_yaw,
                                                                         init_pusher,
                                                                         face=face)
    action = np.array([0, 0])
    env.step(action)
    logger.info("along set %f calculated %f", along_face, predicted_along_face)
    logger.info("pos set %s calculated %s", init_pusher, pusher_pos)
    logger.info("block set %s resulting %s", init_block, env._observe_block())
    logger.info("along error %f", np.linalg.norm(along_face - predicted_along_face))
    logger.info("block error %f", np.linalg.norm(init_block - env._observe_block()))
    for simTime in range(100):
        env.step(action)
        time.sleep(0.1)


def test_env_control():
    init_block_pos = [0, 0]
    init_block_yaw = 0
    face = block_push.BlockFace.LEFT
    along_face = 0
    env = block_push.PushAgainstWallStickyEnv(mode=p.GUI, init_pusher=along_face, face=face,
                                              init_block=init_block_pos, init_yaw=init_block_yaw)
    ctrl = controller.FullRandomController(2, (-0.01, 0), (0.01, 0.03))
    sim = block_push.InteractivePush(env, ctrl, num_frames=100, plot=False, save=False)
    seed = rand.seed()
    sim.run(seed)


def test_friction():
    init_block_pos = [0, 0]
    init_block_yaw = 2
    face = block_push.BlockFace.LEFT
    along_face = block_push.MAX_ALONG * -0.5
    env = block_push.PushAgainstWallStickyEnv(mode=p.GUI, init_pusher=along_face, face=face,
                                              init_block=init_block_pos, init_yaw=init_block_yaw)
    num_frames = 50
    ctrl = controller.PreDeterminedController([(0.0, 0.02) for _ in range(num_frames)])
    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=True, save=False)
    seed = rand.seed()
    sim.run(seed)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # test_pusher_placement_inverse()
    # test_env_control()
    test_friction()
    # TODO test pushing in one direction (diagonal to face); check friction cone; what angle do we start sliding
