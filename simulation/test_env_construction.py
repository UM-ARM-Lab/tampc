import pybullet as p
import numpy as np
import logging
import time
from meta_contact.experiment import interactive_block_pushing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def test_pusher_placement_inverse():
    init_block_pos = [-0.4, 0.3]
    init_block_yaw = 1.2
    face = interactive_block_pushing.BlockFace.LEFT
    along_face = -0.04
    from_center = 0.096
    init_pusher = interactive_block_pushing.pusher_pos_for_touching(init_block_pos, init_block_yaw,
                                                                    face=face, from_center=from_center,
                                                                    along_face=along_face)
    # initializing env to visually confirm the first function works
    env = interactive_block_pushing.PushAgainstWallEnv(mode=p.GUI, init_pusher=init_pusher,
                                                       init_block=init_block_pos, init_yaw=init_block_yaw)
    pusher_pos = env._observe_pusher()
    init_block = np.array((*init_block_pos, init_block_yaw))
    predicted_along_face = interactive_block_pushing.pusher_pos_along_face(init_block_pos, init_block_yaw, init_pusher,
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


if __name__ == "__main__":
    test_pusher_placement_inverse()
