import logging
import time

import numpy as np
import pybullet as p
from arm_pytorch_utilities import rand
from matplotlib import pyplot as plt

from meta_contact.controller import controller
from meta_contact.env import block_push

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


def test_simulator_friction_isometry():
    import os
    from meta_contact import cfg
    import pybullet_data
    from arm_pytorch_utilities import math_utils
    import torch

    init_block_pos = [0.0, 0.0]
    init_block_yaw = -0.
    # face = block_push.BlockFace.LEFT
    # along_face = block_push.MAX_ALONG / 1.5
    # # initializing env to visually confirm the first function works
    # env = block_push.PushAgainstWallStickyEnv(mode=p.GUI, init_pusher=along_face,
    #                                           init_block=init_block_pos, init_yaw=init_block_yaw)
    # pusher_pos = env._observe_pusher()[:2]
    # init_block = np.array((*init_block_pos, init_block_yaw))
    # predicted_along_face, from_center = block_push.pusher_pos_along_face(init_block_pos, init_block_yaw,
    #                                                                      pusher_pos,
    #                                                                      face=face)

    physics_client = p.connect(p.GUI)
    p.setTimeStep(1. / 240.)
    p.setRealTimeSimulation(False)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), tuple(init_block_pos) + (-0.02,),
                         p.getQuaternionFromEuler([0, 0, init_block_yaw]))
    planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                 cameraTargetPosition=[0, 0, 1])
    STATIC_VELOCITY_THRESHOLD = 1e-6

    def _static_environment():
        v, va = p.getBaseVelocity(blockId)
        if (np.linalg.norm(v) > STATIC_VELOCITY_THRESHOLD) or (
                np.linalg.norm(va) > STATIC_VELOCITY_THRESHOLD):
            return False
        return True

    def _observe_block():
        blockPose = p.getBasePositionAndOrientation(blockId)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
        return np.array((xb, yb, yaw))

    def get_dx(px, cx):
        dpos = cx[:2] - px[:2]
        dyaw = math_utils.angular_diff(cx[2], px[2])
        dx = torch.from_numpy(np.r_[dpos, dyaw])
        return dx

    def dx_to_dz(px, dx):
        dz = torch.zeros_like(dx)
        # dyaw is the same
        dz[2] = dx[2]
        # dz[:2] = math_utils.rotate_wrt_origin(dx[:2], px[2])
        dz[:2] = math_utils.batch_rotate_wrt_origin(dx[:2].view(1, -1), -px[2].view(1, -1))
        return dz

    # p.changeDynamics(blockId, -1, lateralFriction=0.1)
    p.changeDynamics(blockId, 0, lateralFriction=0.1)
    p.changeDynamics(planeId, 0, lateralFriction=0.1)
    F = 0.5
    for simTime in range(100):
        # observe difference from pushing
        px = _observe_block()
        p.applyExternalForce(blockId, -1, [F, F, 0], [-0.075, 0.075, 0], p.LINK_FRAME)
        p.stepSimulation()
        while not _static_environment():
            for _ in range(100):
                p.stepSimulation()
            # p.resetBaseVelocity(blockId, [0, 0, 0], [0, 0, 0])
        cx = _observe_block()
        # difference in world frame
        dx = get_dx(px, cx)
        # TODO compute difference in block frame block frame
        dz = dx_to_dz(torch.from_numpy(px), dx)
        logger.info("dx %s dz %s", dx, dz)
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
    # test_friction()
    test_simulator_friction_isometry()
    # TODO test pushing in one direction (diagonal to face); check friction cone; what angle do we start sliding
