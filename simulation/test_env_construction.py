import logging
import math
import pybullet as p
import time

import numpy as np
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


from arm_pytorch_utilities import math_utils


def get_dx(px, cx):
    dpos = cx[:2] - px[:2]
    dyaw = math_utils.angular_diff(cx[2], px[2])
    dx = np.r_[dpos, dyaw]
    return dx


def dx_to_dz(px, dx):
    dz = np.zeros_like(dx)
    # dyaw is the same
    dz[2] = dx[2]
    dz[:2] = math_utils.rotate_wrt_origin(dx[:2], -px[2])
    return dz


def test_simulator_friction_isometry():
    import os
    from meta_contact import cfg
    import pybullet_data

    init_block_pos = [0.0, 0.0]
    init_block_yaw = -0.

    physics_client = p.connect(p.GUI)
    p.setTimeStep(1. / 240.)
    p.setRealTimeSimulation(False)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), tuple(init_block_pos) + (-0.02,),
                         p.getQuaternionFromEuler([0, 0, init_block_yaw]))
    planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                 cameraTargetPosition=[0, 0, 1])
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "test_invariant.mp4")

    STATIC_VELOCITY_THRESHOLD = 1e-6

    def _observe_block(blockId):
        blockPose = p.getBasePositionAndOrientation(blockId)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
        return np.array((xb, yb, yaw))

    def _static_environment():
        v, va = p.getBaseVelocity(blockId)
        if (np.linalg.norm(v) > STATIC_VELOCITY_THRESHOLD) or (
                np.linalg.norm(va) > STATIC_VELOCITY_THRESHOLD):
            return False
        return True

    p.setGravity(0, 0, -10)
    # p.changeDynamics(blockId, -1, lateralFriction=0.1)
    p.changeDynamics(planeId, -1, lateralFriction=0.5, spinningFriction=0.3, rollingFriction=0.1)
    f_mag = 1000
    f_dir = np.pi / 4
    ft = math.sin(f_dir) * f_mag
    fn = math.cos(f_dir) * f_mag

    MAX_ALONG = 0.075 + 0.2

    for _ in range(100):
        p.stepSimulation()

    N = 300
    yaws = np.zeros(N)
    z_os = np.zeros((N, 3))
    for simTime in range(N):
        # observe difference from pushing
        px = _observe_block(blockId)
        yaws[simTime] = px[2]
        # p.applyExternalForce(blockId, -1, [fn, ft, 0], [-MAX_ALONG, MAX_ALONG, 0.025], p.LINK_FRAME)
        p.applyExternalTorque(blockId, -1, [0, 0, 150], p.LINK_FRAME)
        p.stepSimulation()
        while not _static_environment():
            for _ in range(100):
                p.stepSimulation()
        cx = _observe_block(blockId)
        # difference in world frame
        dx = get_dx(px, cx)
        # difference in block frame
        dz = dx_to_dz(px, dx)
        z_os[simTime] = dz
        logger.info("dx %s dz %s", dx, dz)
        time.sleep(0.1)
    plot_and_analyze_body_frame_invariance(yaws, z_os)


def plot_and_analyze_body_frame_invariance(yaws, z_os):
    logger.info("std / abs(mean) of each body frame dimension (should be ~0)")
    logger.info(z_os.std(0) / np.abs(np.mean(z_os, 0)))
    plt.subplot(3, 1, 1)
    v = z_os[:, 2]
    plt.scatter(yaws, v)
    plt.ylabel('dyaw')
    plt.ylim(np.min(v), np.max(v))
    plt.subplot(3, 1, 2)
    v = z_os[:, 0]
    plt.scatter(yaws, v)
    plt.ylabel('dx_body')
    plt.ylim(np.min(v), np.max(v))
    plt.subplot(3, 1, 3)
    v = z_os[:, 1]
    plt.scatter(yaws, v)
    plt.xlabel('yaw')
    plt.ylabel('dy_body')
    plt.ylim(np.min(v), np.max(v))
    plt.show()


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


def tune_direct_push():
    # determine number of pushes to reach a fixed distance and make a u-turn
    max_N = 200  # definitely won't take this many steps
    init_block_pos = [0, 0]
    init_block_yaw = 0
    env = block_push.PushWithForceDirectlyEnv(mode=p.GUI, init_pusher=0,
                                              init_block=init_block_pos, init_yaw=init_block_yaw)

    env.draw_user_text('test 1-meter dash')
    ctrl = controller.PreDeterminedController([(0.0, 1, 0) for _ in range(max_N)])
    # record how many steps of pushing to reach 1m
    obs = env.reset()
    step = 0
    while True:
        action = ctrl.command(obs)
        obs, _, _, _ = env.step(action)
        block_pose = env.get_block_pose(obs)
        step += 1
        if block_pose[0] > 1:
            break

    logger.info("took %d steps to get to %f", step, block_pose[0])

    along_face = 1.0
    env.set_task_config(init_pusher=along_face)
    env.draw_user_text('test u-turn')
    ctrl = controller.PreDeterminedController([(0.0, 1, 0) for _ in range(max_N)])
    # record how many steps of pushing to make a u-turn (yaw > pi or > -pi)
    px = env.reset()
    step = 0
    u_turn_step = None
    yaws = np.zeros(max_N)
    z_os = np.zeros((max_N, 3))
    xs = np.zeros((max_N, env.nx))
    while True:
        block_pose = env.get_block_pose(px)
        yaws[step] = block_pose[2]
        xs[step] = px

        action = ctrl.command(px)
        cx, _, _, _ = env.step(action)
        block_pose = env.get_block_pose(cx)
        dx = get_dx(px, cx)
        dz = dx_to_dz(px, dx)
        z_os[step] = dz

        px = cx
        step += 1
        if u_turn_step is None and block_pose[2] > 0:
            u_turn_step = step
        # stop after making a full turn
        if step > 10 and 0 > block_pose[2] > -0.3:
            yaws = yaws[:step]
            z_os = z_os[:step]
            xs = xs[:step]
            break

    # turning clockwise so y will be negative
    turning_radius = -np.min(xs[:, 1])
    logger.info("took %d steps to to u-turn, %d steps for full turn %f turn radius", u_turn_step, step, turning_radius)
    # analyze how rotationally invariant the dynamics is
    plot_and_analyze_body_frame_invariance(yaws, z_os)


def run_direct_push():
    N = 20
    init_block_pos = [0., 0.18]
    init_block_yaw = -1.4
    env = block_push.PushWithForceDirectlyEnv(mode=p.GUI, init_pusher=0,
                                              init_block=init_block_pos, init_yaw=init_block_yaw, environment_level=0)
    # env = block_push.PushWithForceIndirectlyEnv(mode=p.GUI, init_pusher=-0,
    #                                           init_block=init_block_pos, init_yaw=init_block_yaw, environment_level=1)

    env.draw_user_text('run direct push')
    ctrl = controller.PreDeterminedController([(0.0, 1, 0.2) for _ in range(N)])
    # record how many steps of pushing to reach 1m
    obs = env.reset()
    while True:
        action = ctrl.command(obs)
        obs, _, _, _ = env.step(action)
        time.sleep(0.3)


if __name__ == "__main__":
    # test_pusher_placement_inverse()
    # test_env_control()
    # test_simulator_friction_isometry()
    # tune_direct_push()
    run_direct_push()
