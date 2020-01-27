import logging
import math
import os
import pybullet as p
import time

import numpy as np
import torch
from arm_pytorch_utilities import load_data as load_utils, math_utils
from arm_pytorch_utilities.make_data import datasource
from hybrid_sysid import simulation
from matplotlib import pyplot as plt
from meta_contact import cfg
from meta_contact.env.myenv import MyPybulletEnv

logger = logging.getLogger(__name__)


class BlockFace:
    RIGHT = 0
    TOP = 1
    LEFT = 2
    BOT = 3


# TODO This is specific to this pusher and block; how to generalize this?
DIST_FOR_JUST_TOUCHING = 0.096 - 0.00001 + 0.2
_MAX_ALONG = 0.075 + 0.2
_BLOCK_HEIGHT = 0.05


def pusher_pos_for_touching(block_pos, block_yaw, from_center=DIST_FOR_JUST_TOUCHING, face=BlockFace.LEFT,
                            along_face=0):
    """
    Get pusher (x,y) for it to be adjacent the face of the block
    :param block_pos: (x,y) of the block
    :param block_yaw: rotation of the block in radians
    :param from_center: how perpendicular to the face to extend out in m
    :param face: which block face to be adjacent to
    :param along_face: how far up along the given face of the block the pusher is in m
    :return:
    """
    if face == BlockFace.RIGHT:
        dxy = (from_center, along_face)
    elif face == BlockFace.TOP:
        dxy = (along_face, from_center)
    elif face == BlockFace.LEFT:
        dxy = (-from_center, along_face)
    else:
        dxy = (along_face, -from_center)

    # rotate by yaw to match (around origin since these are differences)
    dxy = math_utils.rotate_wrt_origin(dxy, block_yaw)
    pusher_pos = np.add(block_pos, dxy)
    return pusher_pos


def pusher_pos_along_face(block_pos, block_yaw, pusher_pos, face=BlockFace.LEFT):
    """
    Get how far along the given face the pusher is (the reverse of the previous function essentially)
    :param block_pos: (x,y) of the block
    :param block_yaw: rotation of the block in radians
    :param pusher_pos: (x,y) of the pusher
    :param face: which block face to be adjacent to
    :return:
    """
    dxy = np.subtract(pusher_pos, block_pos)
    # rotate back by yaw to get wrt origin
    dxy = math_utils.rotate_wrt_origin(dxy, -block_yaw)
    if face == BlockFace.RIGHT:
        from_center, along_face = dxy
    elif face == BlockFace.TOP:
        along_face, from_center = dxy
    elif face == BlockFace.LEFT:
        from_center, along_face = dxy
        from_center = - from_center
    else:
        along_face, from_center = dxy
        from_center = - from_center
    return along_face, from_center


def _draw_debug_2d_pose(line_unique_ids, pose, color=(0, 0, 0), length=0.15 / 2, height=_BLOCK_HEIGHT):
    location = (pose[0], pose[1], height)
    side_lines = math_utils.rotate_wrt_origin((0, length * 0.2), pose[2])
    pointer = math_utils.rotate_wrt_origin((length, 0), pose[2])
    # replace previous debug lines
    line_unique_ids[0] = p.addUserDebugLine(np.add(location, [side_lines[0], side_lines[1], 0]),
                                            np.add(location, [-side_lines[0], -side_lines[1], 0]),
                                            color, 2, replaceItemUniqueId=line_unique_ids[0])
    line_unique_ids[1] = p.addUserDebugLine(np.add(location, [0, 0, 0]),
                                            np.add(location, [pointer[0], pointer[1], 0]),
                                            color, 2, replaceItemUniqueId=line_unique_ids[1])


class PushLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, **kwargs):
        super().__init__(file_cfg, *args, **kwargs)

    def _apply_masks(self, d, x, y):
        """Handle common logic regardless of x and y"""
        cc = d['contact'][1:]
        u = d['U'][:-1]
        # potentially many trajectories, get rid of buffer state in between
        mask = d['mask']

        x = x[:-1]
        xu = np.column_stack((x, u))

        # pack expanded pxu into input if config allows (has to be done before masks)
        # otherwise would use cross-file data)
        if self.config.expanded_input:
            # move y down 1 row (first element can't be used)
            # (xu, pxu)
            xu = np.column_stack((xu[1:], xu[:-1]))
            y = y[1:]
            cc = cc[1:]

            mask = mask[1:-1]
        else:
            mask = mask[:-1]

        mask = mask.reshape(-1) != 0

        xu = xu[mask]
        cc = cc[mask]
        y = y[mask]

        self.config.load_data_info(x, u, y, xu)
        return xu, y, cc

    def _process_file_raw_data(self, d):
        x = d['X']

        # separate option deciding whether to predict output of pusher positions or not
        state_col_offset = 0 if self.config.predict_all_dims else 2
        if self.config.predict_difference:
            dpos = x[1:, state_col_offset:-1] - x[:-1, state_col_offset:-1]
            dyaw = math_utils.angular_diff_batch(x[1:, -1], x[:-1, -1])
            y = np.concatenate((dpos, dyaw.reshape(-1, 1)), axis=1)
        else:
            y = x[1:, state_col_offset:]

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class PushLoaderRestricted(PushLoader):
    """
    When the environment restricts the pusher to be next to the block, so that our state is
    xb, yb, yaw, along
    """

    def _process_file_raw_data(self, d):
        x = d['X']

        if self.config.predict_difference:
            dpos = x[1:, :2] - x[:-1, :2]
            dyaw = math_utils.angular_diff_batch(x[1:, 2], x[:-1, 2])
            dalong = x[1:, 3] - x[:-1, 3]
            y = np.concatenate((dpos, dyaw.reshape(-1, 1), dalong.reshape(-1, 1)), axis=1)
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class PushAgainstWallEnv(MyPybulletEnv):
    nu = 2
    nx = 5
    ny = 3

    def __init__(self, goal=(1.0, 0.), init_pusher=(-0.25, 0), init_block=(0., 0.), init_yaw=0.,
                 environment_level=0, max_move_step=0.001, **kwargs):
        super().__init__(**kwargs)
        self.initRestFrames = 50
        self.max_move_step = max_move_step
        self.level = environment_level

        # initial config
        self.goal = None
        self.initPusherPos = None
        self.initBlockPos = None
        self.initBlockYaw = None

        # debugging objects
        self._goal_debug_lines = [-1, -1]
        self._block_debug_lines = [-1, -1]
        self._traj_debug_lines = []
        self._debug_text = -1
        self._user_debug_text = -1
        self._camera_pos = None
        self.set_task_config(goal, init_pusher, init_block, init_yaw)

        # quadratic cost
        self.Q = np.diag([0, 0, 1, 1, 0])
        self.R = np.diag([1 for _ in range(self.nu)])

        self._setup_experiment()
        # start at rest
        while not self._static_environment():
            for _ in range(30):
                p.stepSimulation()
        self.state = self._obs()

    @staticmethod
    def get_control_bounds():
        # depends on the environment; these are the limits for StickyEnv
        u_min = np.array([-0.03, 0.03])
        u_max = np.array([0.03, 0.03])
        return u_min, u_max

    def set_task_config(self, goal=None, init_pusher=None, init_block=None, init_yaw=None):
        """Change task configuration; assumes only goal position is specified #TOOD relax assumption"""
        if goal is not None:
            self._set_goal(goal)
            self._draw_goal()
        if init_block is not None:
            self._set_init_block_pos(init_block)
        if init_yaw is not None:
            self._set_init_block_yaw(init_yaw)
        if init_pusher is not None:
            self._set_init_pusher(init_pusher)

    def _set_goal(self, goal):
        # ignore the pusher position
        self.goal = np.array(tuple(goal) + tuple(goal) + (0.0,))

    def _set_init_pusher(self, init_pusher):
        self.initPusherPos = tuple(init_pusher) + (0.10,)

    def _set_init_block_pos(self, init_block_pos):
        self.initBlockPos = tuple(init_block_pos) + (0.03,)

    def _set_init_block_yaw(self, init_yaw):
        self.initBlockYaw = init_yaw

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.pusherId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "pusher.urdf"), self.initPusherPos)
        self.blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), self.initBlockPos,
                                  p.getQuaternionFromEuler([0, 0, self.initBlockYaw]))

        self.walls = []
        wall_z = 0.05
        if self.level == 0:
            pass
        elif self.level == 1:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, -0.32, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
        elif self.level == 2:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-1, 0.5, wall_z],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, -0.32, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.75, 2, wall_z],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, 2, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [1.5, 0.5, wall_z],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))

        self.set_camera_position([0, 0])
        self._draw_goal()
        _draw_debug_2d_pose(self._block_debug_lines, self._get_block_pose(self._obs()))

        # set gravity
        p.setGravity(0, 0, -10)

        # set robot init config
        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)

    def set_camera_position(self, camera_pos):
        self._camera_pos = camera_pos
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                     cameraTargetPosition=[camera_pos[0], camera_pos[1], 1])

    def clear_debug_trajectories(self):
        for line in self._traj_debug_lines:
            p.removeUserDebugItem(line)
        self._traj_debug_lines = []

    def _draw_goal(self):
        _draw_debug_2d_pose(self._goal_debug_lines, self._get_goal_block_pose())

    def _get_goal_block_pose(self):
        return self.goal[2:5]

    @staticmethod
    def _get_block_pose(state):
        return state[2:5]

    def _move_pusher(self, end):
        if self.max_move_step is None:
            p.changeConstraint(self.pusherConstraint, end, maxForce=200)
        else:
            # linearly interpolate to position from current position
            start = self._observe_pusher()
            move_dir = np.subtract(end, start)
            # normalize move direction so we're moving a fixed amount each time
            moves_required = np.linalg.norm(move_dir) / self.max_move_step
            move_step = move_dir / moves_required
            while moves_required > 0:
                if moves_required <= 1:
                    this_end = end
                else:
                    this_end = np.add(start, move_step)

                p.changeConstraint(self.pusherConstraint, this_end, maxForce=300)
                for _ in range(5):
                    p.stepSimulation()

                start = this_end
                moves_required -= 1

    def _observe_block(self):
        blockPose = p.getBasePositionAndOrientation(self.blockId)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
        return xb, yb, yaw

    def _observe_pusher(self):
        pusherPose = p.getBasePositionAndOrientation(self.pusherId)
        return pusherPose[0]

    def _observe_contact(self):
        info = {'contact_force': 0, 'contact_count': 0}
        contactInfo = p.getContactPoints(self.pusherId, self.blockId)
        if len(contactInfo) > 0:
            f_c_temp = 0
            for i in range(len(contactInfo)):
                f_c_temp += contactInfo[i][9]
            info['contact_force'] = f_c_temp
            info['contact_count'] = len(contactInfo)
        return info

    STATIC_VELOCITY_THRESHOLD = 5e-5
    REACH_COMMAND_THRESHOLD = 1e-4

    def _static_environment(self):
        v, va = p.getBaseVelocity(self.blockId)
        if (np.linalg.norm(v) > self.STATIC_VELOCITY_THRESHOLD) or (
                np.linalg.norm(va) > self.STATIC_VELOCITY_THRESHOLD):
            return False
        v, va = p.getBaseVelocity(self.pusherId)
        if (np.linalg.norm(v) > self.STATIC_VELOCITY_THRESHOLD) or (
                np.linalg.norm(va) > self.STATIC_VELOCITY_THRESHOLD):
            return False
        return True

    def _reached_command(self, eePos):
        pos = self._observe_pusher()
        return (np.linalg.norm(np.subtract(pos, eePos)[:2])) < self.REACH_COMMAND_THRESHOLD

    def _obs(self):
        x, y, z = self._observe_pusher()
        return np.array((x, y) + self._observe_block())

    def _move_and_wait(self, eePos):
        # execute the action
        self._move_pusher(eePos)
        self._traj_debug_lines.append(p.addUserDebugLine(eePos, np.add(eePos, [0, 0, 0.01]), [1, 1, 0], 4))
        # handle trying to go into wall (if we don't succeed)
        # we use a force insufficient for going into the wall
        while not self._reached_command(eePos):
            for _ in range(50):
                p.stepSimulation()
        # if rest == self.initRestFrames:
        #     logger.warning("Ran out of steps push")

        # wait until simulation becomes static
        while not self._static_environment():
            for _ in range(50):
                p.stepSimulation()

    @staticmethod
    def compare_to_goal(state, goal):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(goal.shape) == 1:
            goal = goal.reshape(1, -1)
        dyaw = math_utils.angular_diff_batch(state[:, 4], goal[:, 4])
        dpos = state[:, :4] - goal[:, :4]
        if torch.tensor(state):
            diff = torch.cat((dpos, dyaw.view(-1, 1)), dim=1)
        else:
            diff = np.column_stack((dpos, dyaw.reshape(-1, 1)))
        return diff

    def evaluate_cost(self, state, action=None):
        diff = self.compare_to_goal(state, self.goal)
        diff = diff.reshape(-1)
        cost = diff @ self.Q @ diff
        done = cost < 0.04
        if action is not None:
            cost += action @ self.R @ action
        return cost, done

    def step(self, action):
        old_state = self._obs()
        d = action
        # set end effector pose
        z = self.initPusherPos[2]
        eePos = [old_state[0] + d[0], old_state[1] + d[1], z]

        # execute the action
        self._move_and_wait(eePos)

        cost, done, info = self._observe_finished_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def _observe_finished_action(self, old_state, action):
        # get the net contact force between robot and block
        info = self._observe_contact()
        self.state = np.array(self._obs())
        # track trajectory
        prev_block = self._get_block_pose(old_state)
        new_block = self._get_block_pose(self.state)
        self._traj_debug_lines.append(
            p.addUserDebugLine([prev_block[0], prev_block[1], _BLOCK_HEIGHT],
                               (new_block[0], new_block[1], _BLOCK_HEIGHT),
                               [0, 0, 1], 2))

        # render current pose
        _draw_debug_2d_pose(self._block_debug_lines, new_block)

        cost, done = self.evaluate_cost(self.state, action)

        self._debug_text = self._draw_text('{0:.3f}'.format(cost), self._debug_text, 0)

        return cost, done, info

    def draw_user_text(self, text):
        self._user_debug_text = self._draw_text(text, self._user_debug_text, 1)

    def _draw_text(self, text, text_id, location_index):
        move_down = location_index * 0.15
        return p.addUserDebugText(str(text), [self._camera_pos[0] + 1, self._camera_pos[1] + 1 - move_down, 0.1],
                                  textColorRGB=[0.5, 0.1, 0.1],
                                  textSize=2,
                                  replaceItemUniqueId=text_id)

    def reset(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pusherId, self.initPusherPos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.blockId, self.initBlockPos,
                                          p.getQuaternionFromEuler([0, 0, self.initBlockYaw]))
        # set robot init config
        if self.pusherConstraint:
            p.removeConstraint(self.pusherConstraint)
        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)
        # start at rest
        while not self._static_environment():
            for _ in range(50):
                p.stepSimulation()
        self.state = self._obs()
        return np.copy(self.state)

    @staticmethod
    def state_names():
        return ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)']


class PushAgainstWallStickyEnv(PushAgainstWallEnv):
    """
    Pusher in this env is forced to stick to the block; control is how much to slide along the side of the block and
    how much to push perpendicularly into the adjacent face
    """
    nu = 2
    nx = 4
    ny = 4
    MAX_SLIDE = 0.3
    MAX_INTO = 0.01

    def __init__(self, init_pusher=0, face=BlockFace.LEFT, **kwargs):
        # initial config
        self.face = face
        super().__init__(init_pusher=init_pusher, **kwargs)

        # quadratic cost
        self.Q = np.diag([10, 10, 0, 0.01])
        self.R = np.diag([0.01 for _ in range(self.nu)])
        assert self.Q.shape[0] == self.nx
        assert self.R.shape[0] == self.nu

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, 0])
        u_max = np.array([1, 1])
        return u_min, u_max

    def _set_goal(self, goal):
        # ignore the pusher position
        self.goal = np.array(tuple(goal) + (0.0, 0))

    def _set_init_pusher(self, init_pusher):
        pos = pusher_pos_for_touching(self.initBlockPos[:2], self.initBlockYaw, face=self.face,
                                      along_face=init_pusher * _MAX_ALONG)
        super()._set_init_pusher(pos)

    @staticmethod
    def compare_to_goal(state, goal):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(goal.shape) == 1:
            goal = goal.reshape(1, -1)
        dyaw = math_utils.angular_diff_batch(state[:, 2], goal[:, 2])
        dpos = state[:, :2] - goal[:, :2]
        dalong = state[:, 3] - goal[:, 3]
        if torch.is_tensor(state):
            diff = torch.cat((dpos, dyaw.view(-1, 1), dalong.view(-1, 1)), dim=1)
        else:
            diff = np.column_stack((dpos, dyaw.reshape(-1, 1), dalong.reshape(-1, 1)))
        return diff

    def _get_goal_block_pose(self):
        return self.goal[:3]

    @staticmethod
    def _get_block_pose(state):
        return state[:3]

    def _obs(self):
        xb, yb, yaw = self._observe_block()
        x, y, z = self._observe_pusher()
        along, from_center = pusher_pos_along_face((xb, yb), yaw, (x, y), self.face)
        # debugging to make sure we're quasi-static and adjacent to the block
        # logger.debug("dist between pusher and block %f", from_center - DIST_FOR_JUST_TOUCHING)
        return xb, yb, yaw, along / _MAX_ALONG

    def step(self, action):
        old_state = self._obs()
        # first action is difference in along
        d_along = action[0] * self.MAX_SLIDE
        # second action is how much to go into the perpendicular face (>= 0)
        d_into = max(0, action[1]) * self.MAX_INTO

        from_center = DIST_FOR_JUST_TOUCHING - d_into
        # restrict sliding of pusher along the face (never to slide off)
        along = np.clip(old_state[3] + d_along, -1, 1)
        # logger.debug("along %f dalong %f", along, d_along)
        pos = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=from_center, face=self.face,
                                      along_face=along * _MAX_ALONG)
        # set end effector pose
        z = self.initPusherPos[2]
        eePos = np.concatenate((pos, (z,)))

        # execute the action
        self._move_and_wait(eePos)

        cost, done, info = self._observe_finished_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    @staticmethod
    def state_names():
        return ['x block (m)', 'y block (m)', 'block rotation (rads)', 'pusher along face (m)']


class PushWithForceDirectlyEnv(PushAgainstWallStickyEnv):
    """
    Pusher in this env is abstracted and always sticks to the block; control is how much to slide along the side of the
    block, the magnitude of force to push with, and the angle to push wrt the block
    """
    nu = 3
    nx = 4
    ny = 4
    MAX_PUSH_ANGLE = math.pi / 4  # 45 degree on either side of normal
    MAX_SLIDE = 0.3  # can slide at most 30/200 = 15% of the face in 1 move
    MAX_FORCE = 800

    def __init__(self, init_pusher=0, **kwargs):
        # initial config
        self.along = init_pusher
        super().__init__(init_pusher=init_pusher, face=BlockFace.LEFT, **kwargs)

    @staticmethod
    def get_control_bounds():
        # depends on the env to perform normalization
        u_min = np.array([-1, 0, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    def _set_init_pusher(self, init_pusher):
        self.along = init_pusher
        super()._set_init_pusher(init_pusher)

    def _setup_experiment(self):
        super()._setup_experiment()
        # disable collision since we're applying a force directly on the block (pusher is for visualization for now)
        p.setCollisionFilterPair(self.pusherId, self.blockId, -1, -1, 0)

    def _obs(self):
        xb, yb, yaw = self._observe_block()
        return np.array((xb, yb, yaw, self.along))

    def _keep_pusher_adjacent(self):
        state = self._obs()
        pos = pusher_pos_for_touching(state[:2], state[2], from_center=DIST_FOR_JUST_TOUCHING, face=self.face,
                                      along_face=self.along * _MAX_ALONG)
        z = self.initPusherPos[2]
        eePos = np.concatenate((pos, (z,)))
        self._move_pusher(eePos)

    def step(self, action):
        old_state = self._obs()
        # normalize action such that the input can be within a fixed range
        # first action is difference in along
        d_along = action[0] * self.MAX_SLIDE
        # TODO consider having u as fn and ft
        # second action is push magnitude
        f_mag = max(0, action[1] * self.MAX_FORCE)
        # third option is push angle (0 being perpendicular to face)
        f_dir = np.clip(action[2], -1, 1) * self.MAX_PUSH_ANGLE

        # execute action
        ft = math.sin(f_dir) * f_mag
        fn = math.cos(f_dir) * f_mag
        # apply force on the left face of the block at along
        p.applyExternalForce(self.blockId, -1, [fn, ft, 0], [-_MAX_ALONG, self.along * _MAX_ALONG, 0], p.LINK_FRAME)
        p.stepSimulation()
        while not self._static_environment():
            for _ in range(20):
                # also move the pusher along visually
                self._keep_pusher_adjacent()
                for _ in range(5):
                    p.stepSimulation()

        # apply the sliding along side after the push settles down
        self.along = np.clip(old_state[3] + d_along, -1, 1)
        self._keep_pusher_adjacent()

        while not self._static_environment():
            for _ in range(20):
                p.stepSimulation()

        cost, done, info = self._observe_finished_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    @staticmethod
    def state_names():
        return ['x block (m)', 'y block (m)', 'block rotation (rads)', 'pusher along face (m)']


class InteractivePush(simulation.Simulation):
    def __init__(self, env: PushAgainstWallEnv, controller, num_frames=1000, save_dir='pushing', observation_period=1,
                 terminal_cost_multiplier=1, stop_when_done=True, **kwargs):

        super(InteractivePush, self).__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.mode = env.mode
        self.observation_period = observation_period
        self.stop_when_done = stop_when_done

        self.env = env
        self.ctrl = controller

        # keep track of last run's rewards
        self.terminal_cost_multiplier = terminal_cost_multiplier
        self.last_run_cost = []

        # plotting
        self.fig = None
        self.axes = None
        self.fu = None
        self.au = None

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        self.ctrl.set_goal(self.env.goal)
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, self.env.nx))
        self.u = np.zeros((self.num_frames, self.env.nu))
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        self.contactForce = np.zeros((self.num_frames,))
        self.contactCount = np.zeros_like(self.contactForce)
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.time, self.contactForce, self.contactCount = (data[:frame] for data in (
            self.traj, self.u, self.time, self.contactForce, self.contactCount))

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs

            start = time.perf_counter()

            action = self.ctrl.command(obs)
            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.debug("cost %.2f took %.3fs done %d action %-20s obs %s", cost, time.perf_counter() - start, done,
                         np.round(action, 2), obs)

            self.last_run_cost.append(cost)
            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs
            self.contactForce[simTime] = info['contact_force']
            self.contactCount[simTime] = info['contact_count']

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                self._truncate_data(simTime + 2)
                break

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        # confirm dynamics is as expected
        # if self.env.level == 0:
        #     xy = self.traj[:, :self.env.nu]
        #     nxy = xy + self.u
        #     du = np.linalg.norm(nxy[:-1] - xy[1:], axis=1)
        #     if np.any(du > 2e-3):
        #         logger.error(du)
        #         raise RuntimeError("Dynamics not behaving as expected")

        # contact force mask - get rid of trash in the beginning
        # self.contactForce[:300] = 0

        # compress observations
        self.u = self._compress_observation(self.u)
        self.traj = self._compress_observation(self.traj)
        self.contactForce = self._compress_observation(self.contactForce)
        self.contactCount = self._compress_observation(self.contactCount)
        assert len(self.last_run_cost) == self.u.shape[0]

        return simulation.ReturnMeaning.SUCCESS

    def _compress_observation(self, obs):
        return obs[::self.observation_period]

    def _export_data_dict(self):
        # output (1 step prediction; only need block state)
        X = self.traj
        contact_flag = self.contactCount > 0
        contact_flag = contact_flag.reshape(-1, 1)
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        mask[-1] = 0
        return {'X': X, 'U': self.u, 'contact': contact_flag, 'mask': mask.reshape(-1, 1)}

    def start_plot_runs(self):
        axis_name = self.env.state_names() + ['contact force (N)', 'contact count']
        state_dim = self.traj.shape[1] + 2
        assert state_dim == len(axis_name)
        ctrl_dim = self.u.shape[1]

        self.fig, self.axes = plt.subplots(1, state_dim, figsize=(18, 5))
        self.fu, self.au = plt.subplots(1, ctrl_dim)

        for i in range(state_dim):
            self.axes[i].set_xlabel(axis_name[i])
        for i in range(ctrl_dim):
            self.au[i].set_xlabel('$u_{}$'.format(i))

        plt.ion()
        plt.show()

    def _plot_data(self):
        if self.fig is None:
            self.start_plot_runs()
            time.sleep(0.05)

        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i])
        self.axes[self.traj.shape[1]].plot(self.contactForce)
        self.axes[self.traj.shape[1] + 1].step(self._compress_observation(self.time), self.contactCount)
        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        time.sleep(0.01)

    def _reset_sim(self):
        return self.env.reset()


class PushDataSource(datasource.FileDataSource):
    loader_map = {PushAgainstWallEnv: PushLoader,
                  PushAgainstWallStickyEnv: PushLoaderRestricted,
                  PushWithForceDirectlyEnv: PushLoaderRestricted}

    def __init__(self, env, data_dir='pushing', **kwargs):
        loader = self.loader_map.get(type(env), None)
        if not loader:
            raise RuntimeError("Unrecognized data source for env {}".format(env))
        super().__init__(loader, data_dir, **kwargs)
