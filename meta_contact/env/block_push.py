import logging
import math
import os
import pybullet as p
import time
import functools
import abc

import numpy as np
import torch
from arm_pytorch_utilities import load_data as load_utils, math_utils
from arm_pytorch_utilities.make_data import datasource
from hybrid_sysid import simulation
from matplotlib import pyplot as plt
from meta_contact import cfg
from meta_contact.env.myenv import MyPybulletEnv
from meta_contact.controller import controller

logger = logging.getLogger(__name__)


class BlockFace:
    RIGHT = 0
    TOP = 1
    LEFT = 2
    BOT = 3


class ContactInfo:
    """Semantics for indices of a contact info from getContactPoints"""
    POS_A = 5
    NORMAL_DIR_B = 7
    NORMAL_MAG = 9
    LATERAL1_MAG = 10
    LATERAL1_DIR = 11
    LATERAL2_MAG = 12
    LATERAL2_DIR = 13


# TODO This is specific to this pusher and block; how to generalize this?
_MAX_ALONG = 0.075 + 0.1  # half length of block
_BLOCK_HEIGHT = 0.05
_PUSHER_MID = 0.10
DIST_FOR_JUST_TOUCHING = _MAX_ALONG + 0.021 - 0.00001


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


class DebugDrawer:
    def __init__(self):
        self._debug_ids = {}
        self._camera_pos = [0, 0]

    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, height=_BLOCK_HEIGHT):
        if name not in self._debug_ids:
            self._debug_ids[name] = [-1, -1]
        uids = self._debug_ids[name]

        # use default height if point is 2D, otherwise use point's z coordinate
        if point.shape[0] == 3:
            height = point[2]

        location = (point[0], point[1], height)
        uids[0] = p.addUserDebugLine(np.add(location, [length, 0, 0]), np.add(location, [-length, 0, 0]), color, 2,
                                     replaceItemUniqueId=uids[0])
        uids[1] = p.addUserDebugLine(np.add(location, [0, length, 0]), np.add(location, [0, -length, 0]), color, 2,
                                     replaceItemUniqueId=uids[1])

    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=_BLOCK_HEIGHT):
        if name not in self._debug_ids:
            self._debug_ids[name] = [-1, -1]
        uids = self._debug_ids[name]

        location = (pose[0], pose[1], height)
        side_lines = math_utils.rotate_wrt_origin((0, length * 0.2), pose[2])
        pointer = math_utils.rotate_wrt_origin((length, 0), pose[2])
        uids[0] = p.addUserDebugLine(np.add(location, [side_lines[0], side_lines[1], 0]),
                                     np.add(location, [-side_lines[0], -side_lines[1], 0]),
                                     color, 2, replaceItemUniqueId=uids[0])
        uids[1] = p.addUserDebugLine(np.add(location, [0, 0, 0]),
                                     np.add(location, [pointer[0], pointer[1], 0]),
                                     color, 2, replaceItemUniqueId=uids[1])

    def draw_2d_line(self, name, start, diff, color=(0, 0, 0), size=2., scale=0.4):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        self._debug_ids[name] = p.addUserDebugLine(start, np.add(start, [diff[0] * scale, diff[1] * scale,
                                                                         diff[2] * scale if len(diff) is 3 else 0]),
                                                   color, lineWidth=size, replaceItemUniqueId=uid)

    def draw_contact_point(self, name, contact, flip=True):
        start = contact[ContactInfo.POS_A]
        f_all = _get_total_contact_force(contact, flip)
        # combined normal vector (adding lateral friction)
        f_size = np.linalg.norm(f_all)
        self.draw_2d_line("{} xy".format(name), start, f_all, size=f_size, scale=0.03, color=(1, 1, 0))
        # _draw_contact_friction(line_unique_ids, contact, flip)
        return f_size

    def draw_contact_friction(self, name, contact, flip=True):
        start = list(contact[ContactInfo.POS_A])
        start[2] = _BLOCK_HEIGHT
        # friction along y
        scale = 0.1
        c = (1, 0.4, 0.7)
        fyd, fxd = _get_lateral_friction_forces(contact, flip)
        self.draw_2d_line('{}y'.format(name), start, fyd, size=np.linalg.norm(fyd), scale=scale, color=c)
        self.draw_2d_line('{}x'.format(name), start, fxd, size=np.linalg.norm(fxd), scale=scale, color=c)

    def draw_transition(self, prev_block, new_block):
        name = 't'
        if name not in self._debug_ids:
            self._debug_ids[name] = []

        self._debug_ids[name].append(
            p.addUserDebugLine([prev_block[0], prev_block[1], _BLOCK_HEIGHT],
                               (new_block[0], new_block[1], _BLOCK_HEIGHT),
                               [0, 0, 1], 2))

    def clear_transitions(self):
        name = 't'
        if name in self._debug_ids:
            for line in self._debug_ids[name]:
                p.removeUserDebugItem(line)
            self._debug_ids[name] = []

    def draw_text(self, name, text, location_index, left_offset=1):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        move_down = location_index * 0.15
        self._debug_ids[name] = p.addUserDebugText(str(text),
                                                   [self._camera_pos[0] + left_offset,
                                                    self._camera_pos[1] + 1 - move_down, 0.1],
                                                   textColorRGB=[0.5, 0.1, 0.1],
                                                   textSize=2,
                                                   replaceItemUniqueId=uid)


def _get_lateral_friction_forces(contact, flip=True):
    force_sign = -1 if flip else 1
    fy = force_sign * contact[ContactInfo.LATERAL1_MAG]
    fyd = [fy * v for v in contact[ContactInfo.LATERAL1_DIR]]
    fx = force_sign * contact[ContactInfo.LATERAL2_MAG]
    fxd = [fx * v for v in contact[ContactInfo.LATERAL2_DIR]]
    return fyd, fxd


def _get_total_contact_force(contact, flip=True):
    force_sign = -1 if flip else 1
    force = force_sign * contact[ContactInfo.NORMAL_MAG]
    dv = [force * v for v in contact[ContactInfo.NORMAL_DIR_B]]
    fyd, fxd = _get_lateral_friction_forces(contact, flip)
    f_all = [sum(i) for i in zip(dv, fyd, fxd)]
    return f_all


class PushLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, **kwargs):
        super().__init__(file_cfg, *args, **kwargs)

    def _apply_masks(self, d, x, y):
        """Handle common logic regardless of x and y"""
        r = d['reaction'][1:]
        e = d['model error'][1:]
        r = np.column_stack((r, e))

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
            r = r[1:]

            mask = mask[1:-1]
        else:
            mask = mask[:-1]

        mask = mask.reshape(-1) != 0

        xu = xu[mask]
        r = r[mask]
        y = y[mask]

        self.config.load_data_info(x, u, y, xu)
        return xu, y, r

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

        xu, y, r = self._apply_masks(d, x, y)

        return xu, y, r


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


class PushLoaderWithReaction(PushLoaderRestricted):
    """
    Include reaction force as part of state that we need to predict
    """

    def _process_file_raw_data(self, d):
        x = d['X']
        if x.shape[1] > PushWithForceDirectlyEnv.nx:
            x = x[:, :PushWithForceDirectlyEnv.nx]
        # from testing, it's better if these guys are delayed 1 time step (to prevent breaking causality)
        # ignore force in z
        r = d['reaction'][:, :2]
        if self.config.predict_difference:
            dpos = x[1:, :2] - x[:-1, :2]
            dyaw = math_utils.angular_diff_batch(x[1:, 2], x[:-1, 2])
            dalong = x[1:, 3] - x[:-1, 3]
            dr = r[1:] - r[:-1]
            y = np.concatenate((dpos, dyaw.reshape(-1, 1), dalong.reshape(-1, 1), dr), axis=1)
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        x = np.concatenate((x, r), axis=1)
        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class PushLoaderPhysicalPusherWithReaction(PushLoaderRestricted):
    def _process_file_raw_data(self, d):
        x = d['X']
        if x.shape[1] != PushPhysicallyAnyAlongEnv.nx:
            raise RuntimeError(
                "Incompatible dataset; expected nx = {} got nx = {}".format(PushPhysicallyAnyAlongEnv.nx, x.shape[1]))

        if self.config.predict_difference:
            dpos = x[1:, :2] - x[:-1, :2]
            dyaw = math_utils.angular_diff_batch(x[1:, 2], x[:-1, 2])
            dr = x[1:, 3:5] - x[:-1, 3:5]
            y = np.concatenate((dpos, dyaw.reshape(-1, 1), dr), axis=1)
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class DebugVisualization:
    FRICTION = 0
    WALL_ON_BLOCK = 1
    REACTION_ON_PUSHER = 2
    ACTION = 3
    BLOCK_ON_PUSHER = 4


class ReactionForceStrategy:
    MAX_OVER_CONTROL_STEP = 0
    MAX_OVER_MINI_STEPS = 1
    AVG_OVER_MINI_STEPS = 2
    MEDIAN_OVER_MINI_STEPS = 3


def handle_data_format_for_state_diff(state_diff):
    @functools.wraps(state_diff)
    def data_format_handler(state, other_state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(other_state.shape) == 1:
            other_state = other_state.reshape(1, -1)
        diff = state_diff(state, other_state)
        if torch.is_tensor(state):
            diff = torch.cat(diff, dim=1)
        else:
            diff = np.column_stack(diff)
        return diff

    return data_format_handler


class PushAgainstWallEnv(MyPybulletEnv):
    nu = 2
    nx = 5
    ny = 3

    @staticmethod
    def state_names():
        return ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)']

    @staticmethod
    def get_block_pose(state):
        return state[2:5]

    @staticmethod
    def get_pusher_pos(state):
        return state[0:2]

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        dyaw = math_utils.angular_diff_batch(state[:, 4], other_state[:, 4])
        dpos = state[:, :4] - other_state[:, :4]
        return dpos, dyaw.reshape(-1, 1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$']

    @staticmethod
    def get_control_bounds():
        # depends on the environment; these are the limits for StickyEnv
        u_min = np.array([-0.03, 0.03])
        u_max = np.array([0.03, 0.03])
        return u_min, u_max

    def __init__(self, goal=(1.0, 0.), init_pusher=(-0.25, 0), init_block=(0., 0.), init_yaw=0.,
                 environment_level=0, sim_step_wait=None, mini_steps=100, wait_sim_steps_per_mini_step=20,
                 max_pusher_force=20, debug_visualizations=None,
                 reaction_force_strategy=ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS, **kwargs):
        """
        :param goal:
        :param init_pusher:
        :param init_block:
        :param init_yaw:
        :param environment_level: what obstacles should show up in the environment
        :param sim_step_wait: how many seconds to wait between each sim step to show intermediate states
        (0.01 seems reasonable for visualization)
        :param mini_steps how many mini control steps to divide the control step into;
        more is better for controller and allows greater force to prevent sliding
        :param wait_sim_steps_per_mini_step how many sim steps to wait per mini control step executed;
        inversely proportional to mini_steps
        :param reaction_force_strategy how to aggregate measured reaction forces over control step into 1 value
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.level = environment_level
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.max_pusher_force = max_pusher_force
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy

        # initial config
        self.goal = None
        self.initPusherPos = None
        self.initBlockPos = None
        self.initBlockYaw = None

        self._dd = DebugDrawer()
        self._debug_visualizations = {
            DebugVisualization.FRICTION: False,
            DebugVisualization.REACTION_ON_PUSHER: True,
            DebugVisualization.WALL_ON_BLOCK: False,
            DebugVisualization.ACTION: True,
            DebugVisualization.BLOCK_ON_PUSHER: False,
        }
        if debug_visualizations is not None:
            self._debug_visualizations.update(debug_visualizations)

        # avoid the spike at the start of each mini step from rapid acceleration
        self._steps_since_start_to_get_reaction = 3
        self._clear_state_between_control_steps()

        # quadratic cost
        self.Q = np.diag([0, 0, 1, 1, 0])
        self.R = np.diag([1 for _ in range(self.nu)])

        self.set_task_config(goal, init_pusher, init_block, init_yaw)
        self._setup_experiment()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()

    def verify_dims(self):
        u_min, u_max = self.get_control_bounds()
        assert u_min.shape[0] == u_max.shape[0]
        assert u_min.shape[0] == self.nu
        assert len(self.state_names()) == self.nx
        assert len(self.control_names()) == self.nu
        assert self.Q.shape[0] == self.nx
        assert self.R.shape[0] == self.nu

    # --- initialization and task configuration
    def _clear_state_between_control_steps(self):
        self._sim_step = 0
        self._mini_step_contact = {'full': np.zeros((self.mini_steps, 2)), 'mag': np.zeros(self.mini_steps)}
        self._contact_info = {}
        self._largest_contact = {}
        self._reaction_force = np.zeros(2)

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
        self.initPusherPos = tuple(init_pusher) + (_PUSHER_MID,)

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

        # adjust dynamics for better stability
        p.changeDynamics(self.planeId, -1, lateralFriction=0.3, spinningFriction=0.025, rollingFriction=0.01)

        self.walls = []
        wall_z = 0.05
        if self.level == 0:
            pass
        elif self.level == 1:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.55, -0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.8))
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
        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

        # set robot init config
        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)

    # --- visualization (rarely overriden)
    def visualize_rollouts(self, states):
        """In GUI mode, show how the sequence of states will look like"""
        if states is None:
            return
        # assume states is iterable, so could be a bunch of row vectors
        T = len(states)
        for t in range(T):
            pose = self.get_block_pose(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_2d_pose('rx{}'.format(t), pose, (0, c, c))

    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""
        pred_pose = self.get_block_pose(predicted_state)
        c = (0.5, 0, 0.5)
        self._dd.draw_2d_pose('ep', pred_pose, c)
        pose = self.get_block_pose(self.state)
        # use size to represent error in rotation
        angle_diff = abs(math_utils.angular_diff(pred_pose[2], pose[2]))
        pose[2] = _BLOCK_HEIGHT
        # draw line from current pose to predicted pose
        self._dd.draw_2d_line('el', pose, (pred_pose - pose)[:2], c, scale=20, size=angle_diff * 50)

    def set_camera_position(self, camera_pos):
        self._dd._camera_pos = camera_pos
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                     cameraTargetPosition=[camera_pos[0], camera_pos[1], 1])

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_goal(self):
        self._dd.draw_2d_pose('goal', self.get_block_pose(self.goal))

    def _draw_state(self):
        self._dd.draw_2d_pose('state', self.get_block_pose(self.state))

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self._observe_pusher()
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    def draw_user_text(self, text, location_index=1, left_offset=1):
        if location_index is 0:
            raise RuntimeError("Can't use same location index (0) as cost")
        self._dd.draw_text('user{}'.format(location_index), text, location_index, left_offset)

    # --- set current state
    def set_state(self, state):
        assert state.shape[0] == self.nx
        prev_block_pose = p.getBasePositionAndOrientation(self.blockId)
        zb = prev_block_pose[0][2]

        block_pose = self.get_block_pose(state)
        # keep previous height rather than reset since we don't know what's the height at ground level
        p.resetBasePositionAndOrientation(self.blockId, (block_pose[0], block_pose[1], zb),
                                          p.getQuaternionFromEuler([0, 0, block_pose[2]]))

        pusher_pos = self.get_pusher_pos(state)
        p.resetBasePositionAndOrientation(self.pusherId, (pusher_pos[0], pusher_pos[1], _PUSHER_MID),
                                          p.getQuaternionFromEuler([0, 0, 0]))
        self.state = state
        self._draw_state()

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        x, y, z = self._observe_pusher()
        return np.array((x, y) + self._observe_block())

    def _observe_block(self):
        blockPose = p.getBasePositionAndOrientation(self.blockId)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
        return xb, yb, yaw

    def _observe_pusher(self):
        pusherPose = p.getBasePositionAndOrientation(self.pusherId)
        return pusherPose[0]

    def _observe_reaction_force(self):
        """Return representative reaction force for simulation steps up to current one since last control step"""
        if self.reaction_force_strategy is ReactionForceStrategy.AVG_OVER_MINI_STEPS:
            return self._mini_step_contact['full'].mean(axis=0)
        if self.reaction_force_strategy is ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS:
            median_mini_step = np.argsort(self._mini_step_contact['mag'])[self.mini_steps // 2]
            return self._mini_step_contact['full'][median_mini_step]
        if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_MINI_STEPS:
            max_mini_step = np.argmax(self._mini_step_contact['mag'])
            return self._mini_step_contact['full'][max_mini_step]
        else:
            return self._reaction_force[:2]

    def _observe_additional_info(self, info, visualize=True):
        pass

    def _observe_info(self, visualize=True):
        info = {}

        # number of wall contacts
        info['wc'] = 0
        if self.level > 0:
            for wallId in self.walls:
                contactInfo = p.getContactPoints(self.blockId, wallId)
                info['wc'] += len(contactInfo)

        # block velocity
        v, va = p.getBaseVelocity(self.blockId)
        info['bv'] = np.linalg.norm(v)
        info['bva'] = np.linalg.norm(va)

        # pusher velocity
        v, va = p.getBaseVelocity(self.pusherId)
        info['pv'] = np.linalg.norm(v)
        info['pva'] = np.linalg.norm(va)

        # block-pusher distance
        x, y, _ = self._observe_pusher()
        xb, yb, theta = self._observe_block()

        along, from_center = pusher_pos_along_face((xb, yb), theta, (x, y))
        info['pusher dist'] = from_center - DIST_FOR_JUST_TOUCHING
        info['pusher along'] = along

        self._observe_additional_info(info, visualize)
        self._sim_step += 1

        for key, value in info.items():
            if key not in self._contact_info:
                self._contact_info[key] = []
            self._contact_info[key].append(value)

    def _observe_raw_reaction_force(self, info, reaction_force, visualize=True):
        reaction_force[2] = 0
        # save reaction force
        name = 'r'
        info[name] = reaction_force
        reaction_force_size = np.linalg.norm(reaction_force)
        # see if we should save it as the reaction force for this mini-step
        mini_step, step_since_start = divmod(self._sim_step, self.wait_sim_step_per_mini_step)
        if step_since_start is self._steps_since_start_to_get_reaction:
            self._mini_step_contact['full'][mini_step] = reaction_force[:2]
            self._mini_step_contact['mag'][mini_step] = reaction_force_size
            if self.reaction_force_strategy is not ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_ON_PUSHER] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))
        # update our running count of max force
        if reaction_force_size > self._largest_contact.get(name, 0):
            self._largest_contact[name] = reaction_force_size
            self._reaction_force = reaction_force
            if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_ON_PUSHER] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))

    def _aggregate_info(self):
        info = {key: np.stack(value, axis=0) for key, value in self._contact_info.items() if len(value)}
        info['reaction'] = self._observe_reaction_force()
        info['wall_contact'] = info['wc'].max()
        return info

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        diff = self.state_difference(state, self.goal)
        diff = diff.reshape(-1)
        cost = diff @ self.Q @ diff
        done = cost < 0.04
        if action is not None:
            cost += action @ self.R @ action
        return cost, done

    def _finish_action(self, old_state, action):
        """Evaluate action after finishing it; step should not modify state after calling this"""
        self.state = np.array(self._obs())

        # track trajectory
        prev_block = self.get_block_pose(old_state)
        new_block = self.get_block_pose(self.state)
        self._dd.draw_transition(prev_block, new_block)

        # render current pose
        self._draw_state()

        cost, done = self.evaluate_cost(self.state, action)
        self._dd.draw_text('cost', '{0:.3f}'.format(cost), 0)

        # summarize information per sim step into information for entire control step
        info = self._aggregate_info()

        # prepare for next control step
        self._clear_state_between_control_steps()

        return cost, done, info

    # --- control (commonly overridden)
    def _move_pusher(self, end):
        p.changeConstraint(self.pusherConstraint, end, maxForce=self.max_pusher_force)

    def _move_and_wait(self, eePos, steps_to_wait=50):
        # execute the action
        self._move_pusher(eePos)
        p.stepSimulation()
        for _ in range(steps_to_wait):
            self._observe_info()
            p.stepSimulation()
            if self.mode is p.GUI and self.sim_step_wait:
                time.sleep(self.sim_step_wait)

    def step(self, action):
        old_state = self._obs()
        d = action
        # set end effector pose
        z = self.initPusherPos[2]
        eePos = [old_state[0] + d[0], old_state[1] + d[1], z]

        # execute the action
        self._move_and_wait(eePos)
        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

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
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        self._dd.draw_2d_pose('x0', self.get_block_pose(self.state), color=(0, 1, 0))
        return np.copy(self.state)


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

    @staticmethod
    def state_names():
        return ['x block (m)', 'y block (m)', 'block rotation (rads)', 'pusher along face (m)']

    @staticmethod
    def get_block_pose(state):
        return state[:3]

    @staticmethod
    def get_pusher_pos(state):
        along = state[3]
        pos = pusher_pos_for_touching(state[:2], state[2], from_center=DIST_FOR_JUST_TOUCHING, face=BlockFace.LEFT,
                                      along_face=along * _MAX_ALONG)
        return pos

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        dyaw = math_utils.angular_diff_batch(state[:, 2], other_state[:, 2])
        dpos = state[:, :2] - other_state[:, :2]
        dalong = state[:, 3] - other_state[:, 3]
        return dpos, dyaw.reshape(-1, 1), dalong.reshape(-1, 1)

    @staticmethod
    def control_names():
        return ['d$p$', 'd push forward (m)']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, 0])
        u_max = np.array([1, 1])
        return u_min, u_max

    def __init__(self, init_pusher=0, face=BlockFace.LEFT, **kwargs):
        # initial config
        self.face = face
        super().__init__(init_pusher=init_pusher, **kwargs)

        # quadratic cost
        self.Q = np.diag([10, 10, 0, 0.01])
        self.R = np.diag([0.01 for _ in range(self.nu)])

    def _set_goal(self, goal):
        # ignore the pusher position
        self.goal = np.array(tuple(goal) + (0.0, 0))

    def _set_init_pusher(self, init_pusher):
        pos = pusher_pos_for_touching(self.initBlockPos[:2], self.initBlockYaw, face=self.face,
                                      along_face=init_pusher * _MAX_ALONG)
        super()._set_init_pusher(pos)

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

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info


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
    MAX_FORCE = 40

    @staticmethod
    def control_names():
        return ['d$p$', 'f push magnitude', '$\\beta$ push direction']

    @staticmethod
    def get_control_bounds():
        # depends on the env to perform normalization
        u_min = np.array([-1, 0, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    def __init__(self, init_pusher=0, **kwargs):
        # initial config
        self.along = init_pusher
        super().__init__(init_pusher=init_pusher, face=BlockFace.LEFT, **kwargs)

    def _set_init_pusher(self, init_pusher):
        self.along = init_pusher
        super()._set_init_pusher(init_pusher)

    def _setup_experiment(self):
        super()._setup_experiment()
        # disable collision since we're applying a force directly on the block (pusher is for visualization for now)
        p.setCollisionFilterPair(self.pusherId, self.blockId, -1, -1, 0)

    def _draw_action(self, f_mag, f_dir_world):
        start = self._observe_pusher()
        pointer = math_utils.rotate_wrt_origin((f_mag / self.MAX_FORCE, 0), f_dir_world)
        self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=0.4)

    def _obs(self):
        xb, yb, yaw = self._observe_block()
        return np.array((xb, yb, yaw, self.along))

    def _observe_additional_info(self, info, visualize=True):
        # assume there's 4 contact points between block and plane
        info['bp'] = np.zeros((4, 2))
        # push of plane onto block
        contactInfo = p.getContactPoints(self.blockId, self.planeId)
        # get reaction force on pusher
        reaction_force = [0, 0, 0]
        for i, contact in enumerate(contactInfo):
            if visualize and self._debug_visualizations[DebugVisualization.FRICTION]:
                self._dd.draw_contact_friction('bp{}'.format(i), contact)
            info['bp'][i] = (contact[ContactInfo.LATERAL1_MAG], contact[ContactInfo.LATERAL2_MAG])
            fy, fx = _get_lateral_friction_forces(contact)
            reaction_force = [sum(i) for i in zip(reaction_force, fy, fx)]

        if self.level > 0:
            # assume at most 4 contact points
            info['bw'] = np.zeros(4)
            for wallId in self.walls:
                contactInfo = p.getContactPoints(self.blockId, wallId)
                for i, contact in enumerate(contactInfo):
                    name = 'w{}'.format(i)
                    info['bw'][i] = contact[ContactInfo.NORMAL_MAG]

                    f_contact = _get_total_contact_force(contact)
                    reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]

                    # only visualize the largest contact
                    if abs(contact[ContactInfo.NORMAL_MAG]) > abs(self._largest_contact.get(name, 0)):
                        self._largest_contact[name] = contact[ContactInfo.NORMAL_MAG]
                        if visualize and self._debug_visualizations[DebugVisualization.WALL_ON_BLOCK]:
                            self._dd.draw_contact_point(name, contact)

        self._observe_raw_reaction_force(info, reaction_force, info)

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
        # second action is push magnitude
        f_mag = max(0, action[1] * self.MAX_FORCE)
        # third option is push angle (0 being perpendicular to face)
        f_dir = np.clip(action[2], -1, 1) * self.MAX_PUSH_ANGLE
        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(f_mag, f_dir + old_state[2])

        # execute action
        ft = math.sin(f_dir) * f_mag
        fn = math.cos(f_dir) * f_mag

        # repeat action so that each resulting step can be larger
        for _ in range(self.mini_steps):
            # apply force on the left face of the block at along
            p.applyExternalForce(self.blockId, -1, [fn, ft, 0], [-_MAX_ALONG, self.along * _MAX_ALONG, 0], p.LINK_FRAME)
            p.stepSimulation()

            # also move the pusher along visually
            self._keep_pusher_adjacent()
            for t in range(self.wait_sim_step_per_mini_step):
                self._observe_info()

                p.stepSimulation()
                if self.mode is p.GUI and self.sim_step_wait:
                    time.sleep(self.sim_step_wait)

        # apply the sliding along side after the push settles down
        self.along = np.clip(old_state[3] + d_along, -1, 1)
        self._keep_pusher_adjacent()

        for _ in range(20):
            p.stepSimulation()
            if self.mode is p.GUI and self.sim_step_wait:
                time.sleep(self.sim_step_wait)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info


REACTION_Q_COST = 0.0


class PushWithForceDirectlyReactionInStateEnv(PushWithForceDirectlyEnv):
    """
    Same as before, but have reaction force inside the state
    """
    nu = 3
    nx = 6
    ny = 6

    @staticmethod
    def state_names():
        return ['$x_b$ (m)', '$y_b$ (m)', '$\\theta$ (rads)', '$p$ (m)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        dyaw = math_utils.angular_diff_batch(state[:, 2], other_state[:, 2])
        dpos = state[:, :2] - other_state[:, :2]
        dalong = state[:, 3] - other_state[:, 3]
        dreaction = state[:, 4:6] - other_state[:, 4:6]
        return dpos, dyaw.reshape(-1, 1), dalong.reshape(-1, 1), dreaction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = np.diag([10, 10, 0, 0.01, REACTION_Q_COST, REACTION_Q_COST])
        # we render this directly when rendering state so no need to double count
        self._debug_visualizations[DebugVisualization.REACTION_ON_PUSHER] = False

    def _set_goal(self, goal_pos):
        # want 0 reaction force
        self.goal = np.array(tuple(goal_pos) + (0, 0) + (0, 0))

    def visualize_prediction_error(self, predicted_state):
        super().visualize_prediction_error(predicted_state)
        self._draw_reaction_force(predicted_state[4:6], 'pr', (0.5, 0, 0.5))

    def _draw_state(self):
        super()._draw_state()
        # NOTE this is visualizing the reaction from the previous action, rather than the current action
        self._draw_reaction_force(self.state[3:5], 'sr', (0, 0, 0))

    def _obs(self):
        state = super()._obs()
        state = np.concatenate((state, self._observe_reaction_force()))
        return state


class PushPhysicallyAnyAlongEnv(PushAgainstWallStickyEnv):
    """
    Pusher in this env is abstracted and always sticks to the block; control is change in position of pusher
    in block frame, and where along the side of the block to push
    """
    nu = 3
    nx = 5
    ny = 5
    MAX_PUSH_ANGLE = math.pi / 4  # 45 degree on either side of normal
    MAX_PUSH_DIST = _MAX_ALONG / 7  # effectively how many moves of pushing straight to move a half block

    @staticmethod
    def state_names():
        return ['$x_b$ (m)', '$y_b$ (m)', '$\\theta$ (rads)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    def control_names():
        return ['$p$', 'd push distance', '$\\beta$ push angle (wrt normal)']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, 0, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        dyaw = math_utils.angular_diff_batch(state[:, 2], other_state[:, 2])
        dpos = state[:, :2] - other_state[:, :2]
        dreaction = state[:, 3:5] - other_state[:, 3:5]
        return dpos, dyaw.reshape(-1, 1), dreaction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Q = np.diag([10, 10, 0, REACTION_Q_COST, REACTION_Q_COST])

    def _set_goal(self, goal_pos):
        self.goal = np.array(tuple(goal_pos) + (0,) + (0, 0))

    def visualize_prediction_error(self, predicted_state):
        super().visualize_prediction_error(predicted_state)
        self._draw_reaction_force(predicted_state[3:5], 'pr', (0.5, 0, 0.5))

    def _draw_state(self):
        super()._draw_state()
        # NOTE this is visualizing the reaction from the previous action, rather than the instantaneous reaction
        self._draw_reaction_force(self.state[3:5], 'sr', (0, 0, 0))

    def _draw_action(self, push_dist, push_dir_world):
        start = self._observe_pusher()
        pointer = math_utils.rotate_wrt_origin((push_dist, 0), push_dir_world)
        self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=5)

    def _obs(self):
        state = np.concatenate((self._observe_block(), self._observe_reaction_force()))
        return state

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]

        contactInfo = p.getContactPoints(self.pusherId, self.blockId)
        info['npb'] = len(contactInfo)
        for i, contact in enumerate(contactInfo):
            f_contact = _get_total_contact_force(contact, False)
            reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]

            name = 'r{}'.format(i)
            if abs(contact[ContactInfo.NORMAL_MAG]) > abs(self._largest_contact.get(name, 0)):
                self._largest_contact[name] = contact[ContactInfo.NORMAL_MAG]
                if visualize and self._debug_visualizations[DebugVisualization.BLOCK_ON_PUSHER]:
                    self._dd.draw_contact_point(name, contact, False)

        self._observe_raw_reaction_force(info, reaction_force, visualize)

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        push_along = action[0] * (_MAX_ALONG * 0.98)  # avoid corner to avoid leaving contact
        push_dist = action[1] * self.MAX_PUSH_DIST
        push_dir = action[2] * self.MAX_PUSH_ANGLE

        old_state = self._obs()

        pos = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=DIST_FOR_JUST_TOUCHING, face=self.face,
                                      along_face=push_along)
        start_ee_pos = np.concatenate((pos, (self.initPusherPos[2],)))
        self._dd.draw_point('start eepos', start_ee_pos, color=(0, 0.5, 0.8))

        # first get to desired starting push position (should experience no reaction force during this move)
        # self._move_and_wait(start_ee_pos, steps_to_wait=50)
        # alternatively reset pusher (this avoids knocking the block over)
        p.resetBasePositionAndOrientation(self.pusherId, start_ee_pos, p.getQuaternionFromEuler([0, 0, 0]))

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(push_dist, push_dir + old_state[2])

        dx = np.cos(push_dir) * push_dist
        dy = np.sin(push_dir) * push_dist
        pos = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=DIST_FOR_JUST_TOUCHING - dx,
                                      face=self.face, along_face=push_along + dy)
        final_ee_pos = np.concatenate((pos, (self.initPusherPos[2],)))
        self._dd.draw_point('final eepos', final_ee_pos, color=(0, 0.5, 0.5))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(start_ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class InteractivePush(simulation.Simulation):
    def __init__(self, env: PushAgainstWallEnv, controller, num_frames=1000, save_dir='pushing',
                 terminal_cost_multiplier=1, stop_when_done=True, visualize_rollouts=True, **kwargs):

        super(InteractivePush, self).__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        env.verify_dims()
        self.mode = env.mode
        self.stop_when_done = stop_when_done
        self.visualize_rollouts = visualize_rollouts

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
        self.fd = None
        self.ad = None

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        self.ctrl.set_goal(self.env.goal)
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, self.env.nx))
        self.pred_traj = np.zeros_like(self.traj)
        self.u = np.zeros((self.num_frames, self.env.nu))
        self.reaction_force = np.zeros((self.num_frames, 2))
        self.wall_contact = np.zeros((self.num_frames,))
        self.model_error = np.zeros_like(self.traj)
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.reaction_force, self.wall_contact, self.model_error, self.time = (data[:frame] for data
                                                                                                  in
                                                                                                  (self.traj, self.u,
                                                                                                   self.reaction_force,
                                                                                                   self.wall_contact,
                                                                                                   self.model_error,
                                                                                                   self.time))

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.MPC)

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        info = None
        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            self.env.draw_user_text("{}".format(simTime), 1)

            start = time.perf_counter()

            action = self.ctrl.command(obs, info)
            # plot expected state rollouts from this point
            if self.visualize_rollouts:
                self.env.visualize_rollouts(self.ctrl.get_rollouts(obs))
            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.debug("cost %-5.2f took %.3fs done %d action %-12s obs %s", cost, time.perf_counter() - start, done,
                         np.round(action, 2), np.round(obs, 3))

            self.last_run_cost.append(cost)
            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs
            # reaction force felt as we apply this action, as observed at the start of the next time step
            self.reaction_force[simTime + 1, :] = info['reaction']
            self.wall_contact[simTime + 1] = info['wall_contact']
            if self._predicts_state():
                self.pred_traj[simTime + 1, :] = self.ctrl.prev_predicted_x.cpu().numpy()
                # model error from the previous prediction step (can only evaluate it at the current step)
                if self.ctrl.diff_predicted is not None:
                    self.model_error[simTime, :] = self.ctrl.diff_predicted.cpu().numpy()
                if self.visualize_rollouts:
                    self.env.visualize_prediction_error(self.ctrl.prev_predicted_x.view(-1).cpu().numpy())

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                self._truncate_data(simTime + 2)
                break

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        assert len(self.last_run_cost) == self.u.shape[0]

        return simulation.ReturnMeaning.SUCCESS

    def _export_data_dict(self):
        # output (1 step prediction; only need block state)
        X = self.traj
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        # need to also throw out first step if predicting reaction force since there's no previous state
        if isinstance(self.env, PushWithForceDirectlyReactionInStateEnv):
            mask[0] = 0
        mask[-1] = 0
        u_norm = np.linalg.norm(self.u, axis=1)
        # shift by 1 since the control at t-1 affects the model error at t
        u_norm = np.roll(u_norm, 1).reshape(-1, 1)
        scaled_model_error = np.divide(self.model_error, u_norm, out=np.zeros_like(self.model_error), where=u_norm != 0)
        return {'X': X, 'U': self.u, 'reaction': self.reaction_force, 'model error': self.model_error,
                'scaled model error': scaled_model_error,
                'mask': mask.reshape(-1, 1)}

    def start_plot_runs(self):
        axis_name = self.env.state_names()
        state_dim = self.traj.shape[1]
        assert state_dim == len(axis_name)
        ctrl_dim = self.u.shape[1]

        self.fig, self.axes = plt.subplots(state_dim, 1, sharex=True)
        self.fu, self.au = plt.subplots(ctrl_dim, 1, sharex=True)
        if self._predicts_state():
            self.fd, self.ad = plt.subplots(state_dim, 1, sharex=True)
        # plot of other info
        self.fo, self.ao = plt.subplots(2, 1, sharex=True)
        self.ao[0].set_ylabel('reaction magitude')
        self.ao[1].set_ylabel('wall contacts')

        for i in range(state_dim):
            self.axes[i].set_ylabel(axis_name[i])
            if self._predicts_state():
                self.ad[i].set_ylabel('d' + axis_name[i])
        for i in range(ctrl_dim):
            self.au[i].set_ylabel('$u_{}$'.format(i))

        plt.ion()
        plt.show()

    def _plot_data(self):
        if self.fig is None:
            self.start_plot_runs()
            plt.pause(0.0001)

        t = np.arange(1, self.pred_traj.shape[0])
        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i], label='true')
            if self._predicts_state():
                self.axes[i].scatter(t, self.pred_traj[1:, i], marker='*', color='k', label='predicted')
                self.ad[i].plot(self.model_error[:, i])
        self.axes[0].legend()

        mag = np.linalg.norm(self.reaction_force, axis=1)
        self.ao[0].plot(mag)
        self.ao[1].plot(self.wall_contact)

        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        plt.pause(0.0001)

    def _reset_sim(self):
        return self.env.reset()


class PushDataSource(datasource.FileDataSource):
    loader_map = {PushAgainstWallEnv: PushLoader,
                  PushAgainstWallStickyEnv: PushLoaderRestricted,
                  PushWithForceDirectlyEnv: PushLoaderRestricted,
                  PushWithForceDirectlyReactionInStateEnv: PushLoaderWithReaction,
                  PushPhysicallyAnyAlongEnv: PushLoaderPhysicalPusherWithReaction}

    def __init__(self, env, data_dir='pushing', **kwargs):
        loader = self.loader_map.get(type(env), None)
        if not loader:
            raise RuntimeError("Unrecognized data source for env {}".format(env))
        super().__init__(loader, data_dir, **kwargs)
