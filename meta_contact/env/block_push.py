import logging
import math
import os
import pybullet as p
import time
import torch

import numpy as np
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import simulation
from meta_contact import cfg
from meta_contact.env.pybullet_env import PybulletEnv, ContactInfo, PybulletLoader, handle_data_format_for_state_diff, \
    get_total_contact_force, get_lateral_friction_forces, PybulletEnvDataSource
from meta_contact.env.pybullet_sim import PybulletSim

logger = logging.getLogger(__name__)


class BlockFace:
    RIGHT = 0
    TOP = 1
    LEFT = 2
    BOT = 3


_MAX_ALONG = 0.3 / 2  # half length of block
_BLOCK_HEIGHT = 0.05
_PUSHER_MID = 0.10
_RADIUS_GYRATION = math.sqrt(((2 * _MAX_ALONG) ** 2 + (2 * _MAX_ALONG) ** 2) / 12)
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


class PushLoader(PybulletLoader):
    @staticmethod
    def _info_names():
        return ['reaction', 'model error', 'wall contact']

    def _process_file_raw_data(self, d):
        x = d['X']

        # separate option deciding whether to predict output of pusher positions or not
        state_col_offset = 0 if self.config.predict_all_dims else 2
        if self.config.predict_difference:
            y = PushAgainstWallEnv.state_difference(x[1:], x[:-1])
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
            y = PushAgainstWallStickyEnv.state_difference(x[1:], x[:-1])
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
            y = PushPhysicallyAnyAlongEnv.state_difference(x[1:], x[:-1])
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


class PushAgainstWallEnv(PybulletEnv):
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
    def get_pusher_pos(state, action=None):
        return state[0:2]

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        dyaw = math_utils.angular_diff_batch(state[:, 4], other_state[:, 4])
        dpos = state[:, :4] - other_state[:, :4]
        return dpos, dyaw.reshape(-1, 1)

    @classmethod
    def state_cost(cls):
        return np.diag([0, 0, 1, 1, 0])

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$']

    @staticmethod
    def get_control_bounds():
        # depends on the environment; these are the limits for StickyEnv
        u_min = np.array([-0.03, 0.03])
        u_max = np.array([0.03, 0.03])
        return u_min, u_max

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, goal=(1.0, 0.), init_pusher=(-0.25, 0), init_block=(0., 0.), init_yaw=0.,
                 environment_level=0, sim_step_wait=None, mini_steps=100, wait_sim_steps_per_mini_step=20,
                 max_pusher_force=20, debug_visualizations=None, state_cost_for_done=0.06,
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
        super().__init__(**kwargs, default_debug_height=_BLOCK_HEIGHT)
        self.level = environment_level
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.max_pusher_force = max_pusher_force
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy
        self.state_cost_for_done = state_cost_for_done

        # initial config
        self.goal = None
        self.initPusherPos = None
        self.initBlockPos = None
        self.initBlockYaw = None

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
        self._steps_since_start_to_get_reaction = 5
        self._clear_state_between_control_steps()

        self.set_task_config(goal, init_pusher, init_block, init_yaw)
        self._setup_experiment()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()

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
        p.changeDynamics(self.planeId, -1, lateralFriction=0.3, spinningFriction=0.0, rollingFriction=0.0)

        self.walls = []
        wall_z = 0.05
        if self.level == 0:
            pass
        elif self.level in [1, 4]:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.55, -0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.8))
        elif self.level == 2:
            translation = 10
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.55 + translation, -0.25 + translation, wall_z],
                           p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                           globalScaling=0.8))
        elif self.level == 3:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.3, 0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, -math.pi / 4]), useFixedBase=True,
                                         globalScaling=0.8))
        elif self.level == 5:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.5, 0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 4]), useFixedBase=True,
                                         globalScaling=0.8))
        elif self.level == 6:
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

        if self.level == 2:
            self.set_camera_position([10, 10])
        else:
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
        # clear previous rollout buffer
        self._dd.clear_visualization_after('rx', T)

    def visualize_goal_set(self, states):
        if states is None:
            return
        T = len(states)
        for t in range(T):
            pose = self.get_block_pose(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_2d_pose('gs{}'.format(t), pose, (c, c, c))
        self._dd.clear_visualization_after('gs', T)

    def visualize_trap_set(self, trap_set):
        if trap_set is None:
            return
        T = len(trap_set)
        for t in range(T):
            state, action = trap_set[t]
            pose = self.get_block_pose(state)
            c = (t + 1) / (T + 1)
            self._dd.draw_2d_pose('ts{}'.format(t), pose, (1, 0, c))
            self._draw_action(action.cpu().numpy(), old_state=state.cpu().numpy(), debug=t + 1)
        self._dd.clear_visualization_after('ts', T)
        self._dd.clear_visualization_after('u', T + 1)

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

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_goal(self):
        self._dd.draw_2d_pose('goal', self.get_block_pose(self.goal))

    def _draw_state(self):
        self._dd.draw_2d_pose('state', self.get_block_pose(self.state))

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self._observe_pusher()
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    def draw_user_text(self, text, location_index=1, left_offset=1.0):
        if location_index is 0:
            raise RuntimeError("Can't use same location index (0) as cost")
        self._dd.draw_text('user{}_{}'.format(location_index, left_offset), text, location_index, left_offset)

    # --- set current state
    def set_state(self, state, action=None, block_id=None):
        assert state.shape[0] == self.nx
        if block_id is None:
            block_id = self.blockId
        prev_block_pose = p.getBasePositionAndOrientation(self.blockId)
        zb = prev_block_pose[0][2]

        block_pose = self.get_block_pose(state)
        # keep previous height rather than reset since we don't know what's the height at ground level
        p.resetBasePositionAndOrientation(block_id, (block_pose[0], block_pose[1], zb),
                                          p.getQuaternionFromEuler([0, 0, block_pose[2]]))

        self.state = state
        self._draw_state()
        if action is not None:
            pusher_pos = self.get_pusher_pos(state, action)
            p.resetBasePositionAndOrientation(self.pusherId, (pusher_pos[0], pusher_pos[1], _PUSHER_MID),
                                              p.getQuaternionFromEuler([0, 0, 0]))
            self._draw_action(action, old_state=state)

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
        done = cost < self.state_cost_for_done
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
    def get_pusher_pos(state, action=None):
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
    def state_distance(state_difference):
        state_difference[:, 2] *= _RADIUS_GYRATION
        return state_difference[:, :3].norm(dim=1)

    @classmethod
    def state_cost(cls):
        return np.diag([10, 10, 0, 0.01])

    @staticmethod
    def control_names():
        return ['d$p$', 'd push forward (m)']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, 0])
        u_max = np.array([1, 1])
        return u_min, u_max

    @classmethod
    def control_cost(cls):
        return np.diag([0.01 for _ in range(cls.nu)])

    def __init__(self, init_pusher=0, face=BlockFace.LEFT, **kwargs):
        # initial config
        self.face = face
        super().__init__(init_pusher=init_pusher, **kwargs)

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
    MAX_FORCE = 20

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

    def _draw_action(self, action, old_state=None, debug=0):
        old_state = self._obs()
        d_along, f_mag, f_dir = self._unpack_action(action)
        f_dir_world = f_dir + old_state[2]
        start = self._observe_pusher()
        start[2] = _BLOCK_HEIGHT
        pointer = math_utils.rotate_wrt_origin((f_mag / self.MAX_FORCE, 0), f_dir_world)
        if debug:
            self._dd.draw_2d_line('u{}'.format(debug), start, pointer, (1, debug / 10, debug / 10), scale=0.4)
        else:
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
            fy, fx = get_lateral_friction_forces(contact)
            reaction_force = [sum(i) for i in zip(reaction_force, fy, fx)]

        if self.level > 0:
            # assume at most 4 contact points
            info['bw'] = np.zeros(4)
            for wallId in self.walls:
                contactInfo = p.getContactPoints(self.blockId, wallId)
                for i, contact in enumerate(contactInfo):
                    name = 'w{}'.format(i)
                    info['bw'][i] = contact[ContactInfo.NORMAL_MAG]

                    f_contact = get_total_contact_force(contact)
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

    def _unpack_action(self, action):
        # normalize action such that the input can be within a fixed range
        # first action is difference in along
        d_along = action[0] * self.MAX_SLIDE
        # second action is push magnitude
        f_mag = max(0, action[1] * self.MAX_FORCE)
        # third option is push angle (0 being perpendicular to face)
        f_dir = np.clip(action[2], -1, 1) * self.MAX_PUSH_ANGLE
        return d_along, f_mag, f_dir

    def step(self, action):
        old_state = self._obs()
        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)

        d_along, f_mag, f_dir = self._unpack_action(action)

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

    @classmethod
    def state_cost(cls):
        return np.diag([10, 10, 0, 0.01, REACTION_Q_COST, REACTION_Q_COST])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    MAX_PUSH_DIST = _MAX_ALONG / 4  # effectively how many moves of pushing straight to move a half block

    @staticmethod
    def state_names():
        return ['$x_b$ (m)', '$y_b$ (m)', '$\\theta$ (rads)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    def get_pusher_pos(state, action=None):
        along = 0 if action is None else action[0]
        pos = pusher_pos_for_touching(state[:2], state[2], from_center=DIST_FOR_JUST_TOUCHING, face=BlockFace.LEFT,
                                      along_face=along * _MAX_ALONG)
        return pos

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        dyaw = math_utils.angular_diff_batch(state[:, 2], other_state[:, 2])
        dpos = state[:, :2] - other_state[:, :2]
        dreaction = state[:, 3:5] - other_state[:, 3:5]
        return dpos, dyaw.reshape(-1, 1), dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([10, 10, 0, REACTION_Q_COST, REACTION_Q_COST])

    @staticmethod
    def control_names():
        return ['$p$', 'd push distance', '$\\beta$ push angle (wrt normal)']

    @classmethod
    def control_cost(cls):
        return np.diag([0, 1, 0])

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, 0, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def control_similarity(u1, u2):
        # TODO should probably keep the API numpy only
        u1 = torch.stack((u1[:, 0], u1[:, 2]), dim=1)
        u2 = torch.stack((u2[:, 0], u2[:, 2]), dim=1)
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    def _set_goal(self, goal_pos):
        self.goal = np.array(tuple(goal_pos) + (0,) + (0, 0))

    def visualize_prediction_error(self, predicted_state):
        super().visualize_prediction_error(predicted_state)
        self._draw_reaction_force(predicted_state[3:5], 'pr', (0.5, 0, 0.5))

    def _draw_state(self):
        super()._draw_state()
        # NOTE this is visualizing the reaction from the previous action, rather than the instantaneous reaction
        self._draw_reaction_force(self.state[3:5], 'sr', (0, 0, 0))

    def _draw_action(self, action, old_state=None, debug=0):
        if old_state is None:
            old_state = self._obs()
        push_along, push_dist, push_dir = self._unpack_action(action)
        start = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=DIST_FOR_JUST_TOUCHING, face=self.face,
                                        along_face=push_along)
        start = np.concatenate((start, (_BLOCK_HEIGHT,)))
        pointer = math_utils.rotate_wrt_origin((push_dist, 0), push_dir + old_state[2])
        if debug:
            self._dd.draw_2d_line('u{}'.format(debug), start, pointer, (1, debug / 30, debug / 10), scale=5)
        else:
            self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=5)

    def _obs(self):
        state = np.concatenate((self._observe_block(), self._observe_reaction_force()))
        return state

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]

        contactInfo = p.getContactPoints(self.pusherId, self.blockId)
        info['npb'] = len(contactInfo)
        for i, contact in enumerate(contactInfo):
            f_contact = get_total_contact_force(contact, False)
            reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]

            name = 'r{}'.format(i)
            if abs(contact[ContactInfo.NORMAL_MAG]) > abs(self._largest_contact.get(name, 0)):
                self._largest_contact[name] = contact[ContactInfo.NORMAL_MAG]
                if visualize and self._debug_visualizations[DebugVisualization.BLOCK_ON_PUSHER]:
                    self._dd.draw_contact_point(name, contact, False)

        self._observe_raw_reaction_force(info, reaction_force, visualize)

    def _unpack_action(self, action):
        push_along = action[0] * (_MAX_ALONG * 0.98)  # avoid corner to avoid leaving contact
        push_dist = action[1] * self.MAX_PUSH_DIST
        push_dir = action[2] * self.MAX_PUSH_ANGLE
        return push_along, push_dist, push_dir

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = self._obs()
        push_along, push_dist, push_dir = self._unpack_action(action)

        pos = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=DIST_FOR_JUST_TOUCHING, face=self.face,
                                      along_face=push_along)
        start_ee_pos = np.concatenate((pos, (self.initPusherPos[2],)))
        self._dd.draw_point('start eepos', start_ee_pos, color=(0, 0.5, 0.8))

        # first get to desired starting push position (should experience no reaction force during this move)
        # self._move_and_wait(start_ee_pos, steps_to_wait=50)
        # alternatively reset pusher (this avoids knocking the block over)
        p.resetBasePositionAndOrientation(self.pusherId, start_ee_pos, p.getQuaternionFromEuler([0, 0, 0]))

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)

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


class InteractivePush(PybulletSim):
    def __init__(self, env: PushAgainstWallEnv, ctrl, save_dir='pushing', **kwargs):
        super(InteractivePush, self).__init__(env, ctrl, save_dir=save_dir, **kwargs)

    def _setup_experiment(self):
        self.ctrl.set_goal(self.env.goal)
        return simulation.ReturnMeaning.SUCCESS


class PushDataSource(PybulletEnvDataSource):
    loader_map = {PushAgainstWallEnv: PushLoader,
                  PushAgainstWallStickyEnv: PushLoaderRestricted,
                  PushWithForceDirectlyEnv: PushLoaderRestricted,
                  PushWithForceDirectlyReactionInStateEnv: PushLoaderWithReaction,
                  PushPhysicallyAnyAlongEnv: PushLoaderPhysicalPusherWithReaction, }

    @staticmethod
    def _default_data_dir():
        return "pushing"

    @staticmethod
    def _loader_map(env_type):
        return PushDataSource.loader_map.get(env_type, None)
