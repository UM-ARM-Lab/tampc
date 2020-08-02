import logging
import math
import os
import pybullet as p
import time
import enum
import torch

import numpy as np
from tampc import cfg
from tampc.env.pybullet_env import PybulletEnv, PybulletLoader, handle_data_format_for_state_diff, \
    get_total_contact_force, ContactInfo, PybulletEnvDataSource
from tampc.env.pybullet_sim import PybulletSim

logger = logging.getLogger(__name__)

_PEG_MID = 0.075
_BOARD_TOP = 0.05
_EE_PEG_Z_DIFF = 0.12

_DIR = "peg"

pandaEndEffectorIndex = 11
pandaNumDofs = 7


class PegLoader(PybulletLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if x.shape[1] != PegInHoleEnv.nx:
            raise RuntimeError(
                "Incompatible dataset; expected nx = {} got nx = {}".format(PegInHoleEnv.nx, x.shape[1]))

        if self.config.predict_difference:
            y = PegInHoleEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class DebugVisualization(enum.IntEnum):
    FRICTION = 0
    WALL_ON_BLOCK = 1
    REACTION_ON_PUSHER = 2
    ACTION = 3
    BLOCK_ON_PUSHER = 4
    REACTION_IN_STATE = 5


class ReactionForceStrategy(enum.IntEnum):
    MAX_OVER_CONTROL_STEP = 0
    MAX_OVER_MINI_STEPS = 1
    AVG_OVER_MINI_STEPS = 2
    MEDIAN_OVER_MINI_STEPS = 3


class PandaGripperID(enum.IntEnum):
    FINGER_A = 9
    FINGER_B = 10


class PandaJustGripperID(enum.IntEnum):
    FINGER_A = 0
    FINGER_B = 1


class PegInHoleEnv(PybulletEnv):
    nu = 2
    nx = 5
    MAX_FORCE = 5 * 240
    MAX_GRIPPER_FORCE = 20
    MAX_PUSH_DIST = 0.03
    FINGER_OPEN = 0.04
    FINGER_CLOSED = 0.01

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        dreaction = state[:, 3:5] - other_state[:, 3:5]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 0, 0, 0])

    @staticmethod
    def state_distance(state_difference):
        return state_difference[:, :2].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1])
        u_max = np.array([1, 1])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def control_similarity(u1, u2):
        # TODO should probably keep the API numpy only
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, hole=(0.0, 0.0), init_peg=(-0.3, 0.),
                 environment_level=0, sim_step_wait=None, mini_steps=50, wait_sim_steps_per_mini_step=20,
                 debug_visualizations=None, dist_for_done=0.02,
                 reaction_force_strategy=ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS, **kwargs):
        """
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
        super().__init__(**kwargs, default_debug_height=_PEG_MID, camera_dist=1.0)
        self.level = environment_level
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy
        self.dist_for_done = dist_for_done

        # initial config
        self.hole = None
        self.initPeg = None
        self.armId = None
        self.boardId = None

        self._debug_visualizations = {
            DebugVisualization.FRICTION: False,
            DebugVisualization.REACTION_ON_PUSHER: False,
            DebugVisualization.WALL_ON_BLOCK: False,
            DebugVisualization.ACTION: True,
            DebugVisualization.BLOCK_ON_PUSHER: False,
            DebugVisualization.REACTION_IN_STATE: False,
        }
        if debug_visualizations is not None:
            self._debug_visualizations.update(debug_visualizations)

        # avoid the spike at the start of each mini step from rapid acceleration
        self._steps_since_start_to_get_reaction = 5
        self._clear_state_between_control_steps()

        self.set_task_config(hole, init_peg)
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

    def set_task_config(self, hole=None, init_peg=None):
        """Change task configuration; assumes only goal position is specified #TOOD relax assumption"""
        if hole is not None:
            self._set_hole(hole)
        if init_peg is not None:
            self._set_init_peg(init_peg)

    def _set_hole(self, hole):
        # ignore the pusher position
        self.hole = np.array(hole)
        if self.boardId is not None:
            p.resetBasePositionAndOrientation(self.boardId, [self.hole[0], self.hole[1], 0],
                                              p.getQuaternionFromEuler([0, 0, 0]))

    def _set_init_peg(self, init_peg):
        self.initPeg = tuple(init_peg) + (_PEG_MID,)
        if self.armId is not None:
            self._calculate_init_joints_to_hold_peg()

    def _calculate_init_joints_to_hold_peg(self):
        self.initJoints = list(p.calculateInverseKinematics(self.armId,
                                                            self.endEffectorIndex,
                                                            [self.initPeg[0], self.initPeg[1] + 0.025, self._ee_z],
                                                            self.endEffectorOrientation))

    def _open_gripper(self):
        p.resetJointState(self.armId, PandaGripperID.FINGER_A, self.FINGER_OPEN)
        p.resetJointState(self.armId, PandaGripperID.FINGER_B, self.FINGER_OPEN)

    def _close_gripper(self):
        p.setJointMotorControlArray(self.armId,
                                    [PandaGripperID.FINGER_A, PandaGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.FINGER_CLOSED, self.FINGER_CLOSED],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def _setup_ee(self):
        p.resetBasePositionAndOrientation(self.pegId, self.initPeg, [0, 0, 0, 1])
        # wait for peg to fall
        for _ in range(1000):
            p.stepSimulation()

        pegPos = p.getBasePositionAndOrientation(self.pegId)[0]
        self._ee_z = pegPos[2] + _EE_PEG_Z_DIFF
        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.pegId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "peg.urdf"), self.initPeg)
        # add board with hole
        self.boardId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "hole_board.urdf"), [self.hole[0], self.hole[1], 0],
                                  useFixedBase=True)

        self._setup_ee()
        self._setup_gripper()

        self.walls = []
        wall_z = _BOARD_TOP / 2
        if self.level == 0:
            pass
        elif self.level == 1:
            # add protrusions to board
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.1, -0.07, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.1, 0.07, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.1, 0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.02, 0.35, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.16, 0.35, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))

        elif self.level == 2:
            self._set_hole([0, 0.2])
            # a "well" around the hole
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0., 0.1, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.1, 0.2, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.1, 0.2, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))

        elif self.level in [3, 4]:
            width = 0.037
            y = 0.21
            self._set_hole([0, 0.2])
            # a "well" around the hole
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [0., 0.16, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.07))
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [-width, y, wall_z],
                           p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                           globalScaling=0.06))
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [width, y, wall_z],
                           p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                           globalScaling=0.06))

        elif self.level is 5:
            self._set_hole([0, 0.2])
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0., 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.12, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.12, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.17, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.17, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))

        elif self.level is 6:
            self._set_hole([-0.04, 0.125])
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0., 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.15, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.15, 0.17, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.06))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, 0.075, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.06))
        elif self.level is 7:
            translation = 10
            self._set_hole([-0.04 + translation, 0.125 + translation])
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0. + translation, 0.17 + translation, wall_z],
                           p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                           globalScaling=0.06))
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.15 + translation, 0.17 + translation, wall_z],
                           p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                           globalScaling=0.06))
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.15 + translation, 0.17 + translation, wall_z],
                           p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                           globalScaling=0.06))
            self.walls.append(
                p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0 + translation, 0.075 + translation, wall_z],
                           p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                           globalScaling=0.06))
        elif self.level == 10:
            self._set_hole([0, 0.2])
            # a "well" around the hole
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [0., 0.09, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.1))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [-0.03, 0.09, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.05))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "short_wall.urdf"), [0.03, 0.09, wall_z],
                                         p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                         globalScaling=0.05))

        for wallId in self.walls:
            p.changeVisualShape(wallId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])

        if self.level == 7:
            self.set_camera_position([10, 10])
        else:
            self.set_camera_position([0, 0])
        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

    def _setup_gripper(self):
        # add kuka arm
        # self.armId = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
        self.armId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        p.resetBasePositionAndOrientation(self.armId, [0, -0.500000, 0.070000],
                                          p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        for j in range(p.getNumJoints(self.armId)):
            p.changeDynamics(self.armId, j, linearDamping=0, angularDamping=0)
        # orientation of the end effector (pointing down)
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.endEffectorIndex = pandaEndEffectorIndex
        self.numJoints = p.getNumJoints(self.armId)
        # get the joint ids
        self.armInds = [i for i in range(pandaNumDofs)]

        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.armId,
                               9,
                               self.armId,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # joint damping coefficents
        # self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]

        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        self._open_gripper()
        self._close_gripper()

    def visualize_rollouts(self, states):
        """In GUI mode, show how the sequence of states will look like"""
        if states is None:
            return
        # assume states is iterable, so could be a bunch of row vectors
        T = len(states)
        for t in range(T):
            pos = self.get_ee_pos(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_point('rx{}'.format(t), pos, (0, c, c))
        self._dd.clear_visualization_after('rx', T)

    def visualize_goal_set(self, states):
        if states is None:
            return
        T = len(states)
        for t in range(T):
            pos = self.get_ee_pos(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_point('gs{}'.format(t), pos, (c, c, c))
        self._dd.clear_visualization_after('gs', T)

    def visualize_trap_set(self, trap_set):
        if trap_set is None:
            return
        T = len(trap_set)
        for t in range(T):
            state, action = trap_set[t]
            pose = self.get_ee_pos(state)
            c = (t + 1) / (T + 1)
            self._dd.draw_point('ts{}'.format(t), pose, (1, 0, c))
            self._draw_action(action.cpu().numpy(), old_state=state.cpu().numpy(), debug=t + 1)
        self._dd.clear_visualization_after('ts', T)
        self._dd.clear_visualization_after('u', T + 1)

    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""
        pred = self.get_ee_pos(predicted_state)
        c = (0.5, 0, 0.5)
        self._dd.draw_point('ep', pred, c)

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_state(self):
        self._dd.draw_point('state', self.get_ee_pos(self.state))
        if self._debug_visualizations[DebugVisualization.REACTION_IN_STATE]:
            self._draw_reaction_force(self.state[3:5], 'sr', (0, 0, 0))

    def _draw_action(self, action, old_state=None, debug=0):
        if old_state is None:
            old_state = self._obs()
        start = old_state[:3]
        start[2] = _PEG_MID
        pointer = np.concatenate((action, (0,)))
        if debug:
            self._dd.draw_2d_line('u{}'.format(debug), start, pointer, (1, debug / 30, debug / 10), scale=0.2)
        else:
            self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=0.2)

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self._observe_ee()
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    def draw_user_text(self, text, location_index=1, left_offset=1):
        if location_index is 0:
            raise RuntimeError("Can't use same location index (0) as cost")
        self._dd.draw_text('user{}'.format(location_index), text, location_index, left_offset)

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = np.concatenate((self._observe_ee(), self._observe_reaction_force()))
        return state

    def _observe_ee(self):
        link_info = p.getLinkState(self.armId, self.endEffectorIndex)
        pos = link_info[0]
        return pos

    def _observe_peg(self):
        return p.getBasePositionAndOrientation(self.pegId)[0]

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
        reaction_force = [0, 0, 0]

        # TODO handle contact for gripper and peg
        contactInfo = p.getContactPoints(self.pegId, self.armId)
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

    def _observe_info(self, visualize=True):
        info = {}

        # number of wall contacts
        info['wc'] = 0
        if self.level > 0:
            for wallId in self.walls:
                contactInfo = p.getContactPoints(self.pegId, wallId)
                info['wc'] += len(contactInfo)

        # pusher velocity
        v, va = p.getBaseVelocity(self.pegId)
        info['pv'] = np.linalg.norm(v)
        info['pva'] = np.linalg.norm(va)

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
        peg_pos = self._observe_peg()
        diff = peg_pos[:2] - self.hole
        dist = np.linalg.norm(diff)
        done = dist < self.dist_for_done
        return (dist * 10) ** 2, done

    def _finish_action(self, old_state, action):
        """Evaluate action after finishing it; step should not modify state after calling this"""
        self.state = np.array(self._obs())

        # track trajectory
        prev_block = self.get_ee_pos(old_state)
        new_block = self.get_ee_pos(self.state)
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
        jointPoses = p.calculateInverseKinematics(self.armId,
                                                  self.endEffectorIndex,
                                                  end,
                                                  self.endEffectorOrientation)

        num_arm_indices = len(self.armInds)
        p.setJointMotorControlArray(self.armId, self.armInds, controlMode=p.POSITION_CONTROL,
                                    targetPositions=jointPoses[:num_arm_indices],
                                    targetVelocities=[0] * num_arm_indices,
                                    forces=[self.MAX_FORCE] * num_arm_indices,
                                    positionGains=[0.3] * num_arm_indices,
                                    velocityGains=[1] * num_arm_indices)
        self._close_gripper()

    def _move_and_wait(self, eePos, steps_to_wait=50):
        # execute the action
        self._move_pusher(eePos)
        p.stepSimulation()
        for _ in range(steps_to_wait):
            self._observe_info()
            p.stepSimulation()
            if self.mode is p.GUI and self.sim_step_wait:
                time.sleep(self.sim_step_wait)

    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        return dx, dy

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = self._obs()
        dx, dy = self._unpack_action(action)

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)

        ee_pos = self.get_ee_pos(old_state)
        # TODO apply force into the floor?
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, self._ee_z))
        self._dd.draw_point('final eepos', final_ee_pos, color=(0, 0.5, 0.5))

        # TODO might have to set start_ee_pos's height to be different from ee_pos
        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        self._setup_ee()

        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        self._open_gripper()
        self._close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        self._dd.draw_point('x0', self.get_ee_pos(self.state), color=(0, 1, 0))
        return np.copy(self.state)


class PegFloatingGripperEnv(PegInHoleEnv):
    nu = 2
    nx = 5
    MAX_FORCE = 10
    MAX_GRIPPER_FORCE = 30
    MAX_PUSH_DIST = 0.03
    OPEN_ANGLE = 0.025
    CLOSE_ANGLE = 0.01

    # --- set current state
    def set_state(self, state, action=None):
        p.resetBasePositionAndOrientation(self.pegId, (state[0], state[1], self._ee_z - _EE_PEG_Z_DIFF),
                                          [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.gripperId, (state[0], state[1], self._ee_z),
                                          self.endEffectorOrientation)
        self.state = state
        self._draw_state()
        if action is not None:
            self._draw_action(action, old_state=state)

    def _observe_ee(self):
        gripperPose = p.getBasePositionAndOrientation(self.gripperId)
        return gripperPose[0]

    def _calculate_init_joints_to_hold_peg(self):
        self.initGripperPos = [self.initPeg[0], self.initPeg[1], self._ee_z]

    def _open_gripper(self):
        p.resetJointState(self.gripperId, PandaJustGripperID.FINGER_A, self.OPEN_ANGLE)
        p.resetJointState(self.gripperId, PandaJustGripperID.FINGER_B, self.OPEN_ANGLE)

    def _close_gripper(self):
        p.setJointMotorControlArray(self.gripperId,
                                    [PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.CLOSE_ANGLE, self.CLOSE_ANGLE],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def _move_pusher(self, end):
        p.changeConstraint(self.gripperConstraint, end, maxForce=self.MAX_FORCE)
        self._close_gripper()

    def _setup_gripper(self):
        # orientation of the end effector (pointing down)
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, -np.pi, np.pi / 2])

        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

        # use a floating gripper
        self.gripperId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "panda_gripper.urdf"), self.initGripperPos,
                                    self.endEffectorOrientation)
        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_A, lateralFriction=2)
        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_B, lateralFriction=2)

        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.gripperId,
                               PandaJustGripperID.FINGER_A,
                               self.gripperId,
                               PandaJustGripperID.FINGER_B,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.initGripperPos, self.endEffectorOrientation)

        self._open_gripper()
        self._close_gripper()

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = self._obs()
        dx, dy = self._unpack_action(action)

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action, old_state=old_state)

        ee_pos = self.get_ee_pos(old_state)
        # apply force into the floor
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, self._ee_z - 0.02))
        self._dd.draw_point('final eepos', final_ee_pos, color=(0, 0.5, 0.5))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]

        # handle contact for gripper and peg
        contactInfo = p.getContactPoints(self.pegId, self.gripperId)
        info['npb'] = len(contactInfo)
        for i, contact in enumerate(contactInfo):
            if contact[ContactInfo.LINK_B] not in (PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B):
                continue
            f_contact = get_total_contact_force(contact, False)
            reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]

            name = 'r{}'.format(i)
            if abs(contact[ContactInfo.NORMAL_MAG]) > abs(self._largest_contact.get(name, 0)):
                self._largest_contact[name] = contact[ContactInfo.NORMAL_MAG]
                if visualize and self._debug_visualizations[DebugVisualization.BLOCK_ON_PUSHER]:
                    self._dd.draw_contact_point(name, contact, False)

        self._observe_raw_reaction_force(info, reaction_force, visualize)

    def reset(self):
        self._setup_ee()

        self._open_gripper()
        if self.gripperConstraint:
            p.removeConstraint(self.gripperConstraint)
        p.resetBasePositionAndOrientation(self.gripperId, self.initGripperPos, self.endEffectorOrientation)
        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.initGripperPos, self.endEffectorOrientation)
        self._close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        self._dd.draw_point('x0', self.get_ee_pos(self.state), color=(0, 1, 0))
        return np.copy(self.state)


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class PegInHole(PybulletSim):
    def __init__(self, env: PegInHoleEnv, ctrl, save_dir=_DIR, **kwargs):
        super(PegInHole, self).__init__(env, ctrl, save_dir=save_dir, **kwargs)


class PegInHoleDataSource(PybulletEnvDataSource):
    loader_map = {PegInHoleEnv: PegLoader, PegFloatingGripperEnv: PegLoader}

    @staticmethod
    def _default_data_dir():
        return _DIR

    @staticmethod
    def _loader_map(env_type):
        return PegInHoleDataSource.loader_map.get(env_type, None)
