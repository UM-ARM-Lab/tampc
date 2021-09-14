import logging
import math
import pybullet as p
import time
import enum
import torch
import os
import random
import scipy.stats

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

from pybullet_object_models import ycb_objects

from arm_pytorch_utilities import tensor_utils
from stucco.env.pybullet_env import PybulletEnv, get_total_contact_force, make_box, state_action_color_pairs, \
    ContactInfo, make_cylinder, closest_point_on_surface
from stucco.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource, InfoKeys, \
    PlanarPointToConfig
from stucco.env.panda import PandaJustGripperID
from stucco.env.pybullet_sim import PybulletSim
from stucco import cfg
from stucco import tracking
from stucco.defines import NO_CONTACT_ID
from stucco.detection import ContactDetector
from stucco.detection_impl import ContactDetectorPlanarPybulletGripper

logger = logging.getLogger(__name__)

DIR = "arm"

kukaEndEffectorIndex = 6
pandaNumDofs = 7


class Levels(enum.IntEnum):
    FREESPACE = 0
    WALL = 1
    WALL_BROKEN_JOINT = 2
    MOVEABLE_CANS = 3
    STRAIGHT_LINE = 4
    NCB_C = 5
    WALL_BEHIND = 6
    NCB_S = 7
    NCB_T = 8
    RANDOM = 9
    SELECT1 = 10
    SELECT2 = 11
    SELECT3 = 12
    SELECT4 = 13

    # levels for object retrieval
    NO_CLUTTER = 20
    SIMPLE_CLUTTER = 21
    FLAT_BOX = 22
    BEHIND_CAN = 23
    IN_BETWEEN = 24


selected_levels = [Levels.SELECT1, Levels.SELECT2, Levels.SELECT3, Levels.SELECT4]

task_map = {str(c).split('.')[1]: c for c in Levels}

DEFAULT_MOVABLE_RGBA = [0.8, 0.7, 0.3, 0.8]


class ArmLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if self.config.predict_difference:
            y = ArmEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class DebugVisualization(enum.IntEnum):
    STATE = 2
    ACTION = 3
    REACTION_MINI_STEP = 4
    REACTION_IN_STATE = 5
    GOAL = 6
    INIT = 7


class ReactionForceStrategy(enum.IntEnum):
    MAX_OVER_CONTROL_STEP = 0
    MAX_OVER_MINI_STEPS = 1
    AVG_OVER_MINI_STEPS = 2
    MEDIAN_OVER_MINI_STEPS = 3


class ArmEnv(PybulletEnv):
    """To start with we have a fixed gripper orientation so the state is 3D position only"""
    nu = 3
    nx = 6
    MAX_FORCE = 1 * 40
    MAX_GRIPPER_FORCE = 20
    MAX_PUSH_DIST = 0.03
    FINGER_OPEN = 0.04
    FINGER_CLOSED = 0.01

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)', '$r_x$ (N)', '$r_y$ (N)', '$r_z$ (N)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @staticmethod
    def get_ee_reaction(state):
        return state[-2:]

    @staticmethod
    @tensor_utils.ensure_2d_input
    def get_ee_pos_states(states):
        return states[:, :3]

    @classmethod
    @tensor_utils.ensure_2d_input
    def get_state_ee_pos(cls, pos):
        raise NotImplementedError()

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        dreaction = state[:, 3:] - other_state[:, 3:]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1, 0, 0, 0])

    @classmethod
    def state_distance(cls, state_difference):
        return state_difference[:, :3].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$', 'd$z_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @classmethod
    @handle_data_format_for_state_diff
    def control_similarity(cls, u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    @property
    def robot_id(self):
        return self.armId

    def __init__(self, goal=(0.8, 0.0, 0.3), init=(0.3, 0.6, 0.2),
                 environment_level=0, sim_step_wait=None, mini_steps=15, wait_sim_steps_per_mini_step=20,
                 debug_visualizations=None, dist_for_done=0.04, camera_dist=1.5,
                 contact_residual_threshold=1.,
                 contact_residual_precision=None,
                 reaction_force_strategy=ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS, **kwargs):
        """
        :param environment_level: what obstacles should show up in the environment
        :param sim_step_wait: how many seconds to wait between each sim step to show intermediate states
        (0.01 seems reasonable for visualization)
        :param mini_steps how many mini control steps to divide the control step into;
        more is better for controller and allows greater force to prevent sliding
        :param wait_sim_steps_per_mini_step how many sim steps to wait per mini control step executed;
        inversely proportional to mini_steps
        :param contact_residual_threshold magnitude threshold on the reaction residual (measured force and torque
        at end effector) for when we should consider to be in contact with an object
        :param contact_residual_precision if specified, the inverse of a matrix representing the expected measurement
        error of the contact residual; if left none default values for the environment is used. This is used to
        normalize the different dimensions of the residual so that they are around the same expected magnitude.
        Typically estimated from the training set, but for simulated environments can just pass in 1/max training value
        for normalization.
        :param reaction_force_strategy how to aggregate measured reaction forces over control step into 1 value
        :param kwargs:
        """
        super().__init__(**kwargs, default_debug_height=0.1, camera_dist=camera_dist)
        self._dd.toggle_3d(True)
        self.level = Levels(environment_level)
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy
        self.dist_for_done = dist_for_done

        # object IDs
        self.immovable = []
        self.movable = []

        # debug parameter for extruding objects so penetration is measured wrt the x-y plane
        self.extrude_objects_in_z = False

        # initial config
        self.goal = None
        self.init = None
        self.armId = None

        self._debug_visualizations = {
            DebugVisualization.STATE: False,
            DebugVisualization.ACTION: False,
            DebugVisualization.REACTION_MINI_STEP: False,
            DebugVisualization.REACTION_IN_STATE: False,
            DebugVisualization.GOAL: False,
            DebugVisualization.INIT: False,
        }
        if debug_visualizations is not None:
            self._debug_visualizations.update(debug_visualizations)
        self._contact_debug_names = []

        # avoid the spike at the start of each mini step from rapid acceleration
        self._steps_since_start_to_get_reaction = 5
        self._clear_state_between_control_steps()

        self.set_task_config(goal, init)
        self._setup_experiment()
        self._contact_detector = self.create_contact_detector(contact_residual_threshold, contact_residual_precision)
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()

    @property
    def contact_detector(self) -> ContactDetector:
        return self._contact_detector

    # --- initialization and task configuration
    def _clear_state_between_control_steps(self):
        self._sim_step = 0
        self._mini_step_contact = {'full': np.zeros((self.mini_steps + 1, 3)),
                                   'torque': np.zeros((self.mini_steps + 1, 3)),
                                   'mag': np.zeros(self.mini_steps + 1),
                                   'id': np.ones(self.mini_steps + 1) * NO_CONTACT_ID}
        self._contact_info = {}
        self._largest_contact = {}
        self._reaction_force = np.zeros(2)

    def set_task_config(self, goal=None, init=None):
        if goal is not None:
            self._set_goal(goal)
        if init is not None:
            self._set_init(init)

    def _set_goal(self, goal):
        # ignore the pusher position
        self.goal = np.array(tuple(goal) + (0, 0, 0))
        if self._debug_visualizations[DebugVisualization.GOAL]:
            self._dd.draw_point('goal', self.goal)

    def _set_init(self, init):
        # initial position of end effector
        self.init = init
        if self._debug_visualizations[DebugVisualization.INIT]:
            self._dd.draw_point('init', self.init, color=(0, 1, 0.2))
        if self.armId is not None:
            self._calculate_init_joints()

    def _calculate_init_joints(self):
        self.initJoints = list(p.calculateInverseKinematics(self.armId,
                                                            self.endEffectorIndex,
                                                            self.init,
                                                            self.endEffectorOrientation))

    def set_state(self, state, action=None):
        for i in self.armInds:
            p.resetJointState(self.armId, i, state[i])
        self.state = state
        self._draw_state()
        if action is not None:
            self._draw_action(action, old_state=state)

    # def open_gripper(self):
    #     p.resetJointState(self.armId, PandaGripperID.FINGER_A, self.FINGER_OPEN)
    #     p.resetJointState(self.armId, PandaGripperID.FINGER_B, self.FINGER_OPEN)
    #
    # def close_gripper(self):
    #     p.setJointMotorControlArray(self.armId,
    #                                 [PandaGripperID.FINGER_A, PandaGripperID.FINGER_B],
    #                                 p.POSITION_CONTROL,
    #                                 targetPositions=[self.FINGER_CLOSED, self.FINGER_CLOSED],
    #                                 forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])
    def _setup_objects(self):
        self.immovable = []
        if self.level == 0:
            pass
        elif self.level in [1, 2]:
            half_extents = [0.2, 0.05, 0.3]
            colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visId = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 0.8])
            wallId = p.createMultiBody(0, colId, visId, basePosition=[0.6, 0.30, 0.2],
                                       baseOrientation=p.getQuaternionFromEuler([0, 0, 1.1]))
            p.changeDynamics(wallId, -1, lateralFriction=1)
            self.immovable.append(wallId)

        for wallId in self.immovable:
            p.changeVisualShape(wallId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self._setup_gripper()
        self._setup_objects()

        self.set_camera_position([0, 0], yaw=113, pitch=-40)

        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

    def _setup_gripper(self):
        # add kuka arm
        # self.armId = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
        # self.armId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.armId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        p.resetBasePositionAndOrientation(self.armId, [0, 0, 0], [0, 0, 0, 1])

        # TODO modify dynamics to induce traps
        # for j in range(p.getNumJoints(self.armId)):
        #     p.changeDynamics(self.armId, j, linearDamping=0, angularDamping=0)

        # orientation of the end effector
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        self.endEffectorIndex = kukaEndEffectorIndex
        self.numJoints = p.getNumJoints(self.armId)
        # get the joint ids
        # TODO try out arm with attached grippers
        # self.armInds = [i for i in range(pandaNumDofs)]
        self.armInds = [i for i in range(self.numJoints)]

        # create a constraint to keep the fingers centered
        # c = p.createConstraint(self.armId,
        #                        9,
        #                        self.armId,
        #                        10,
        #                        jointType=p.JOINT_GEAR,
        #                        jointAxis=[1, 0, 0],
        #                        parentFramePosition=[0, 0, 0],
        #                        childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # joint damping coefficents
        # self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # self.jd = [
        #     0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        #     0.00001, 0.00001, 0.00001, 0.00001
        # ]

        p.enableJointForceTorqueSensor(self.armId, self.endEffectorIndex)
        self._calculate_init_joints()
        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        # self.open_gripper()
        # self.close_gripper()

        self._make_robot_translucent(self.armId)

    def visualize_rollouts(self, rollout, state_cmap='Blues_r', contact_cmap='Reds_r'):
        """In GUI mode, show how the sequence of states will look like"""
        if rollout is None:
            return
        if type(rollout) is tuple and len(rollout) == 3:
            states, contact_model_active, center_points = rollout
        else:
            states = rollout
            contact_model_active = np.zeros(len(states))
            center_points = [None]
        # assume states is iterable, so could be a bunch of row vectors
        T = len(states)
        if T > 0:
            smap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=T), cmap=state_cmap)
            cmap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=T), cmap=contact_cmap)
            prev_pos = None
            for t in range(T):
                pos = self.get_ee_pos(states[t])
                rgba = cmap.to_rgba(t) if contact_model_active[t] else smap.to_rgba(t)
                self._dd.draw_point('rx{}.{}'.format(state_cmap, t), pos, rgba[:-1])
                if t > 0:
                    self._dd.draw_2d_line('tx{}.{}'.format(state_cmap, t), prev_pos, pos - prev_pos, scale=1,
                                          color=rgba[:-1])
                prev_pos = pos
        self._dd.clear_visualization_after('rx{}'.format(state_cmap), T)
        self._dd.clear_visualization_after('tx{}'.format(state_cmap), T)

        if center_points[0] is not None:
            obj_center_color_maps = ['Purples_r', 'Greens_r', 'Greys_r']
            # only consider the first sample (m = 0)
            center_points = [pt[:, 0] for pt in center_points]
            center_points = torch.stack(center_points)
            num_objs = center_points.shape[1]
            for j in range(num_objs):
                rollout = center_points[:, j]
                self.visualize_rollouts(rollout.cpu().numpy(),
                                        state_cmap=obj_center_color_maps[j % len(obj_center_color_maps)])
            # clear the other colors
            for j in range(num_objs, len(obj_center_color_maps)):
                self.visualize_rollouts([], state_cmap=obj_center_color_maps[j % len(obj_center_color_maps)])

    def visualize_goal_set(self, states):
        if states is None:
            return
        T = len(states)
        for t in range(T):
            pos = self.get_ee_pos(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_point('gs.{}'.format(t), pos, (c, c, c))
        self._dd.clear_visualization_after('gs', T)

    def visualize_trap_set(self, trap_set):
        if trap_set is None:
            return
        T = len(trap_set)
        for t in range(T):
            c = (t + 1) / (T + 1)
            # decide whether we're given state and action or just state
            if len(trap_set[t]) == 2:
                state, action = trap_set[t]
                self._draw_action(action.cpu().numpy(), old_state=state.cpu().numpy(), debug=t + 1)
            else:
                state = trap_set[t]
            pose = self.get_ee_pos(state)
            self._dd.draw_point('ts.{}'.format(t), pose, (1, 0, c))
        self._dd.clear_visualization_after('ts', T)
        self._dd.clear_visualization_after('u', T + 1)

    def visualize_state_actions(self, base_name, states, actions, state_c, action_c, action_scale):
        if torch.is_tensor(states):
            states = states.cpu()
            if actions is not None:
                actions = actions.cpu()
        j = 0
        for j in range(len(states)):
            p = self.get_ee_pos(states[j])
            name = '{}.{}'.format(base_name, j)
            self._dd.draw_point(name, p, color=state_c)
            if actions is not None:
                # draw action
                name = '{}a.{}'.format(base_name, j)
                self._dd.draw_2d_line(name, p, actions[j], color=action_c, scale=action_scale)
        self._dd.clear_visualization_after(base_name, j + 1)
        self._dd.clear_visualization_after('{}a'.format(base_name), j + 1)

    def visualize_contact_set(self, contact_set: tracking.ContactSet):
        if isinstance(contact_set, tracking.ContactSetHard):
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(contact_set):
                for name in self._contact_debug_names:
                    self._dd.clear_visualizations(name)
                self._contact_debug_names = []
            for i, c in enumerate(contact_set):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                name = 'cp{}'.format(i)
                eigval, eigvec = torch.eig(c.cov[0], eigenvectors=True)
                yx_ratio = eigval[1, 0] / eigval[0, 0]
                rot = math.atan2(eigvec[1, 0], eigvec[0, 0])
                l = eigval[0, 0] * 100
                w = c.weight
                self._dd.draw_point(name, self.get_ee_pos(c.mu[0]), length=l.item(), length_ratio=yx_ratio, rot=rot,
                                    color=color)
                self._contact_debug_names[i].add(name)

                base_name = str(i)
                self.visualize_state_actions(base_name, c.points, c.actions, color, u_color, 0.1 * w)

                for j in range(len(c.points)):
                    self._contact_debug_names[i].add('{}{}'.format(base_name, j))
                    self._contact_debug_names[i].add('{}{}a'.format(base_name, j))
        elif isinstance(contact_set, tracking.ContactSetSoft):
            pts = contact_set.get_posterior_points()
            if pts is None:
                return
            groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(groups):
                for name in self._contact_debug_names:
                    self._dd.clear_visualizations(name)
                self._contact_debug_names = []
            for i, indices in enumerate(groups):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                base_name = str(i)
                self.visualize_state_actions(base_name, pts[indices], contact_set.acts[indices], color, u_color, 0.1)
                for j in range(len(pts[indices])):
                    self._contact_debug_names[i].add('{}{}'.format(base_name, j))
                    self._contact_debug_names[i].add('{}{}a'.format(base_name, j))

        # clean up any old visualization
        # for i in range(len(contact_set), len(self._contact_debug_names)):
        #     self._dd.clear_visualizations(self._contact_debug_names[i])
        # self._contact_debug_names = self._contact_debug_names[:len(contact_set)]

    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""
        pred = self.get_ee_pos(predicted_state)
        c = (0.5, 0, 0.5)
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('ep', pred, c)

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_state(self):
        if self._debug_visualizations[DebugVisualization.STATE]:
            pos = self.get_ee_pos(self.state)
            self._dd.draw_point('state', pos)
        if self._debug_visualizations[DebugVisualization.REACTION_IN_STATE]:
            self._draw_reaction_force(self.state[3:6], 'sr', (0, 0, 0))

    def _draw_action(self, action, old_state=None, debug=0):
        if old_state is None:
            old_state = self._obs()
        start = old_state[:3]
        pointer = action
        if debug:
            self._dd.draw_2d_line('u.{}'.format(debug), start, pointer, (1, debug / 30, debug / 10), scale=0.2)
        else:
            self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=0.2)

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self.get_ee_pos(self._obs())
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = np.concatenate((self._observe_ee(), self._observe_reaction_force_torque()[0]))
        return state

    def _observe_ee(self, return_z=True, return_orientation=False):
        link_info = p.getLinkState(self.armId, self.endEffectorIndex, computeForwardKinematics=True)
        pos = link_info[4]
        if not return_z:
            pos = pos[:2]
        if return_orientation:
            rot = link_info[5]
            pos = pos, rot
        return pos

    def _observe_reaction_force_torque(self):
        """Return representative reaction force for simulation steps up to current one since last control step"""
        if self.reaction_force_strategy is ReactionForceStrategy.AVG_OVER_MINI_STEPS:
            return self._mini_step_contact['full'].mean(axis=0), self._mini_step_contact['torque'].mean(axis=0)
        if self.reaction_force_strategy is ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS:
            median_mini_step = np.argsort(self._mini_step_contact['mag'])[self.mini_steps // 2]
            return self._mini_step_contact['full'][median_mini_step], \
                   self._mini_step_contact['torque'][median_mini_step]
        if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_MINI_STEPS:
            max_mini_step = np.argmax(self._mini_step_contact['mag'])
            return self._mini_step_contact['full'][max_mini_step], self._mini_step_contact['torque'][max_mini_step]
        else:
            raise NotImplementedError("Not implemented max over control step reaction torque")

    def _observe_additional_info(self, info, visualize=True):
        joint_pos, joint_vel, joint_reaction_force, joint_applied = p.getJointState(self.armId, self.endEffectorIndex)
        info['pv'] = joint_vel

        # transform reaction to world frame
        states = p.getLinkState(self.armId, self.endEffectorIndex)
        world_link_orientation = states[5]
        r = p.rotateVector(world_link_orientation, joint_reaction_force[:3])
        t = p.rotateVector(world_link_orientation, joint_reaction_force[3:])

        self._observe_raw_reaction_force(info, r, t, visualize)

    def _observe_info(self, visualize=True):
        info = {}

        self._observe_additional_info(info, visualize)
        self._sim_step += 1

        for key, value in info.items():
            if key not in self._contact_info:
                self._contact_info[key] = []
            self._contact_info[key].append(value)

    def get_ee_contact_info(self, bodyId):
        # changes when end effector type changes
        return p.getContactPoints(bodyId, self.armId, linkIndexB=self.endEffectorIndex)

    def _observe_raw_reaction_force(self, info, reaction_force, reaction_torque, visualize=True):
        # can estimate change in state only when in contact
        new_ee_pos, new_ee_orientation = self._observe_ee(return_z=True, return_orientation=True)
        pose = (new_ee_pos, new_ee_orientation)
        if self.contact_detector.observe_residual(np.r_[reaction_force, reaction_torque], pose):
            info[InfoKeys.DEE_IN_CONTACT] = np.subtract(new_ee_pos, self.last_ee_pos)
        self.last_ee_pos = new_ee_pos

        # save end effector pose
        info[InfoKeys.HIGH_FREQ_EE_POSE] = np.r_[new_ee_pos, new_ee_orientation]

        # save reaction force
        info[InfoKeys.HIGH_FREQ_REACTION_T] = reaction_torque
        name = InfoKeys.HIGH_FREQ_REACTION_F
        info[name] = reaction_force
        reaction_force_size = np.linalg.norm(reaction_force)
        # see if we should save it as the reaction force for this mini-step
        mini_step, step_since_start = divmod(self._sim_step, self.wait_sim_step_per_mini_step)

        # detect what we are in contact with
        for bodyId in self.movable + self.immovable:
            contactInfo = self.get_ee_contact_info(bodyId)
            # assume at most single body in contact
            if len(contactInfo):
                self._mini_step_contact['id'][mini_step] = bodyId
                pt = contactInfo[0][ContactInfo.POS_A]
                info[InfoKeys.HIGH_FREQ_CONTACT_POINT] = pt
                break
        else:
            info[InfoKeys.HIGH_FREQ_CONTACT_POINT] = [0, 0, 0]

        if step_since_start is self._steps_since_start_to_get_reaction:
            self._mini_step_contact['full'][mini_step] = reaction_force
            self._mini_step_contact['torque'][mini_step] = reaction_torque
            self._mini_step_contact['mag'][mini_step] = reaction_force_size
            if self.reaction_force_strategy is not ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_MINI_STEP] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))
        # update our running count of max force
        if reaction_force_size > self._largest_contact.get(name, 0):
            self._largest_contact[name] = reaction_force_size
            self._reaction_force = reaction_force
            if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_MINI_STEP] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))

    def _aggregate_info(self):
        info = {key: np.stack(value, axis=0) for key, value in self._contact_info.items() if len(value)}
        info['reaction'], info['torque'] = self._observe_reaction_force_torque()
        name = InfoKeys.DEE_IN_CONTACT
        if name in info:
            info[name] = info[name].sum(axis=0)
        else:
            info[name] = np.zeros(3)
        most_frequent_contact = scipy.stats.mode(self._mini_step_contact['id'])
        info[InfoKeys.CONTACT_ID] = int(most_frequent_contact[0])

        # ground truth object information
        if len(self.movable + self.immovable):
            for obj_id in self.movable + self.immovable:
                pose = p.getBasePositionAndOrientation(obj_id)
                c = p.getClosestPoints(obj_id, self.robot_id, 100000)
                info[f"obj{obj_id}pose"] = np.concatenate([pose[0], pose[1]])
                # for multi-link bodies, will return 1 per combination; store the min
                info[f"obj{obj_id}distance"] = min(cc[ContactInfo.DISTANCE] for cc in c)

        return info

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        diff = self.get_ee_pos(state) - self.get_ee_pos(self.goal)
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
        if cost is not None:
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
        self._send_move_command(jointPoses)
        # self.close_gripper()

    def _send_move_command(self, jointPoses):
        num_arm_indices = len(self.armInds)
        p.setJointMotorControlArray(self.armId, self.armInds, controlMode=p.POSITION_CONTROL,
                                    targetPositions=jointPoses[:num_arm_indices],
                                    targetVelocities=[0] * num_arm_indices,
                                    # forces=[self.MAX_FORCE] * num_arm_indices,
                                    forces=[100, 100, 60, 60, 50, 40, 40],
                                    positionGains=[0.3] * num_arm_indices,
                                    velocityGains=[1] * num_arm_indices)

    def _move_and_wait(self, eePos, steps_to_wait=50):
        # execute the action
        self.last_ee_pos = self._observe_ee(return_z=True)
        self._move_pusher(eePos)
        p.stepSimulation()
        for _ in range(steps_to_wait):
            self._observe_info()
            p.stepSimulation()
            if self.mode is p.GUI and self.sim_step_wait:
                time.sleep(self.sim_step_wait)
        self._observe_info()

    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        dz = action[2] * self.MAX_PUSH_DIST
        return dx, dy, dz

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = np.copy(self.state)
        dx, dy, dz = self._unpack_action(action)

        ee_pos = self.get_ee_pos(old_state)
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz))

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)
            self._dd.draw_point('final eepos', final_ee_pos, color=(1, 0.5, 0.5))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        # self._setup_ee()
        self._contact_debug_names = []

        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        # self.open_gripper()
        # self.close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        self._send_move_command(self.initJoints)
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        pos = self.get_ee_pos(self.state)
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('x0', pos, color=(0, 1, 0))
        return np.copy(self.state)


class ArmJointEnv(ArmEnv):
    """Control the joints directly"""
    nu = 6
    nx = 6 + 3
    MAX_FORCE = 1 * 40
    MAX_ANGLE_CHANGE = 0.07

    @staticmethod
    def state_names():
        return ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', '$r_x$ (N)', '$r_y$ (N)', '$r_z$ (N)']

    def get_ee_pos(self, state):
        # do forward kinematics to get ee pos from state
        state = state.reshape(-1)
        for i in range(6):
            p.resetJointState(self.armId, i, state[i])
        ee = np.array(self._observe_ee())
        for i in range(6):
            p.resetJointState(self.armId, i, self.state[i])
        return ee

    def compare_to_goal(self, state, goal):
        # if torch.is_tensor(goal) and not torch.is_tensor(state):
        #     state = torch.from_numpy(state).to(device=goal.device)
        diff = state - goal
        if len(diff.shape) == 1:
            diff = diff.reshape(1, -1)
        return diff

    def _set_goal(self, goal):
        try:
            # get some IK solutions around goal
            # TODO sample many orientations at the goal and include them all
            # TODO change cost function to take the minimum distance to any of these configurations
            goal_orientation = p.getQuaternionFromEuler([0, math.pi, 0])
            self.goal = p.calculateInverseKinematics(self.armId,
                                                     self.endEffectorIndex,
                                                     goal,
                                                     goal_orientation)
            for i in range(6):
                p.resetJointState(self.armId, i, self.goal[i])

            self.goal_pos = np.array(self._observe_ee())
            self._dd.draw_point('goal', self.goal_pos)
            self.goal = np.array(self.goal[:6] + (0, 0, 0))
        except AttributeError:
            logger.warning("setting goal before able to do inverse kinematics; set goal after initialization")
            pass

    @staticmethod
    def get_joints(state):
        return state[:6]

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :6] - other_state[:, :6]
        dreaction = state[:, 6:] - other_state[:, 6:]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1, 1, 1, 0, 0, 0, 0])

    @classmethod
    def state_distance(cls, state_difference):
        return state_difference[:, :6].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$q_1$', 'd$q_2$', 'd$q_3$', 'd$q_4$', 'd$q_5$', 'd$q_6$']

    @classmethod
    def get_control_bounds(cls):
        u_min = np.array([-1 for _ in range(cls.nu)])
        u_max = np.array([1 for _ in range(cls.nu)])
        return u_min, u_max

    def _obs(self):
        state = np.concatenate((self._observe_joints(), self._observe_reaction_force_torque()[0]))
        return state

    def _observe_joints(self):
        states = p.getJointStates(self.armId, self.armInds[:-1])
        # retrieve just joint position
        pos = [state[0] for state in states]
        return pos

    def _move_pusher(self, end):
        # given joint poses directly
        self._send_move_command(end)

    def _unpack_action(self, action):
        return np.array([a * self.MAX_ANGLE_CHANGE for a in action])

    def evaluate_cost(self, state, action=None):
        diff = self.get_ee_pos(state) - self.goal_pos
        dist = np.linalg.norm(diff)
        done = dist < self.dist_for_done
        return (dist * 10) ** 2, done

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = np.copy(self.state)
        old_joints = self.get_joints(old_state)

        dq = self._unpack_action(action)

        new_joints = old_joints + dq

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_joints = interpolate_pos(old_joints, new_joints, (step + 1) / self.mini_steps)
            # use fixed end effector angle
            intermediate_joints = np.r_[intermediate_joints, 0]
            self._move_and_wait(intermediate_joints, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def _draw_state(self):
        pass

    def _draw_action(self, action, old_state=None, debug=0):
        pass


FIXED_Z = 0.1


class PlanarArmEnv(ArmEnv):
    """To start with we have a fixed gripper orientation so the state is 3D position only"""
    nu = 2
    nx = 4

    @staticmethod
    def state_names():
        # TODO allow theta rotation (see block push)
        return ['x ee (m)', 'y ee (m)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    def get_ee_pos(state):
        if torch.is_tensor(state):
            return torch.cat((state[:2], torch.tensor(FIXED_Z, dtype=state.dtype, device=state.device).view(1)))
        return np.r_[state[:2], FIXED_Z]

    @staticmethod
    @tensor_utils.ensure_2d_input
    def get_ee_pos_states(states):
        return states[:, :2]

    @classmethod
    @tensor_utils.ensure_2d_input
    def get_state_ee_pos(cls, pos):
        return torch.cat((pos, torch.zeros(pos.shape[0], cls.nx - pos.shape[1], dtype=pos.dtype, device=pos.device)),
                         dim=1)

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :2] - other_state[:, :2]
        dreaction = state[:, 2:] - other_state[:, 2:]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 0, 0])

    @classmethod
    def state_distance(cls, state_difference):
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
    def compare_to_goal(state, goal):
        # if torch.is_tensor(goal) and not torch.is_tensor(state):
        #     state = torch.from_numpy(state).to(device=goal.device)
        diff = state - goal
        if len(diff.shape) == 1:
            diff = diff.reshape(1, -1)
        return diff

    def __init__(self, goal=(1.0, -0.4), init=(0.5, 0.8), **kwargs):
        super(PlanarArmEnv, self).__init__(goal=goal, init=tuple(init) + (FIXED_Z,), **kwargs)

    def _observe_ee(self, return_z=False, **kwargs):
        return super(PlanarArmEnv, self)._observe_ee(return_z=return_z, **kwargs)

    def _observe_reaction_force_torque(self):
        r, t = super(PlanarArmEnv, self)._observe_reaction_force_torque()
        return r[:2], t

    def _set_goal(self, goal):
        if len(goal) > 2:
            goal = goal[:2]
        # ignore the pusher position
        self.goal = np.array(tuple(goal) + (0, 0))
        if self._debug_visualizations[DebugVisualization.GOAL]:
            self._dd.draw_point('goal', tuple(goal) + (FIXED_Z,))

    def _set_init(self, init):
        if len(init) > 2:
            init = init[:2]
        super(PlanarArmEnv, self)._set_init(tuple(init) + (FIXED_Z,))

    def _setup_objects(self):
        self.immovable = []
        if self.level == 0:
            pass
        elif self.level in [1, 2]:
            self.immovable.append(make_box([0.2, 0.05, 0.3], [0.6, 0.3, 0.2], [0, 0, 1.1], lateral_friction=1))
        for wallId in self.immovable:
            p.changeVisualShape(wallId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self._setup_gripper()
        self._setup_objects()

        self.set_camera_position([0.5, 0.3], yaw=-75, pitch=-80)

        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

    def _setup_gripper(self):
        # add kuka arm
        self.armId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        p.resetBasePositionAndOrientation(self.armId, [0, 0, FIXED_Z * 2],
                                          p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))

        # orientation of the end effector
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        self.endEffectorIndex = kukaEndEffectorIndex
        self.numJoints = p.getNumJoints(self.armId)
        # get the joint ids
        # self.armInds = [i for i in range(pandaNumDofs)]
        self.armInds = [i for i in range(self.numJoints)]

        p.enableJointForceTorqueSensor(self.armId, self.endEffectorIndex)
        self._calculate_init_joints()
        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])

        self._make_robot_translucent(self.armId)

    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        return dx, dy

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = np.copy(self.state)
        dx, dy = self._unpack_action(action)

        ee_pos = self.get_ee_pos(old_state)
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, FIXED_Z))
        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)
            self._dd.draw_point('final eepos', final_ee_pos, color=(1, 0.5, 0.5))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)
        rew = -cost if cost is not None else None

        return np.copy(self.state), rew, done, info

    def _draw_state(self):
        pos = self.get_ee_pos(self.state)
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('state', pos)
        if self._debug_visualizations[DebugVisualization.REACTION_IN_STATE]:
            self._draw_reaction_force(np.r_[self.state[2:], FIXED_Z], 'sr', (0, 0, 0))

    def _draw_action(self, action, old_state=None, debug=0):
        if old_state is None:
            old_state = self._obs()
        start = np.r_[old_state[:2], FIXED_Z]
        pointer = np.r_[action, 0]
        if debug:
            self._dd.draw_2d_line('u{}'.format(debug), start, pointer, (1, debug / 30, debug / 10), scale=0.2)
        else:
            self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=0.2)


class FloatingGripperEnv(PlanarArmEnv):
    nu = 2
    nx = 4
    MAX_FORCE = 30
    MAX_GRIPPER_FORCE = 30
    MAX_PUSH_DIST = 0.03
    OPEN_ANGLE = 0.055
    CLOSE_ANGLE = 0.0

    @property
    def robot_id(self):
        return self.gripperId

    # --- set current state
    def set_state(self, state, action=None):
        p.resetBasePositionAndOrientation(self.gripperId, (state[0], state[1], FIXED_Z),
                                          self.endEffectorOrientation)
        self.state = state
        self._draw_state()
        if action is not None:
            self._draw_action(action, old_state=state)

    def __init__(self, goal=(1.3, -0.4), init=(-.1, 0.4), **kwargs):
        super(FloatingGripperEnv, self).__init__(goal=goal, init=init, camera_dist=1, **kwargs)

    def create_contact_detector(self, residual_threshold, residual_precision) -> ContactDetector:
        if residual_precision is None:
            residual_precision = np.diag([1, 1, 1, 50, 50, 50])
        return ContactDetectorPlanarPybulletGripper("floating_gripper", residual_precision, residual_threshold,
                                                    robot_id=self.robot_id,
                                                    canonical_orientation=self.endEffectorOrientation,
                                                    default_joint_config=[0, 0])

    def _observe_ee(self, return_z=False, return_orientation=False):
        gripperPose = p.getBasePositionAndOrientation(self.gripperId)
        pos = gripperPose[0]
        if not return_z:
            pos = pos[:2]
        if return_orientation:
            pos = pos, gripperPose[1]
        return pos

    def open_gripper(self):
        p.setJointMotorControlArray(self.gripperId,
                                    [PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.OPEN_ANGLE, self.OPEN_ANGLE],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def close_gripper(self):
        p.setJointMotorControlArray(self.gripperId,
                                    [PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.CLOSE_ANGLE, self.CLOSE_ANGLE],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def _move_pusher(self, end):
        p.changeConstraint(self.gripperConstraint, end, self.endEffectorOrientation, maxForce=self.MAX_FORCE)

    def _setup_objects(self):
        self.immovable = []
        self.movable = []
        if self.level == Levels.FREESPACE:
            pass
        elif self.level in [Levels.WALL]:
            # drop movable obstacles
            z = 0.075
            xs = [0.3, 0.8]
            ys = [-0.3, 0.3]
            objId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                               basePosition=[xs[0], ys[0], z])
            self.movable.append(objId)
            objId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                               basePosition=[xs[1], ys[1], z])
            self.movable.append(objId)

            objId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=True,
                               basePosition=[xs[1], ys[0], z])
            self.immovable.append(objId)
            objId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=True,
                               basePosition=[xs[0], ys[1], z])
            self.immovable.append(objId)
        elif self.level == Levels.MOVEABLE_CANS:
            scale = 1.0
            z = 0.075 * scale
            xs = [0.3, 0.7]
            ys = [-0.2, 0.2]
            self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                           basePosition=[xs[0], ys[0], z], globalScaling=scale))
            self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                           basePosition=[xs[1], ys[1], z], globalScaling=scale))
            self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                           basePosition=[xs[1], ys[0], z], globalScaling=scale))
            self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                           basePosition=[xs[0], ys[1], z], globalScaling=scale))
            self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [xs[1] + 0.48, 0., z],
                                             p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True,
                                             globalScaling=0.5))
            self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [xs[0], ys[0] - 0.43, z],
                                             p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                             globalScaling=0.5))
        elif self.level in [Levels.STRAIGHT_LINE, Levels.WALL_BEHIND]:
            scale = 1.0
            z = 0.075 * scale
            self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                           basePosition=[0.5, 0, z]))
            if self.level == Levels.WALL_BEHIND:
                self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.21, 0., z],
                                                 p.getQuaternionFromEuler([0, 0, -np.pi / 2]), useFixedBase=True,
                                                 globalScaling=0.5))
        elif self.level in [Levels.NCB_C, Levels.NCB_S, Levels.NCB_T]:
            scale = 1.0
            z = 0.075 * scale
            y = 0
            width = 0.85
            self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.3, 0, z],
                                             p.getQuaternionFromEuler([0, 0, -np.pi / 2]), useFixedBase=True,
                                             globalScaling=0.5))
            self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.5, -width / 2, z],
                                             p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                             globalScaling=0.5))
            self.immovable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.5, width / 2, z],
                                             p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                             globalScaling=0.5))
            if self.level == Levels.NCB_C:
                self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False,
                                               basePosition=[0.7, y, z]))
            elif self.level == Levels.NCB_S:
                self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_tall.urdf"), useFixedBase=False,
                                               basePosition=[0.7, y, z]))
                p.changeVisualShape(self.movable[-1], -1, rgbaColor=DEFAULT_MOVABLE_RGBA)
            elif self.level == Levels.NCB_T:
                self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "topple_cylinder.urdf"), useFixedBase=False,
                                               basePosition=[0.7, y, z + 0.02],
                                               baseOrientation=p.getQuaternionFromEuler([0, np.pi / 2, np.pi / 2])))
        elif self.level is Levels.RANDOM:
            # first move end effector out of the way
            p.resetBasePositionAndOrientation(self.gripperId, [0, 0, 100], self.endEffectorOrientation)
            if self.gripperConstraint:
                p.removeConstraint(self.gripperConstraint)
            bound = 0.7
            obj_types = ["tester.urdf", "topple_cylinder.urdf", "block_tall.urdf", "wall.urdf"]
            # randomize number of objects, type of object, and size of object
            num_obj = random.randint(2, 5)
            for _ in range(num_obj):
                obj_type = random.choice(obj_types)
                # TODO start with making everything immovable
                # moveable = False
                moveable = obj_type != "wall.urdf"
                global_scale = random.uniform(0.5, 1.5)
                yaw = random.uniform(0, 2 * math.pi)
                z = 0.1 * global_scale
                while True:
                    # randomly sample start location and yaw
                    # convert yaw to quaternion
                    if obj_type == "topple_cylinder.urdf":
                        orientation = p.getQuaternionFromEuler([0, yaw, np.pi / 2])
                    else:
                        orientation = p.getQuaternionFromEuler([0, 0, yaw])
                    position = [random.uniform(-bound, bound), random.uniform(-bound, bound), z]
                    # even if immovable have to initialize as having movable base to enable collision checks
                    obj = p.loadURDF(os.path.join(cfg.ROOT_DIR, obj_type), useFixedBase=False,
                                     globalScaling=global_scale * 0.7 if obj_type == "wall.urdf" else global_scale,
                                     basePosition=position,
                                     baseOrientation=orientation)
                    # let settle
                    for _ in range(1000):
                        p.stepSimulation()
                    # retry positioning if we teleported out of bounds
                    pos = p.getBasePositionAndOrientation(obj)[0]
                    in_bounds = bound > pos[0] > -bound and bound > pos[1] > -bound
                    if not in_bounds:
                        p.removeBody(obj)
                        continue
                    # don't want objects leaning on each other
                    in_contact = False
                    for other_obj in self.movable + self.immovable:
                        c = p.getContactPoints(obj, other_obj)
                        if len(c):
                            in_contact = True
                            break
                    if in_contact:
                        p.removeBody(obj)
                        continue

                    if in_bounds and not in_contact:
                        break

                if moveable:
                    self.movable.append(obj)
                else:
                    # recreate object and make it fixed base
                    pose = p.getBasePositionAndOrientation(obj)
                    p.removeBody(obj)
                    obj = p.loadURDF(os.path.join(cfg.ROOT_DIR, obj_type), useFixedBase=True,
                                     globalScaling=global_scale * 0.7 if obj_type == "wall.urdf" else global_scale,
                                     basePosition=pose[0],
                                     baseOrientation=pose[1])
                    self.immovable.append(obj)
            # restore gripper movement
            p.resetBasePositionAndOrientation(self.gripperId, self.init, self.endEffectorOrientation)
            self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                        self.init, childFrameOrientation=self.endEffectorOrientation)
            self.close_gripper()
        elif self.level in selected_levels:
            z = 0.1
            s = 0.25
            h = 2 if self.extrude_objects_in_z else 0.15
            if self.level is Levels.SELECT1:
                self.immovable.append(make_box([0.4, 0.15, h], [-0.4, 0, z], [0, 0, -np.pi / 2]))
                self.movable.append(make_cylinder(0.15, h, [s, s, z], [0, 0, 0]))
                self._adjust_mass_and_visual(self.movable[-1], 2.2)
                self.movable.append(make_box([0.1 * 1.5, 0.1 * 1.5, h * 1.5 * 0.5], [s, -s, z], [0, 0, 0]))
                self._adjust_box_dynamics(self.movable[-1])
                self._adjust_mass_and_visual(self.movable[-1], 0.8)
            elif self.level is Levels.SELECT2:
                self.movable.append(make_cylinder(0.15 * 0.8, h * 0.8, [-s, s, z], [0, 0, 0]))
                self._adjust_mass_and_visual(self.movable[-1], 1)
                self.movable.append(make_box([0.1 * 1.2, 0.1 * 1.2, h * 1.2 * 0.5], [s, s, z], [0, 0, 0]))
                self._adjust_box_dynamics(self.movable[-1])
                self._adjust_mass_and_visual(self.movable[-1], 1.8)
                self.movable.append(make_cylinder(0.15 * 1.2, h * 1.2, [s, -s, z], [0, 0, 0]))
                self._adjust_mass_and_visual(self.movable[-1], 1)
                self.movable.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "topple_cylinder.urdf"), useFixedBase=False,
                                               basePosition=[-s, -s, z + 0.02],
                                               baseOrientation=p.getQuaternionFromEuler([0, np.pi / 2, np.pi / 2])))
                self._adjust_mass_and_visual(self.movable[-1], 1)
            elif self.level is Levels.SELECT3:
                self.immovable.append(make_box([0.4, 0.15, h], [0, 0.4, z], [0, 0, 0]))
                self.movable.append(make_cylinder(0.15 * 1.1, h * 1.1, [-0.2, 0, z], [0, 0, 0]))
                self._adjust_mass_and_visual(self.movable[-1], 1)
                self.movable.append(make_cylinder(0.15, h, [0.4, -s, z], [0, 0, 0]))
                self._adjust_mass_and_visual(self.movable[-1], 2.2)
            elif self.level is Levels.SELECT4:
                self.immovable.append(make_box([0.3, 0.12, h], [0.3, -0.3, z], [0, 0, np.pi / 4]))
                self.movable.append(make_box([0.1 * 1.5, 0.1 * 1.5, h * 1.5 * 0.5], [s, s, z], [0, 0, 0]))
                self._adjust_box_dynamics(self.movable[-1])
                self._adjust_mass_and_visual(self.movable[-1], 1.8)
                self.movable.append(make_box([0.1 * 1.2, 0.1 * 1.2, h * 1.2 * 0.5], [-s, 0, z], [0, 0, 0]))
                self._adjust_box_dynamics(self.movable[-1])
                self._adjust_mass_and_visual(self.movable[-1], 1.2)

        for objId in self.immovable:
            p.changeVisualShape(objId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])
        self.objects = self.immovable + self.movable

    def _setup_experiment(self):
        # set gravity
        p.setGravity(0, 0, -10)
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self._setup_gripper()
        self._setup_objects()

        if self.level in [Levels.RANDOM, Levels.FREESPACE] + selected_levels:
            self.set_camera_position([0, 0])
        elif int(self.level) >= Levels.NO_CLUTTER:
            self.set_camera_position([0.4, 0.], yaw=-90, pitch=-80)
        else:
            self.set_camera_position([0.5, 0.3], yaw=-75, pitch=-80)

        self.state = self._obs()
        self._draw_state()

    def _adjust_mass_and_visual(self, obj, m):
        # adjust the mass of a pybullet object and indicate it via the color
        p.changeVisualShape(obj, -1, rgbaColor=[1 - m / 3, 0.8 - m / 3, 0.2, 0.8])
        p.changeDynamics(obj, -1, mass=m)

    def _adjust_box_dynamics(self, obj):
        p.changeDynamics(obj, -1, lateralFriction=0.8, spinningFriction=0.05, rollingFriction=0.01)

    def _setup_gripper(self):
        # orientation of the end effector (pointing down)
        # TODO allow gripper to change yaw?
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        # use a floating gripper
        self.gripperId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "panda_gripper.urdf"), self.init,
                                    self.endEffectorOrientation)
        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_A, lateralFriction=2)
        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_B, lateralFriction=2)

        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.init, childFrameOrientation=self.endEffectorOrientation)

        self.close_gripper()
        self._make_robot_translucent(self.gripperId)

    def get_ee_contact_info(self, bodyId):
        return p.getContactPoints(self.gripperId, bodyId)

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]
        reaction_torque = [0, 0, 0]

        ee_pos = self._observe_ee(return_z=True)
        for objectId in self.objects:
            contactInfo = self.get_ee_contact_info(objectId)
            for i, contact in enumerate(contactInfo):
                f_contact = get_total_contact_force(contact, False)
                reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]
                # torque wrt end effector position
                pos_vec = np.subtract(contact[ContactInfo.POS_A], ee_pos)
                r_contact = np.cross(pos_vec, f_contact)
                reaction_torque = np.add(reaction_torque, r_contact)

        self._observe_raw_reaction_force(info, reaction_force, reaction_torque, visualize)

    def reset(self):
        for _ in range(1000):
            p.stepSimulation()

        self.open_gripper()
        if self.gripperConstraint:
            p.removeConstraint(self.gripperConstraint)

        for obj in self.immovable + self.movable:
            p.removeBody(obj)
        self._setup_objects()

        p.resetBasePositionAndOrientation(self.gripperId, self.init, self.endEffectorOrientation)
        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.init, childFrameOrientation=self.endEffectorOrientation)
        self.close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('x0', self.get_ee_pos(self.state), color=(0, 1, 0))
        self.contact_detector.clear()
        return np.copy(self.state)


class ObjectRetrievalEnv(FloatingGripperEnv):
    nu = 2
    nx = 2

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)']

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :2] - other_state[:, :2]
        return dpos,

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1])

    def __init__(self, goal=(0.5, -0.3, 0), init=(-.0, 0.0), **kwargs):
        # here goal is the initial pose of the target object
        super(FloatingGripperEnv, self).__init__(goal=goal, init=init, camera_dist=0.8, **kwargs)

    def _set_goal(self, goal):
        if len(goal) != 3:
            goal = [goal[0], goal[1], 0]
        # ignore the pusher position
        self.goal = np.array(goal)
        if self._debug_visualizations[DebugVisualization.GOAL]:
            self._dd.draw_point('goal', [goal[0], goal[1], FIXED_Z])

    def _obs(self):
        return super(ObjectRetrievalEnv, self)._obs()[:2]

    def _draw_state(self):
        pos = self.get_ee_pos(self.state)
        self._dd.draw_point('state', pos)

    def _setup_objects(self):
        self.immovable = []
        self.movable = []
        z = 0.1
        h = 2 if self.extrude_objects_in_z else 0.15
        separation = 0.7

        self.immovable.append(make_box([0.7, 0.1, h], [1.1, 0, z], [0, 0, -np.pi / 2]))
        self.immovable.append(make_box([0.7, 0.1, h], [0.5, -separation, z], [0, 0, 0]))
        self.immovable.append(make_box([0.7, 0.1, h], [0.5, separation, z], [0, 0, 0]))
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.target_object_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbCrackerBox', "model.urdf"),
                                           [self.goal[0], self.goal[1], z],
                                           p.getQuaternionFromEuler([0, 0, self.goal[2]]), flags=flags)
        p.changeDynamics(self.target_object_id, -1, mass=10)
        p.changeDynamics(self.planeId, -1, lateralFriction=0.6, spinningFriction=0.8)

        self.movable.append(self.target_object_id)
        if self.level == Levels.NO_CLUTTER:
            pass
        elif self.level == Levels.SIMPLE_CLUTTER:
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbTomatoSoupCan', "model.urdf"),
                                           [0.3, 0., z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbTomatoSoupCan', "model.urdf"),
                                           [0.2, -0.3, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
        elif self.level == Levels.FLAT_BOX:
            p.changeDynamics(self.planeId, -1, lateralFriction=0.6, spinningFriction=0.01)
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbTomatoSoupCan', "model.urdf"),
                                           [0.2, -0.1, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"),
                                           [0.25, -0.2, z],
                                           p.getQuaternionFromEuler([0, 0, -1]), flags=flags, globalScaling=1))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMasterChefCan', "model.urdf"),
                                           [0.3, 0.2, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbPottedMeatCan', "model.urdf"),
                                           [0.34, 0.05, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=1.5))
        elif self.level == Levels.BEHIND_CAN:
            p.changeDynamics(self.planeId, -1, lateralFriction=0.6, spinningFriction=0.01)
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbTomatoSoupCan', "model.urdf"),
                                           [0.15, 0.15, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMasterChefCan', "model.urdf"),
                                           [0.15, -0.05, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
        elif self.level == Levels.IN_BETWEEN:
            p.changeDynamics(self.planeId, -1, lateralFriction=0.6, spinningFriction=0.01)
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbTomatoSoupCan', "model.urdf"),
                                           [0.15, 0.12, z],
                                           p.getQuaternionFromEuler([0, 0, 0]), flags=flags, globalScaling=2))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"),
                                           [0.21, 0.21, z],
                                           p.getQuaternionFromEuler([0, 0, 0.6]), flags=flags, globalScaling=1))
            self.movable.append(p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbPottedMeatCan', "model.urdf"),
                                           [0.3, -0.05, z],
                                           p.getQuaternionFromEuler([0, 0, 0.6]), flags=flags, globalScaling=1.5))

        for objId in self.immovable:
            p.changeVisualShape(objId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])
        self.objects = self.immovable + self.movable

    def evaluate_cost(self, state, action=None):
        return None, False


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class ExperimentRunner(PybulletSim):
    def __init__(self, env: ArmEnv, ctrl, save_dir=DIR, **kwargs):
        reaction_dim = 2 if isinstance(env, PlanarArmEnv) else 3
        super(ExperimentRunner, self).__init__(env, ctrl, save_dir=save_dir, reaction_dim=reaction_dim, **kwargs)


class ArmDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {ArmEnv: ArmLoader, ArmJointEnv: ArmLoader, PlanarArmEnv: ArmLoader, FloatingGripperEnv: ArmLoader,
                      ObjectRetrievalEnv: ArmLoader}
        return loader_map.get(env_type, None)


def pt_to_config_dist(env, max_robot_radius, configs, pts):
    M = configs.shape[0]
    N = pts.shape[0]
    dist = torch.zeros((M, N), dtype=pts.dtype, device=pts.device)

    orig_pos, orig_orientation = p.getBasePositionAndOrientation(env.robot_id)
    z = orig_pos[2]

    # to speed up distance checking, we compute distance from center of robot config to point
    # and avoid the expensive check of distance to surface for those that are too far away
    center_dist = torch.cdist(configs[:, :2].view(-1, 2), pts[:, :2].view(-1, 2))

    # if visualize:
    #     for i, pt in enumerate(pts):
    #         env._dd.draw_point(f't{i}', pt, color=(1, 0, 0), height=z)

    for i in range(M):
        p.resetBasePositionAndOrientation(env.robot_id, [configs[i][0], configs[i][1], z], orig_orientation)
        for j in range(N):
            if center_dist[i, j] > max_robot_radius:
                # just have to report something > 0
                dist[i, j] = 1
            else:
                closest = closest_point_on_surface(env.robot_id, [pts[j][0], pts[j][1], z])
                dist[i, j] = closest[ContactInfo.DISTANCE]

    p.resetBasePositionAndOrientation(env.robot_id, orig_pos, orig_orientation)
    return dist


class ArmPointToConfig(PlanarPointToConfig):
    def __init__(self, env):
        # try loading cache
        fullname = os.path.join(cfg.DATA_DIR, f'arm_point_to_config.pkl')
        if os.path.exists(fullname):
            super(ArmPointToConfig, self).__init__(*torch.load(fullname))
        else:
            import trimesh
            hand = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/hand.obj"))
            lf = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/finger.obj"))
            lf.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0.0584]))
            rf = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/finger.obj"))
            rf.apply_transform(
                trimesh.transformations.concatenate_matrices(trimesh.transformations.euler_matrix(0, 0, np.pi),
                                                             trimesh.transformations.translation_matrix(
                                                                 [0, 0, 0.0584])))
            mesh = trimesh.util.concatenate([hand, lf, rf])
            # TODO get resting orientation from environment?
            mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi / 2, 0))
            # cache points inside bounding box of robot to accelerate lookup
            min_x, min_y = mesh.bounding_box.bounds[0, :2]
            max_x, max_y = mesh.bounding_box.bounds[1, :2]
            cache_resolution = 0.001
            # create mesh grid
            x = np.arange(min_x, max_x + cache_resolution, cache_resolution)
            y = np.arange(min_y, max_y + cache_resolution, cache_resolution)
            cache_y_len = len(y)

            orig_pos, orig_orientation = p.getBasePositionAndOrientation(env.robot_id)
            p.resetBasePositionAndOrientation(env.robot_id, [0, 0, 0], orig_orientation)
            d = np.zeros((len(x), len(y)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    closest = closest_point_on_surface(env.robot_id, [xi, yj, 0])
                    d[i, j] = closest[ContactInfo.DISTANCE]
            d_cache = d.reshape(-1)
            data = [d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len]
            torch.save(data, fullname)
            super(ArmPointToConfig, self).__init__(*data)
