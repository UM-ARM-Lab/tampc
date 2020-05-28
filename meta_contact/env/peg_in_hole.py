import logging
import math
import os
import pybullet as p
import time
import enum

import numpy as np
import torch
from arm_pytorch_utilities import simulation
from matplotlib import pyplot as plt
from meta_contact import cfg
from meta_contact.env.pybullet_env import PybulletEnv, PybulletLoader, handle_data_format_for_state_diff, DebugDrawer, \
    get_total_contact_force, ContactInfo, PybulletEnvDataSource
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.controller.gating_function import DynamicsClass

logger = logging.getLogger(__name__)

_PEG_MID = 0.10
_EE_Z = _PEG_MID + 0.3

_DIR = "peg_in_hole"


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


class GripperID(enum.IntEnum):
    ANGLE = 7
    FINGER_A = 8
    FINGER_B = 11
    TIP_A = 10
    TIP_B = 13


class PegInHoleEnv(PybulletEnv):
    nu = 2
    nx = 3
    ny = 3
    MAX_FORCE = 500
    MAX_GRIPPER_FORCE = 30
    MAX_TIP_FORCE = 20
    MAX_PUSH_DIST = 0.03
    OPEN_ANGLE = 0.1
    CLOSE_ANGLE = 0.00

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        return dpos,

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1])

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1])
        u_max = np.array([1, 1])
        return u_min, u_max

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, hole=(0.0, 0.0), init_peg=(-0.3, 0.2),
                 environment_level=0, sim_step_wait=None, mini_steps=100, wait_sim_steps_per_mini_step=20,
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
        super().__init__(**kwargs)
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
        self.kukaId = None

        self._dd = DebugDrawer(_PEG_MID)
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

        # quadratic cost
        self.Q = self.state_cost()
        self.R = self.control_cost()

        self.set_task_config(hole, init_peg)
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

    def set_task_config(self, hole=None, init_peg=None):
        """Change task configuration; assumes only goal position is specified #TOOD relax assumption"""
        if hole is not None:
            self._set_hole(hole)
            self._draw_goal()
        if init_peg is not None:
            self._set_init_peg(init_peg)

    def _set_hole(self, hole):
        # ignore the pusher position
        self.hole = np.array(hole)

    def _set_init_peg(self, init_peg):
        self.initPeg = tuple(init_peg) + (_PEG_MID,)
        if self.kukaId is not None:
            self._calculate_init_joints_to_hold_peg()

    def _calculate_init_joints_to_hold_peg(self):
        self.initJoints = list(p.calculateInverseKinematics(self.kukaId,
                                                            self.kukaEndEffectorIndex,
                                                            [self.initPeg[0], self.initPeg[1] + 0.025, _EE_Z],
                                                            self.endEffectorOrientation,
                                                            jointDamping=self.jd))

    def _open_gripper(self):
        p.resetJointState(self.kukaId, GripperID.FINGER_A, -self.OPEN_ANGLE)
        p.resetJointState(self.kukaId, GripperID.FINGER_B, self.OPEN_ANGLE)

    def _close_gripper(self):
        p.setJointMotorControlArray(self.kukaId,
                                    [GripperID.ANGLE, GripperID.FINGER_A, GripperID.FINGER_B, GripperID.TIP_A,
                                     GripperID.TIP_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[0, -self.CLOSE_ANGLE, self.CLOSE_ANGLE, 0, 0],
                                    forces=[self.MAX_FORCE, self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE,
                                            self.MAX_TIP_FORCE, self.MAX_TIP_FORCE])

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.pegId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "pusher.urdf"), self.initPeg)

        # add kuka arm
        self.kukaId = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
        p.resetBasePositionAndOrientation(self.kukaId, [0, -0.500000, 0.070000],
                                          p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # orientation of the end effector (pointing down)
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.kukaEndEffectorIndex = 6
        self.numJoints = p.getNumJoints(self.kukaId)
        # TODO add gripper and initialize it to hold the peg
        # get the joint ids
        self.armInds = [i for i in range(self.kukaEndEffectorIndex + 1)]

        # joint damping coefficents
        # self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]

        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

        for i in self.armInds:
            p.resetJointState(self.kukaId, i, self.initJoints[i])
        self._open_gripper()
        self._close_gripper()

        # adjust dynamics for better stability
        # p.changeDynamics(self.planeId, -1, lateralFriction=0.3, spinningFriction=0.0, rollingFriction=0.0)
        # TODO add board with hole

        self.walls = []
        wall_z = 0.05
        if self.level == 0:
            pass
        elif self.level == 1:
            # TODO add random protrusions to shape
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.55, -0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.8))

        self.set_camera_position([0, 0])
        self._draw_goal()
        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

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

    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""
        pred = self.get_ee_pos(predicted_state)
        c = (0.5, 0, 0.5)
        self._dd.draw_point('ep', pred, c)

    def set_camera_position(self, camera_pos):
        self._dd._camera_pos = camera_pos
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                     cameraTargetPosition=[camera_pos[0], camera_pos[1], 1])

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_goal(self):
        self._dd.draw_point('goal', self.hole)

    def _draw_state(self):
        self._dd.draw_point('state', self.get_ee_pos(self.state))

    def _draw_action(self, action, debug=0):
        pass

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self._observe_ee()
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    def draw_user_text(self, text, location_index=1, left_offset=1):
        if location_index is 0:
            raise RuntimeError("Can't use same location index (0) as cost")
        self._dd.draw_text('user{}'.format(location_index), text, location_index, left_offset)

    # --- set current state
    def set_state(self, state, action=None, block_id=None):
        # TODO implement
        pass

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = np.concatenate((self._observe_ee(), self._observe_reaction_force()))
        return state

    def _observe_ee(self):
        # TODO do I need to get gripper as well?
        link_info = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
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
        contactInfo = p.getContactPoints(self.pegId, self.kukaId)
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
        # whether we arrived at goal (no cost, or if are not in a hole 1)
        # TODO evaluate correctness; note that we don't actually need state or action
        peg_pos = self._observe_peg()
        diff = peg_pos[:2] - self.hole
        done = np.linalg.norm(diff) < self.dist_for_done
        return 0 if done else 1, done

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

    # TODO implement following methods
    # --- control (commonly overridden)
    def _move_pusher(self, end):
        jointPoses = p.calculateInverseKinematics(self.kukaId,
                                                  self.kukaEndEffectorIndex,
                                                  end,
                                                  self.endEffectorOrientation,
                                                  jointDamping=self.jd)

        num_arm_indices = len(self.armInds)
        p.setJointMotorControlArray(self.kukaId, self.armInds, controlMode=p.POSITION_CONTROL,
                                    targetPositions=jointPoses[:num_arm_indices],
                                    targetVelocities=[0] * num_arm_indices,
                                    forces=[self.MAX_FORCE] * num_arm_indices,
                                    positionGains=[0.3] * num_arm_indices,
                                    velocityGains=[1] * num_arm_indices)
        # self._close_gripper()

    def _move_and_wait(self, eePos, steps_to_wait=50):
        # execute the action
        self._move_pusher(eePos)
        p.stepSimulation()
        for _ in range(steps_to_wait):
            self._observe_info()
            logger.info("%s", self._observe_ee())
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
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, _EE_Z))
        self._dd.draw_point('final eepos', final_ee_pos, color=(0, 0.5, 0.5))

        # TODO might have to set start_ee_pos's height to be different from ee_pos
        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            logger.info("goal %s", intermediate_ee_pos)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pegId, self.initPeg, [0, 0, 0, 1])
        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

        for i in self.armInds:
            p.resetJointState(self.kukaId, i, self.initJoints[i])
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
    nx = 3
    ny = 3
    MAX_FORCE = 200
    MAX_GRIPPER_FORCE = 30
    MAX_TIP_FORCE = 20
    MAX_PUSH_DIST = 0.03
    OPEN_ANGLE = 0.1
    CLOSE_ANGLE = 0.00

    def _observe_ee(self):
        gripperPose = p.getBasePositionAndOrientation(self.kukaId)
        return gripperPose[0]

    def _calculate_init_joints_to_hold_peg(self):
        self.initJoints = list(p.calculateInverseKinematics(self.kukaId,
                                                            self.kukaEndEffectorIndex,
                                                            [self.initPeg[0], self.initPeg[1] - 0.025, _EE_Z],
                                                            self.endEffectorOrientation,
                                                            jointDamping=self.jd))

    def _open_gripper(self):
        p.resetJointState(self.kukaId, GripperID.FINGER_A, -self.OPEN_ANGLE)
        p.resetJointState(self.kukaId, GripperID.FINGER_B, self.OPEN_ANGLE)

    def _close_gripper(self):
        p.setJointMotorControlArray(self.kukaId,
                                    [GripperID.ANGLE, GripperID.FINGER_A, GripperID.FINGER_B, GripperID.TIP_A,
                                     GripperID.TIP_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[0, -self.CLOSE_ANGLE, self.CLOSE_ANGLE, 0, 0],
                                    forces=[self.MAX_FORCE, self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE,
                                            self.MAX_TIP_FORCE, self.MAX_TIP_FORCE])

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.pegId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "pusher.urdf"), self.initPeg)

        # TODO use a floating gripper
        self.kukaId = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
        p.resetBasePositionAndOrientation(self.kukaId, [0, -0.500000, 0.070000],
                                          p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        self.pusherConstraint = p.createConstraint(self.kukaId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)

        # orientation of the end effector (pointing down)
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.kukaEndEffectorIndex = 6
        self.numJoints = p.getNumJoints(self.kukaId)
        # TODO add gripper and initialize it to hold the peg
        # get the joint ids
        self.armInds = [i for i in range(self.kukaEndEffectorIndex + 1)]

        # joint damping coefficents
        # self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]

        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

        for i in self.armInds:
            p.resetJointState(self.kukaId, i, self.initJoints[i])
        self._open_gripper()
        self._close_gripper()

        # adjust dynamics for better stability
        # p.changeDynamics(self.planeId, -1, lateralFriction=0.3, spinningFriction=0.0, rollingFriction=0.0)
        # TODO add board with hole

        self.walls = []
        wall_z = 0.05
        if self.level == 0:
            pass
        elif self.level == 1:
            # TODO add random protrusions to shape
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.55, -0.25, wall_z],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                         globalScaling=0.8))

        self.set_camera_position([0, 0])
        self._draw_goal()
        self.state = self._obs()
        self._draw_state()

        # set gravity
        p.setGravity(0, 0, -10)

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = self._obs()
        dx, dy = self._unpack_action(action)

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)

        ee_pos = self.get_ee_pos(old_state)
        # TODO apply force into the floor?
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, _EE_Z))
        self._dd.draw_point('final eepos', final_ee_pos, color=(0, 0.5, 0.5))

        # TODO might have to set start_ee_pos's height to be different from ee_pos
        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def _move_pusher(self, end):
        p.changeConstraint(self.pusherConstraint, end, maxForce=self.MAX_FORCE)
        self._close_gripper()

    def reset(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pegId, self.initPeg, [0, 0, 0, 1])
        # reset joint states to nominal pose
        self._calculate_init_joints_to_hold_peg()

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


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class PegInHole(simulation.Simulation):
    def __init__(self, env: PegInHoleEnv, controller, num_frames=1000, save_dir=_DIR,
                 terminal_cost_multiplier=1, stop_when_done=True, visualize_rollouts=True,
                 visualize_action_sample=False,
                 **kwargs):

        super(PegInHole, self).__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        env.verify_dims()
        self.mode = env.mode
        self.stop_when_done = stop_when_done
        self.visualize_rollouts = visualize_rollouts
        self.visualize_action_sample = visualize_action_sample

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

        self.fu_sample = None
        self.au_sample = None

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        # don't know where the hole is
        # self.ctrl.set_goal(self.env.hole)
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
        self.pred_cls = np.zeros_like(self.wall_contact)
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.reaction_force, self.wall_contact, self.model_error, self.time, self.pred_cls = (
            data[:frame] for data
            in
            (self.traj, self.u,
             self.reaction_force,
             self.wall_contact,
             self.model_error,
             self.time, self.pred_cls))

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        info = None

        U_nom_unadjusted = None

        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            self.env.draw_user_text("{}".format(simTime), 1)

            start = time.perf_counter()

            if U_nom_unadjusted is not None:
                # display what our action would've been if we did not adjust our control
                U_nom_adjusted = self.ctrl.mpc.U
                self.ctrl.mpc.U = U_nom_unadjusted
                action_unadjusted = self.ctrl.command(obs, info)
                self.env._draw_action(action_unadjusted, debug=4)
                self.ctrl.mpc.U = U_nom_adjusted

            action = self.ctrl.command(obs, info)

            # visualizations before taking action
            if self._predicts_dynamics_cls():
                self.pred_cls[simTime] = self.ctrl.dynamics_class
                self.env.draw_user_text("dyn cls {}".format(self.ctrl.dynamics_class), 2)
                self.env.draw_user_text("recovery" if self.ctrl.autonomous_recovery_mode else "", 3)
                for i in range(4):
                    dynamics_class_pred = self.ctrl.dynamics_class_prediction[i]
                    nom_count = (dynamics_class_pred == DynamicsClass.NOMINAL).sum()
                    text = "nom: {:.2f}".format(nom_count.float() / len(dynamics_class_pred))
                    self.env.draw_user_text("t={} {}".format(i, text), 4 + i)
            if self.visualize_action_sample and isinstance(self.ctrl, controller.MPPI_MPC):
                self._plot_action_sample(self.ctrl.mpc.perturbed_action)
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
                self.pred_traj[simTime + 1, :] = self.ctrl.predicted_next_state
                # model error from the previous prediction step (can only evaluate it at the current step)
                self.model_error[simTime, :] = self.ctrl.prediction_error(obs)
                self.env.visualize_prediction_error(self.ctrl.predicted_next_state.reshape(-1))

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
        mask[0] = 0
        mask[-1] = 0
        u_norm = np.linalg.norm(self.u, axis=1)
        # shift by 1 since the control at t-1 affects the model error at t
        u_norm = np.roll(u_norm, 1).reshape(-1, 1)
        scaled_model_error = np.divide(self.model_error, u_norm, out=np.zeros_like(self.model_error), where=u_norm != 0)
        return {'X': X, 'U': self.u, 'reaction': self.reaction_force, 'model error': self.model_error,
                'scaled model error': scaled_model_error, 'wall contact': self.wall_contact.reshape(-1, 1),
                'mask': mask.reshape(-1, 1), 'predicted dynamics_class': self.pred_cls.reshape(-1, 1)}

    def _start_plot_action_sample(self):
        self.fu_sample, self.au_sample = plt.subplots(self.env.nu, 1)
        u_min, u_max = self.env.get_control_bounds()
        u_names = self.env.control_names()
        for i, name in enumerate(u_names):
            self.au_sample[i].set_xbound(u_min[i], u_max[i])
            self.au_sample[i].set_xlabel(name)
        plt.ion()
        plt.show()

    def _plot_action_sample(self, action):

        if self.fu_sample is None:
            self._start_plot_action_sample()
            plt.pause(0.0001)

        # for now just consider the sample over first step
        u = action[:, 0, :].cpu().numpy()
        for i in range(self.env.nu):
            self.au_sample[i].clear()
            self.au_sample[i].hist(u[:, i])
        plt.pause(0.0001)

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
        self.fo, self.ao = plt.subplots(3, 1, sharex=True)
        self.ao[0].set_ylabel('reaction magitude')
        self.ao[1].set_ylabel('wall contacts')
        self.ao[2].set_ylabel('predicted dynamics_class')

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
        self.ao[2].plot(self.pred_cls)

        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        plt.pause(0.0001)

    def _reset_sim(self):
        return self.env.reset()


class PegInHoleDataSource(PybulletEnvDataSource):
    loader_map = {PegInHoleEnv: PegLoader}

    @staticmethod
    def _default_data_dir():
        return _DIR

    @staticmethod
    def _loader_map(env_type):
        return PegInHoleDataSource.loader_map.get(env_type, None)
