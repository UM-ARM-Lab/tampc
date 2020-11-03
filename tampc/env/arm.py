import logging
import math
import pybullet as p
import time
import enum
import torch

import numpy as np
from tampc.env.pybullet_env import PybulletEnv, get_total_contact_force, ContactInfo
from tampc.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource
from tampc.env.pybullet_sim import PybulletSim

logger = logging.getLogger(__name__)

DIR = "arm"

kukaEndEffectorIndex = 6
pandaNumDofs = 7


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
    ACTION = 3
    REACTION_MINI_STEP = 4
    REACTION_IN_STATE = 5


class ReactionForceStrategy(enum.IntEnum):
    MAX_OVER_CONTROL_STEP = 0
    MAX_OVER_MINI_STEPS = 1
    AVG_OVER_MINI_STEPS = 2
    MEDIAN_OVER_MINI_STEPS = 3


class ArmEnv(PybulletEnv):
    """To start with we have a fixed gripper orientation so the state is 3D position only"""
    nu = 3
    nx = 6
    MAX_FORCE = 5 * 240
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
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        dreaction = state[:, 3:] - other_state[:, 3:]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1, 0, 0, 0])

    @staticmethod
    def state_distance(state_difference):
        return state_difference[:, :3].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$', 'd$z_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def control_similarity(u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, goal=(0.8, 0.0, 0.5), init=(0.3, 0.6, 0.2),
                 environment_level=0, sim_step_wait=None, mini_steps=50, wait_sim_steps_per_mini_step=20,
                 debug_visualizations=None, dist_for_done=0.02,
                 reaction_force_strategy=ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS, **kwargs):
        """
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
        super().__init__(**kwargs, default_debug_height=0.1, camera_dist=1.0)
        self._dd.toggle_3d(True)
        self.level = environment_level
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy
        self.dist_for_done = dist_for_done

        # initial config
        self.goal = None
        self.init = None
        self.armId = None

        self._debug_visualizations = {
            DebugVisualization.ACTION: True,
            DebugVisualization.REACTION_MINI_STEP: True,
            DebugVisualization.REACTION_IN_STATE: True,
        }
        if debug_visualizations is not None:
            self._debug_visualizations.update(debug_visualizations)

        # avoid the spike at the start of each mini step from rapid acceleration
        self._steps_since_start_to_get_reaction = 5
        self._clear_state_between_control_steps()

        self.set_task_config(goal, init)
        self._setup_experiment()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()

    # --- initialization and task configuration
    def _clear_state_between_control_steps(self):
        self._sim_step = 0
        self._mini_step_contact = {'full': np.zeros((self.mini_steps, 3)), 'mag': np.zeros(self.mini_steps)}
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
        self.goal = np.array(goal)
        self._dd.draw_point('goal', self.goal)

    def _set_init(self, init):
        # initial position of end effector
        self.init = init
        self._dd.draw_point('init', self.init, color=(0, 1, 0.2))
        if self.armId is not None:
            self._calculate_init_joints()

    def _calculate_init_joints(self):
        self.initJoints = list(p.calculateInverseKinematics(self.armId,
                                                            self.endEffectorIndex,
                                                            self.init,
                                                            self.endEffectorOrientation))

    # def _open_gripper(self):
    #     p.resetJointState(self.armId, PandaGripperID.FINGER_A, self.FINGER_OPEN)
    #     p.resetJointState(self.armId, PandaGripperID.FINGER_B, self.FINGER_OPEN)
    #
    # def _close_gripper(self):
    #     p.setJointMotorControlArray(self.armId,
    #                                 [PandaGripperID.FINGER_A, PandaGripperID.FINGER_B],
    #                                 p.POSITION_CONTROL,
    #                                 targetPositions=[self.FINGER_CLOSED, self.FINGER_CLOSED],
    #                                 forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self._setup_gripper()

        self.walls = []
        if self.level == 0:
            pass
        elif self.level == 1:
            half_extents = [0.2, 0.05, 0.3]
            colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visId = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 0.8])
            wallId = p.createMultiBody(0, colId, visId, basePosition=[0.5, 0.4, 0.2],
                                       baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
            self.walls.append(wallId)

        for wallId in self.walls:
            p.changeVisualShape(wallId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])

        self.set_camera_position([0, 0])
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

        self._calculate_init_joints()
        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        # self._open_gripper()
        # self._close_gripper()

        # make arm translucent
        visual_data = p.getVisualShapeData(self.armId)
        for link in visual_data:
            link_id = link[1]
            if link_id == -1:
                continue
            rgba = list(link[7])
            rgba[3] = 0.4
            p.changeVisualShape(self.armId, link_id, rgbaColor=rgba)

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
        pos = link_info[4]
        return pos

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
            return self._reaction_force[:3]

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]

        # TODO try alternatively using enableJointForceTorqueSensor on the end effector (last joint?)

        # compute contact force between end effector and all bodies in the world
        bodies = [self.planeId] + self.walls
        for bodyId in bodies:
            contactInfo = p.getContactPoints(bodyId, self.armId, linkIndexB=self.endEffectorIndex)
            info['npb'] = len(contactInfo)
            for i, contact in enumerate(contactInfo):
                f_contact = get_total_contact_force(contact, False)
                reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]

                name = 'r{}_{}'.format(bodyId, i)
                if abs(contact[ContactInfo.NORMAL_MAG]) > abs(self._largest_contact.get(name, 0)):
                    self._largest_contact[name] = contact[ContactInfo.NORMAL_MAG]
                    self._dd.draw_contact_point(name, contact, False)

        self._observe_raw_reaction_force(info, reaction_force, visualize)

    def _observe_info(self, visualize=True):
        info = {}

        self._observe_additional_info(info, visualize)
        self._sim_step += 1

        for key, value in info.items():
            if key not in self._contact_info:
                self._contact_info[key] = []
            self._contact_info[key].append(value)

    def _observe_raw_reaction_force(self, info, reaction_force, visualize=True):
        # save reaction force
        name = 'r'
        info[name] = reaction_force
        reaction_force_size = np.linalg.norm(reaction_force)
        # see if we should save it as the reaction force for this mini-step
        mini_step, step_since_start = divmod(self._sim_step, self.wait_sim_step_per_mini_step)
        if step_since_start is self._steps_since_start_to_get_reaction:
            self._mini_step_contact['full'][mini_step] = reaction_force
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
        info['reaction'] = self._observe_reaction_force()
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
        # self._close_gripper()

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
        dz = action[2] * self.MAX_PUSH_DIST
        return dx, dy, dz

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = np.copy(self.state)
        dx, dy, dz = self._unpack_action(action)

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)

        ee_pos = self.get_ee_pos(old_state)
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz))
        self._dd.draw_point('final eepos', final_ee_pos, color=(1, 0.5, 0.5))
        # print("current {} desired {}".format(ee_pos, final_ee_pos))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        # self._setup_ee()

        for i in self.armInds:
            p.resetJointState(self.armId, i, self.initJoints[i])
        # self._open_gripper()
        # self._close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        pos = self.get_ee_pos(self.state)
        self._dd.draw_point('x0', pos, color=(0, 1, 0))
        return np.copy(self.state)


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class ArmRunner(PybulletSim):
    def __init__(self, env: ArmEnv, ctrl, save_dir=DIR, **kwargs):
        super(ArmRunner, self).__init__(env, ctrl, save_dir=save_dir, **kwargs)


class ArmDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {ArmEnv: ArmLoader}
        return loader_map.get(env_type, None)
