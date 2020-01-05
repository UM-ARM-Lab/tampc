import os
import pybullet as p
import time
import torch
import math

import numpy as np
from matplotlib import pyplot as plt

from meta_contact import cfg
from arm_pytorch_utilities.make_data import datasets
from arm_pytorch_utilities import load_data as load_utils, string, math_utils
from hybrid_sysid import simulation, load_data

import logging

from meta_contact.env.myenv import MyPybulletEnv

logger = logging.getLogger(__name__)


class BlockFace:
    RIGHT = 0
    TOP = 1
    LEFT = 2
    BOT = 3


# TODO This is specific to this pusher and block; how to generalize this?
DIST_FOR_JUST_TOUCHING = 0.096 - 0.00001
MAX_ALONG = 0.075


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
    dxy = math_utils.rotate_wrt_origin(dxy, -block_yaw)
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
    dxy = math_utils.rotate_wrt_origin(dxy, block_yaw)
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


class PushLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, **kwargs):
        super().__init__(file_cfg, *args, **kwargs)

    def _process_file_raw_data(self, d):
        x = d['X']
        u = d['U'][:-1]
        cc = d['contact'][1:]

        # separate option deciding whether to predict output of pusher positions or not
        state_col_offset = 0 if self.config.predict_all_dims else 2
        if self.config.predict_difference:
            dpos = x[1:, state_col_offset:-1] - x[:-1, state_col_offset:-1]
            dyaw = math_utils.angular_diff_batch(x[1:, -1], x[:-1, -1])
            y = np.concatenate((dpos, dyaw.reshape(-1, 1)), axis=1)
        else:
            y = x[1:, state_col_offset:]

        x = x[:-1]
        xu = np.column_stack((x, u))

        # xy = xu[:, 2:4]
        # nxy = xy + y[:, :-1]
        # du = np.linalg.norm(nxy[:-1] - xy[1:], axis=1)

        # potentially many trajectories, get rid of buffer state in between
        mask = d['mask']
        # pack expanded pxu into input if config allows (has to be done before masks); otherwise would use cross-file data)
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

        # xy = xu[:, :2]
        # nxy = xy + u[:, :xy.shape[1]]
        # du = np.linalg.norm(nxy[:-1] - xy[1:], axis=1)
        # dd = du[mask[:-1]]

        # _, res_before, _, _ = np.linalg.lstsq(xu, y)

        xu = xu[mask]
        cc = cc[mask]
        y = y[mask]

        # test that we achieve low residuals (proxy for correct masking - no inter-trajectory relationship)
        # _, res, _, _ = np.linalg.lstsq(xu, y)

        self.config.load_data_info(x, u, y, xu)

        # xy = xu[:, 2:4]
        # nxy = xy + y[:, 2:4]
        # du = np.linalg.norm(nxy[:-1] - xy[1:], axis=1)

        return xu, y, cc


class PushDataset(datasets.DataSet):
    def __init__(self, N=None, data_dir='pushing', preprocessor=None, validation_ratio=0.2,
                 config=load_utils.DataConfig(),
                 **kwargs):
        """
        :param N: total number of data points to use, None for all available in data_dir
        :param data_dir: data directory
        :param predict_difference: whether the output should be the state differences or states
        :param preprocessor: data preprocessor, such as StandardizeVariance
        :param validation_ratio: amount of data set aside for validation
        :param kwargs:
        """

        super().__init__(N=N, input_dim=PushAgainstWallEnv.nx + PushAgainstWallEnv.nu, output_dim=PushAgainstWallEnv.ny,
                         **kwargs)

        self.preprocessor = preprocessor
        self.config = config
        self._data_dir = data_dir
        self._validation_ratio = validation_ratio
        self.make_data()

    def make_parameters(self):
        pass

    def make_data(self):
        full_set = load_utils.LoaderXUYDataset(loader=PushLoader, dirs=self._data_dir, max_num=self.N,
                                               config=self.config)
        train_set, validation_set = load_utils.splitTrainValidationSets(full_set,
                                                                        validation_ratio=self._validation_ratio)

        self.N = len(train_set)

        if self.preprocessor:
            self.preprocessor.fit(train_set)
            # apply on training and validation set
            train_set = self.preprocessor.transform(train_set)
            validation_set = self.preprocessor.transform(validation_set)

        self._train = load_data.get_states_from_dataset(train_set)
        self._val = load_data.get_states_from_dataset(validation_set)

    def data_id(self):
        """String identification for this data"""
        return "{}_N_{}".format(self._data_dir, string.f2s(self.N))


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
        self.set_task_config(goal, init_pusher, init_block, init_yaw)

        # quadratic cost
        self.Q = np.diag([0, 0, 1, 1, 0])
        self.R = np.diag([1 for _ in range(self.nu)])

        self._setup_experiment()
        # start at rest
        for _ in range(self.initRestFrames):
            p.stepSimulation()
        self.state = self._obs()

    def set_task_config(self, goal=None, init_pusher=None, init_block=None, init_yaw=None):
        """Change task configuration"""
        if goal is not None:
            # ignore the pusher position
            self.goal = np.array(tuple(goal) + tuple(goal) + (0.1,))
        if init_pusher is not None:
            self.initPusherPos = tuple(init_pusher) + (0.05,)
        if init_block is not None:
            self.initBlockPos = tuple(init_block) + (0.0325,)
        if init_yaw is not None:
            self.initBlockYaw = init_yaw

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)
        self.pusherId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "pusher.urdf"), self.initPusherPos)
        self.blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), self.initBlockPos,
                                  p.getQuaternionFromEuler([0, 0, self.initBlockYaw]))

        self.walls = []
        if self.level == 0:
            pass
        elif self.level == 1:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, -0.32, .0],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
        elif self.level == 2:
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-1, 0.5, .0],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, -0.32, .0],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0.75, 2, .0],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, 2, .0],
                                         p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
            self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [1.5, 0.5, .0],
                                         p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))

        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                     cameraTargetPosition=[0, 0, 1])
        self._draw_goal()

        # set gravity
        p.setGravity(0, 0, -10)

        # set robot init config
        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)

    def _draw_goal(self):
        goalVisualWidth = 0.15 / 2
        goal = np.concatenate((self.goal[2:4], (0.1,)))
        p.addUserDebugLine(np.add(goal, [0, -goalVisualWidth, 0]), np.add(goal, [0, goalVisualWidth, 0]),
                           [0, 1, 0], 2)
        p.addUserDebugLine(np.add(goal, [-goalVisualWidth, 0, 0]), np.add(goal, [goalVisualWidth, 0, 0]),
                           [0, 1, 0], 2)

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

    STATIC_VELOCITY_THRESHOLD = 1e-3
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
        p.addUserDebugLine(eePos, np.add(eePos, [0, 0, 0.01]), [1, 1, 0], 4)
        # handle trying to go into wall (if we don't succeed)
        # we use a force insufficient for going into the wall
        rest = 1
        while not self._reached_command(eePos) and rest < self.initRestFrames:
            p.stepSimulation()
            rest += 1
        # if rest == self.initRestFrames:
        #     logger.warning("Ran out of steps push")

        # wait until simulation becomes static
        rest = 1
        while not self._static_environment() and rest < self.initRestFrames:
            p.stepSimulation()
            rest += 1
        # if rest == self.initRestFrames:
        #     logger.warning("Ran out of steps static")

    def _evaluate_cost(self, action):
        # TODO consider using different cost function for yaw (wrap) - for example take use a compare_to_goal func
        diff = self.state - self.goal
        cost = diff.T.dot(self.Q).dot(diff)
        done = cost < 0.01
        cost += action.T.dot(self.R).dot(action)
        return cost, done

    def step(self, action):
        old_state = self._obs()
        d = action
        # set end effector pose
        z = self.initPusherPos[2]
        eePos = [old_state[0] + d[0], old_state[1] + d[1], z]

        # execute the action
        self._move_and_wait(eePos)

        # get the net contact force between robot and block
        info = self._observe_contact()
        self.state = self._obs()
        # track trajectory
        p.addUserDebugLine([old_state[0], old_state[1], z], [self.state[0], self.state[1], z], [1, 0, 0], 2)
        p.addUserDebugLine([old_state[2], old_state[3], z], [self.state[2], self.state[3], z], [0, 0, 1], 2)

        cost, done = self._evaluate_cost(action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pusherId, self.initPusherPos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.blockId, self.initBlockPos,
                                          p.getQuaternionFromEuler([0, 0, self.initBlockYaw]))
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

    def __init__(self, init_pusher=0, face=BlockFace.LEFT, **kwargs):
        # initial config
        self.face = face
        super().__init__(init_pusher=init_pusher, **kwargs)

        # quadratic cost
        self.Q = np.diag([1, 1, 0, 0.001])
        self.R = np.diag([1 for _ in range(self.nu)])
        assert self.Q.shape[0] == self.nx
        assert self.R.shape[0] == self.nu

    def set_task_config(self, goal=None, init_pusher=None, init_block=None, init_yaw=None):
        """Change task configuration"""
        if goal is not None:
            # ignore the pusher position
            self.goal = np.array(tuple(goal) + (0.1, 0))
        if init_block is not None:
            self.initBlockPos = tuple(init_block) + (0.0325,)
        if init_yaw is not None:
            self.initBlockYaw = init_yaw
        if init_pusher is not None:
            pos = pusher_pos_for_touching(self.initBlockPos[:2], self.initBlockYaw, face=self.face,
                                          along_face=init_pusher)
            self.initPusherPos = tuple(pos) + (0.05,)

    def _draw_goal(self):
        goalVisualWidth = 0.15 / 2
        goal = np.concatenate((self.goal[:2], (0.1,)))
        p.addUserDebugLine(np.add(goal, [0, -goalVisualWidth, 0]), np.add(goal, [0, goalVisualWidth, 0]),
                           [0, 1, 0], 2)
        p.addUserDebugLine(np.add(goal, [-goalVisualWidth, 0, 0]), np.add(goal, [goalVisualWidth, 0, 0]),
                           [0, 1, 0], 2)

    def _obs(self):
        xb, yb, yaw = self._observe_block()
        x, y, z = self._observe_pusher()
        along, from_center = pusher_pos_along_face((xb, yb), yaw, (x, y), self.face)
        # debugging to make sure we're quasi-static and adjacent to the block
        # logger.debug("dist between pusher and block %f", from_center - DIST_FOR_JUST_TOUCHING)
        return xb, yb, yaw, along

    def step(self, action):
        # TODO consider normalizing control to 0 and 1
        old_state = self._obs()
        # first action is difference in along
        d_along = action[0]
        # second action is how much to go into the perpendicular face (>= 0)
        d_into = max(0, action[1])

        from_center = DIST_FOR_JUST_TOUCHING - d_into
        # restrict sliding of pusher along the face (never to slide off)
        along = np.clip(old_state[3] + d_along, -MAX_ALONG, MAX_ALONG)
        # logger.debug("along %f dalong %f", along, d_along)
        pos = pusher_pos_for_touching(old_state[:2], old_state[2], from_center=from_center, face=self.face,
                                      along_face=along)
        # set end effector pose
        z = self.initPusherPos[2]
        eePos = np.concatenate((pos, (z,)))

        # execute the action
        self._move_and_wait(eePos)

        # get the net contact force between robot and block
        info = self._observe_contact()
        self.state = self._obs()
        # track trajectory
        p.addUserDebugLine([old_state[0], old_state[1], z], [self.state[0], self.state[1], z], [0, 0, 1], 2)

        cost, done = self._evaluate_cost(action)

        return np.copy(self.state), -cost, done, info

    @staticmethod
    def state_names():
        return ['x block (m)', 'y block (m)', 'block rotation (rads)', 'pusher along face (m)']


class InteractivePush(simulation.Simulation):
    def __init__(self, env: PushAgainstWallEnv, controller, num_frames=1000, save_dir='pushing', observation_period=1,
                 **kwargs):

        super(InteractivePush, self).__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.mode = env.mode
        self.observation_period = observation_period

        self.env = env
        self.ctrl = controller

        # plotting
        self.fig = None
        self.axes = None

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

    def _run_experiment(self):

        obs = self._reset_sim()
        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            action = self.ctrl.command(obs)
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)

            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs
            self.contactForce[simTime] = info['contact_force']
            self.contactCount[simTime] = info['contact_count']

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

        self.fig, self.axes = plt.subplots(1, state_dim, figsize=(18, 5))

        for i in range(state_dim):
            self.axes[i].set_xlabel(axis_name[i])

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
        time.sleep(0.01)

    def _reset_sim(self):
        return self.env.reset()
