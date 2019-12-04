import os
import pybullet as p
import time
import torch

import numpy as np
from hybrid_sysid import simulation, load_data
from matplotlib import pyplot as plt

from meta_contact import cfg, util


class PushLoader(load_data.DataLoader):
    def _process_file_raw_data(self, d):
        xu = d['X']
        y = d['Y']
        cc = d['contact']
        return xu, y, cc


class RawPushDataset(torch.utils.data.Dataset):
    def __init__(self, dir='pushing', mode='all', max_num=None, make_affine=True):
        dl = PushLoader(dir, config=cfg)
        self.XU, self.Y, self.contact = dl.load()
        self.XU = torch.from_numpy(self.XU).double()
        self.Y = torch.from_numpy(self.Y).double()
        self.contact = torch.from_numpy(self.contact).byte()
        if mode is 'contact' or mode is 'nocontact':
            c = self.contact.squeeze()
            if mode is 'nocontact':
                c = c ^ 1
            self.XU = self.XU[c, :]
            self.Y = self.Y[c, :]
            self.contact = self.contact[c, :]

        if make_affine:
            self.XU = load_data.make_affine(self.XU)

        if max_num is not None:
            self.XU = self.XU[:max_num]
            self.Y = self.Y[:max_num]
            self.contact = self.contact[:max_num]

        super().__init__()

    def __len__(self):
        return self.XU.shape[0]

    def __getitem__(self, idx):
        return self.XU[idx], self.Y[idx], self.contact[idx]


class InteractivePush(simulation.PyBulletSim):
    def __init__(self, controller, num_frames=1000, save_dir='pushing', observation_period=1,
                 goal=(-0.6, 1.1), init_pusher=(0.3, 0.2), init_block=(0.1, 0.1), init_yaw=0.,
                 **kwargs):

        super(InteractivePush, self).__init__(save_dir=save_dir, num_frames=num_frames, **kwargs)
        self.observation_period = observation_period
        self.initRestFrames = 20

        self.ctrl = controller

        # initial config
        self.goal = None
        self.initPusherPos = None
        self.initBlockPos = None
        self.initBlockYaw = None
        self.set_task_config(goal, init_pusher, init_block, init_yaw)

        # plotting
        self.fig = None
        self.axes = None

    def set_task_config(self, goal=None, init_pusher=None, init_block=None, init_yaw=None):
        """Change task configuration"""
        if goal is not None:
            self.goal = tuple(goal) + (0.1,)
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
        self.blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), self.initBlockPos)

        # TODO make these environment elements dependent on the "level" of the simulation
        self.walls = []
        # self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-1, 0.5, .0],
        #                              p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
        # self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, -0.5, .0],
        #                              p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
        # self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [-0.3, 2, .0],
        #                              p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))
        # self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [0, 2, .0],
        #                              p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True))
        # self.walls.append(p.loadURDF(os.path.join(cfg.ROOT_DIR, "wall.urdf"), [1.5, 0.5, .0],
        #                              p.getQuaternionFromEuler([0, 0, math.pi / 2]), useFixedBase=True))

        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                                     cameraTargetPosition=[0, 0, 1])
        self._draw_goal()
        self.ctrl.set_goal(self.goal)

        # set gravity
        p.setGravity(0, 0, -10)

        # set joint damping
        # set robot init config
        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   self.initPusherPos)

        return simulation.ReturnMeaning.SUCCESS

    def _draw_goal(self):
        goalVisualWidth = 0.15 / 2
        p.addUserDebugLine(np.add(self.goal, [0, -goalVisualWidth, 0]), np.add(self.goal, [0, goalVisualWidth, 0]),
                           [0, 1, 0], 2)
        p.addUserDebugLine(np.add(self.goal, [-goalVisualWidth, 0, 0]), np.add(self.goal, [goalVisualWidth, 0, 0]),
                           [0, 1, 0], 2)

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, 5))
        self.u = np.zeros((self.num_frames, 2))
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        self.contactForce = np.zeros((self.num_frames,))
        self.contactCount = np.zeros_like(self.contactForce)

        # reset sim time
        self.t = 0
        return simulation.ReturnMeaning.SUCCESS

    def _move_pusher(self, endEffectorPos):
        p.changeConstraint(self.pusherConstraint, endEffectorPos)

    def _observe_block(self):
        blockPose = p.getBasePositionAndOrientation(self.blockId)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
        return xb, yb, yaw

    def _observe_pusher(self):
        pusherPose = p.getBasePositionAndOrientation(self.pusherId)
        return pusherPose[0]

    def _run_experiment(self):

        self._reset_sim()

        for _ in range(self.initRestFrames):
            p.stepSimulation()

        x, y, z = self._observe_pusher()

        for simTime in range(self.num_frames):
            # d = pushDir * self.push_step
            d = self.ctrl.command((x, y) + self._observe_block())
            self.u[simTime, :] = d
            x, y = np.add([x, y], d)

            # set end effector pose
            eePos = [x, y, z]
            self._move_pusher(eePos)

            p.stepSimulation()

            # get pusher info
            x, y, z = self._observe_pusher()

            # get contact information
            contactInfo = p.getContactPoints(self.pusherId, self.blockId)

            # get the net contact force between robot and block
            if len(contactInfo) > 0:
                f_c_temp = 0
                for i in range(len(contactInfo)):
                    f_c_temp += contactInfo[i][9]
                self.contactForce[simTime] = f_c_temp
                self.contactCount[simTime] = len(contactInfo)

            xb, yb, yaw = self._observe_block()
            self.traj[simTime, :] = np.array([x, y, xb, yb, yaw])
            i = max(simTime - 1, 0)
            p.addUserDebugLine([self.traj[i, 0], self.traj[i, 1], z], [x, y, z], [1, 0, 0], 2)
            p.addUserDebugLine([self.traj[i, 2], self.traj[i, 3], z], [xb, yb, z], [0, 0, 1], 2)

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
        XU = self.traj
        state_col_offset = 2
        Y = XU[1:, state_col_offset:]
        # need to throw out the last state to have 1-to-1 with output
        XU = XU[:-1]
        # just output state difference to handle yaw more linearly
        Y[:, :-1] = Y[:, :-1] - XU[:, state_col_offset:state_col_offset + 2]
        Y[:, -1] = util.angular_diff_batch(Y[:, -1], XU[:, state_col_offset + 2])

        contact_flag = self.contactCount > 0
        contact_flag = contact_flag.reshape(len(contact_flag), 1)
        return {'X': XU, 'U': self.u[:-1], 'Y': Y, 'contact': contact_flag[:-1]}

    def start_plot_runs(self):
        axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)',
                     'contact force (N)', 'contact count']
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
            time.sleep(0.01)

        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i])
        self.axes[self.traj.shape[1]].plot(self.contactForce)
        self.axes[self.traj.shape[1] + 1].step(self._compress_observation(self.time), self.contactCount)
        self.fig.canvas.draw()
        time.sleep(0.01)

    def _reset_sim(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pusherId, self.initPusherPos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.blockId, self.initBlockPos,
                                          p.getQuaternionFromEuler([0, 0, self.initBlockYaw]))
