import pybullet as p
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time

from hybrid_sysid import cfg, simulation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class Controller:
    def __init__(self, push_magnitude):
        self.goal = None
        self.block_width = 0.085
        self.push_magnitude = push_magnitude

    def set_goal(self, goal):
        self.goal = goal[0:2]

    def command(self, obs):
        x, y, xb, yb, yaw = obs
        to_goal = np.subtract(self.goal, (xb, yb))
        logger.info(to_goal)
        desired_pusher_pos = np.subtract((xb, yb), to_goal / np.linalg.norm(to_goal) * self.block_width)
        dpusher = np.subtract(desired_pusher_pos, (x, y))
        ranMag = 0.4
        return np.append((dpusher / np.linalg.norm(dpusher) + (
        np.random.uniform(-ranMag, ranMag), np.random.uniform(-ranMag, ranMag))) * self.push_magnitude, 0)


class InteractivePush(simulation.PyBulletSim):

    def __init__(self, num_frames=4000, save_dir='kuka_push', observation_period=10,
                 push_angle=0.1, push_step=0.03, random_push_angle=0.05,
                 use_rng=False,
                 **kwargs):

        super(InteractivePush, self).__init__(save_dir=save_dir, num_frames=num_frames, **kwargs)
        self.theta = push_angle
        self.use_rng = use_rng
        self.bd = random_push_angle
        self.observation_period = observation_period

        # TODO more options
        self.ctrl = Controller(push_step)

        # plotting
        self.fig = None
        self.axes = None

    def _setup_experiment(self):
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)
        self.initPusherPos = [0.3, 0.2, 0.05]
        self.pusherId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "pusher.urdf"), self.initPusherPos)

        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85, cameraTargetPosition=[0, 0, 1])

        # add the block - we'll reset its position later
        self.initBlockPos = [0.1, 0.1, 0]
        self.blockId = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), self.initBlockPos)

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

        self.goal = [-0.6, 1.1, 0.1]
        goalVisualWidth = 0.15 / 2
        p.addUserDebugLine(np.add(self.goal, [0, -goalVisualWidth, 0]), np.add(self.goal, [0, goalVisualWidth, 0]),
                           [0, 1, 0], 2)
        p.addUserDebugLine(np.add(self.goal, [-goalVisualWidth, 0, 0]), np.add(self.goal, [goalVisualWidth, 0, 0]),
                           [0, 1, 0], 2)

        self.ctrl.set_goal(self.goal)

        # set gravity
        p.setGravity(0, 0, -10)

        # set joint damping
        # set robot init config
        self.robotInitPoseCart = [-0.2, -0.2, 0.01]  # (x,y,z)
        self.orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        self.pusherConstraint = p.createConstraint(self.pusherId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                   [0, 0, 0])

        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, 5))
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

    def _run_experiment(self):

        self._reset_sim()

        x, y, z = self.robotInitPoseCart

        for simTime in range(self.num_frames):
            p.stepSimulation()

            # d = pushDir * self.push_step
            d = self.ctrl.command((x, y) + self._observe_block())
            x, y, z = np.add([x, y, z], d)

            # set end effector pose
            eePos = [x, y, z]
            self._move_pusher(eePos)

            # get pusher info
            pusherPose = p.getBasePositionAndOrientation(self.pusherId)
            x, y, z = pusherPose[0]

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
            p.addUserDebugLine([self.traj[simTime - 1, 0], self.traj[simTime - 1, 1], z], [x, y, z], [1, 0, 0], 2)
            p.addUserDebugLine([self.traj[simTime - 1, 2], self.traj[simTime - 1, 3], z], [xb, yb, z], [0, 0, 1], 2)

            # time.sleep(0.0005)

        # contact force mask - get rid of trash in the beginning
        self.contactForce[:300] = 0

        # compress observations
        self.traj = self._compress_observation(self.traj)
        self.contactForce = self._compress_observation(self.contactForce)
        self.contactCount = self._compress_observation(self.contactCount)

        while True:
            p.stepSimulation()

        return simulation.ReturnMeaning.SUCCESS

    def _compress_observation(self, obs):
        return obs[::self.observation_period]

    def _export_data_dict(self):
        contact_indices = np.where(self.contactCount)[0]

        return {'X': self.traj, 'contact': contact_indices}

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

        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i])
        self.axes[self.traj.shape[1]].plot(self.contactForce)
        self.axes[self.traj.shape[1] + 1].plot(self.contactCount)

    def _reset_sim(self):
        # reset robot to nominal pose
        p.resetBasePositionAndOrientation(self.pusherId, self.initPusherPos, [0, 0, 0, 1])

        # reset block pose
        if self.use_rng:
            # define nominal block pose
            nom_pose = np.array(self.initBlockPos)  # (x,y,theta)

            # define uncertainty bounds
            pos_bd = np.array([0.01, 0.01, 0.1])

            # initialize array
            blockInitPose = np.empty_like(pos_bd)

            for i in range(nom_pose.shape[0]):
                pert = np.random.uniform(-pos_bd[i], pos_bd[i])
                blockInitPose[i] = nom_pose[i] + pert

            blockInitOri = p.getQuaternionFromEuler([0, 0, blockInitPose[-1]])
            p.resetBasePositionAndOrientation(self.blockId, [blockInitPose[0], blockInitPose[1], 0.1], blockInitOri)
        else:
            p.resetBasePositionAndOrientation(self.blockId, self.initBlockPos, [0, 0, 0, 1])

    def _get_push_direction(self, theta, use_random):
        # get unit direction of the push
        if use_random:
            theta = np.random.uniform(low=-self.bd, high=self.bd)

        return np.array([math.sin(theta), math.cos(theta), 0.])


if __name__ == "__main__":
    sim = InteractivePush(mode=p.GUI, plot=True, use_rng=True, save=False)
    sim.run()

    # runs = 0
    # while runs < 30:
    #     runs += sim.run() == simulation.ReturnMeaning.SUCCESS
