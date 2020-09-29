import logging
import torch
import rospy

import numpy as np
from tampc import cfg
from tampc.env.pybullet_env import PybulletLoader, handle_data_format_for_state_diff, PybulletEnvDataSource

from tampc_or import cfg as robot_cfg
from tampc_or_msgs.srv import Calibrate, Action, Observe, CalibStaticWrench

# runner imports
from arm_pytorch_utilities import simulation
from tampc.controller import controller, online_controller
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)

DIR = "peg_real"


class PegRealLoader(PybulletLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        # extract the states
        x = x[:, RealPegEnv.STATE_DIMS]

        if self.config.predict_difference:
            y = RealPegEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class RealPegEnv:
    """Interaction with robot via our ROS node; manages what dimensions of the returned observation
    is necessary for dynamics (passed back as state) and which are extra (passed back as info)"""
    nu = 2
    nx = 5
    STATE_DIMS = [0, 1, 2, 3, 4]

    MAX_PUSH_DIST = 0.02

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

    def __init__(self, environment_level=0, dist_for_done=0.02, obs_time=1, log_video=False):
        self.level = environment_level
        self.dist_for_done = dist_for_done

        # initial config
        self.hole = None
        self.initPeg = None
        self.armId = None
        self.boardId = None

        srv_name = robot_cfg.SRV_NAME_MAP[Action]
        rospy.wait_for_service(srv_name)
        self.srv_action = rospy.ServiceProxy(srv_name, Action)

        srv_name = robot_cfg.SRV_NAME_MAP[CalibStaticWrench]
        rospy.wait_for_service(srv_name)
        self.srv_calib_wrench = rospy.ServiceProxy(srv_name, CalibStaticWrench)
        self.obs_time = obs_time

        srv_name = robot_cfg.SRV_NAME_MAP[Observe]
        rospy.wait_for_service(srv_name)
        self.srv_obs = rospy.ServiceProxy(srv_name, Observe)

        # TODO wait for video service and take video

        self._setup_experiment()
        self.state, _ = self._obs()

    def set_task_config(self, hole=None, init_peg=None):
        """Change task configuration; assumes only goal position is specified"""
        # TODO jog to goal to get goal state
        # if hole is not None:
        #     self._set_hole(hole)
        # if init_peg is not None:
        #     self._set_init_peg(init_peg)

    def _set_init_peg(self, init_peg):
        self.initPeg = self.get_ee_pos(self._obs())

    def _setup_experiment(self):
        # TODO send calibration commands to get static force?
        pass

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state, info = self._unpack_raw_obs(self.srv_obs(self.obs_time).obs)
        return state, info

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        if self.hole:
            peg_pos = self.get_ee_pos(state)
            diff = peg_pos[:2] - self.hole
            dist = np.linalg.norm(diff)
            done = dist < self.dist_for_done
        else:
            dist = 0
            done = False
        return (dist * 10) ** 2, done

    # --- control (commonly overridden)
    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        return dx, dy

    def _unpack_raw_obs(self, raw_obs):
        state = np.array([raw_obs[i] for i in self.STATE_DIMS])
        # info is just everything
        return state, raw_obs

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        dx, dy = self._unpack_action(action)

        action_resp = self.srv_action([dx, dy])
        # separate obs into state and info
        self.state, info = self._unpack_raw_obs(action_resp.obs)
        cost, done = self.evaluate_cost(self.state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self):
        # TODO don't do anything on reset?
        self.state, info = self._obs()
        return np.copy(self.state), info


class ExperimentRunner(simulation.Simulation):
    def __init__(self, env: RealPegEnv, ctrl: controller.Controller, num_frames=500, save_dir=DIR,
                 terminal_cost_multiplier=1, stop_when_done=True,
                 **kwargs):

        super().__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.stop_when_done = stop_when_done

        self.env = env
        self.ctrl = ctrl

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
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        self.traj = []
        self.pred_traj = []
        self.u = []
        self.info = []
        self.model_error = []
        return simulation.ReturnMeaning.SUCCESS

    def _finalize_data(self):
        self.traj = np.stack(self.traj)
        if len(self.pred_traj):
            self.pred_traj = np.stack(self.pred_traj)
        self.u.append(np.zeros(self.env.nu))
        self.u = np.stack(self.u)
        # make same length as state trajectory by appending 0 action
        self.info = np.stack(self.info)
        if len(self.model_error):
            self.model_error = np.stack(self.model_error)

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _has_recovery_policy(self):
        return isinstance(self.ctrl, online_controller.OnlineMPPI)

    def _run_experiment(self):
        self.last_run_cost = []
        obs, info = self._reset_sim()
        self.traj = [obs]
        self.info = [info]

        for simTime in range(self.num_frames - 1):
            start = time.perf_counter()

            action = self.ctrl.command(obs, info)

            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.debug("%d cost %-5.2f took %.3fs done %d action %-12s obs %s", simTime, cost,
                         time.perf_counter() - start, done,
                         np.round(action, 2), np.round(obs, 3))

            self.last_run_cost.append(cost)
            self.u.append(action)
            self.traj.append(obs)
            self.info.append(info)

            if self._predicts_state():
                self.pred_traj.append(self.ctrl.predicted_next_state)
                # model error from the previous prediction step (can only evaluate it at the current step)
                self.model_error.append(self.ctrl.prediction_error(obs))

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                break

        self._finalize_data()

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        assert len(self.last_run_cost) == self.u.shape[0]

        return simulation.ReturnMeaning.SUCCESS

    def _export_data_dict(self):
        # output (1 step prediction)
        # only save the full information rather than states to allow changing dynamics dimensions without recollecting
        X = self.info
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        # need to also throw out first step if predicting reaction force since there's no previous state
        mask[0] = 0
        mask[-1] = 0
        return {'X': X, 'U': self.u, 'X_pred': self.pred_traj, 'model error': self.model_error,
                'mask': mask.reshape(-1, 1)}

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

        self.fig, self.axes = plt.subplots(state_dim, 1, sharex='all')
        self.fu, self.au = plt.subplots(ctrl_dim, 1, sharex='all')
        if self._predicts_state():
            self.fd, self.ad = plt.subplots(state_dim, 1, sharex='all')
        # plot of other info
        self.fo, self.ao = plt.subplots(3, 1, sharex='all')
        self.ao[0].set_ylabel('reaction magnitude')
        self.ao[1].set_ylabel('wall contacts')
        self.ao[2].set_ylabel('predicted dynamics_class')

        for i in range(state_dim):
            self.axes[i].set_ylabel(axis_name[i])
            if self._predicts_state():
                self.ad[i].set_ylabel('d' + axis_name[i])
        for i in range(ctrl_dim):
            self.au[i].set_ylabel('$u_{}$'.format(i))

        self.fig.tight_layout()
        self.fu.tight_layout()
        self.fo.tight_layout()

        plt.ion()
        plt.show()

    def _plot_data(self):
        if self.fig is None:
            self.start_plot_runs()
            plt.pause(0.0001)

        t = np.arange(1, self.pred_traj.shape[0])
        for i in range(self.traj.shape[1] - 1):
            self.axes[i].plot(self.traj[:, i], label='true')
            if self._predicts_state():
                self.axes[i].scatter(t, self.pred_traj[1:, i + 1], marker='*', color='k', label='predicted')
                self.ad[i].plot(self.model_error[:, i])
        self.axes[0].legend()

        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        plt.pause(0.0001)

    def _reset_sim(self):
        return self.env.reset()


class PegRealDataSource(PybulletEnvDataSource):
    loader_map = {RealPegEnv: PegRealLoader}

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        return PegRealDataSource.loader_map.get(env_type, None)
