import logging
import torch
import rospy

import numpy as np
from tampc import cfg
from stucco.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource, Env
from stucco.env.real_env import DebugRvizDrawer

from tampc_or import cfg as robot_cfg
from tampc_or_msgs.srv import Action, Observe, CalibStaticWrench

# drawer imports

# runner imports
from arm_pytorch_utilities import simulation
from tampc.controller import controller, online_controller
import matplotlib.pyplot as plt
import time
from math import cos, sin

logger = logging.getLogger(__name__)

DIR = "peg_real"


class PegRealLoader(TrajectoryLoader):
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


class RealPegEnv(Env):
    """Interaction with robot via our ROS node; manages what dimensions of the returned observation
    is necessary for dynamics (passed back as state) and which are extra (passed back as info)"""
    nu = 2
    nx = 5
    STATE_DIMS = [0, 1, 2, 3, 4]

    MAX_PUSH_DIST = 0.02
    RESET_RAISE_BY = 0.025

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)', '$r_x$ (N)', '$r_y$ (N)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        dreaction = state[:, 3:5] - other_state[:, 3:5]
        return dpos, dreaction

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 0, 0, 0])

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

    @classmethod
    @handle_data_format_for_state_diff
    def control_similarity(cls, u1, u2):
        # TODO should probably keep the API numpy only
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, environment_level=0, dist_for_done=0.015, obs_time=1, stub=True):
        self.level = environment_level
        self.dist_for_done = dist_for_done

        # initial config
        self.hole = None
        self.initPeg = None
        self.armId = None
        self.boardId = None

        self.obs_time = obs_time
        if not stub:
            srv_name = robot_cfg.SRV_NAME_MAP[Action]
            rospy.wait_for_service(srv_name)
            self.srv_action = rospy.ServiceProxy(srv_name, Action)

            srv_name = robot_cfg.SRV_NAME_MAP[CalibStaticWrench]
            rospy.wait_for_service(srv_name)
            self.srv_calib_wrench = rospy.ServiceProxy(srv_name, CalibStaticWrench)

            srv_name = robot_cfg.SRV_NAME_MAP[Observe]
            rospy.wait_for_service(srv_name)
            self.srv_obs = rospy.ServiceProxy(srv_name, Observe)

            self.state, _ = self._obs()

    def set_task_config(self, hole=None, init_peg=None):
        """Change task configuration; assumes only goal position is specified"""
        self.hole = hole
        self.initPeg = init_peg
        if hole is not None and len(hole) != 2:
            raise RuntimeError("Expected hole to be (x, y), instead got {}".format(hole))
        if init_peg is not None and len(init_peg) != 2:
            raise RuntimeError("Expected peg to be (x, y), instead got {}".format(init_peg))

    def setup_experiment(self):
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

    def step(self, action, dz=0.):
        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        dx, dy = self._unpack_action(action)

        action_resp = self.srv_action([dx, dy, dz])
        # separate obs into state and info
        self.state, info = self._unpack_raw_obs(action_resp.obs)
        cost, done = self.evaluate_cost(self.state, action)

        return np.copy(self.state), -cost, done, info

    def reset(self, action=None):
        if action is not None:
            dx, dy = self._unpack_action(action)
            action_resp = self.srv_action([dx, dy])
            self.state, info = self._unpack_raw_obs(action_resp.obs)
        else:
            self.state, info = self._obs()
            if self.initPeg is not None:
                # navigate to init peg by first raising and then dropping
                dx = self.initPeg[0] - self.state[0]
                dy = self.initPeg[1] - self.state[1]
                self.srv_action([0, 0, self.RESET_RAISE_BY])
                self.srv_action([dx, dy, self.RESET_RAISE_BY])
                action_resp = self.srv_action([0, 0])
                self.state, info = self._unpack_raw_obs(action_resp.obs)

        return np.copy(self.state), info

    def close(self):
        pass


class ReplayEnv(RealPegEnv):
    def __init__(self, ds, **kwargs):
        self.ds = ds
        self.xu, self.y, _ = ds.training_set(original=True)
        self.x = self.xu[:, :self.nx].cpu().numpy()
        self.t = 0
        super(ReplayEnv, self).__init__(**kwargs)

    def step(self, action):
        if self.t < self.x.shape[0] - 1:
            self.t += 1
        self.state = self.x[self.t]
        cost, done = self.evaluate_cost(self.state, action)
        return np.copy(self.state), -cost, done, None

    def reset(self, action=None):
        self.t = 0
        return np.copy(self.x[self.t]), None


class ExperimentRunner(simulation.Simulation):
    def __init__(self, env: RealPegEnv, ctrl: controller.Controller, num_frames=500, save_dir=DIR,
                 terminal_cost_multiplier=1, stop_when_done=True, spiral_explore=True,
                 **kwargs):

        super().__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.stop_when_done = stop_when_done

        self.env = env
        self.ctrl = ctrl
        self.dd = DebugRvizDrawer()

        # behaviour when we near where we think the hole is
        self.spiral_explore = spiral_explore

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
        self.env.setup_experiment()
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        return simulation.ReturnMeaning.SUCCESS

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _has_recovery_policy(self):
        return isinstance(self.ctrl, online_controller.TAMPC)

    def clear_markers(self):
        self.dd.clear_markers("state_trajectory", delete_all=True)
        self.dd.clear_markers("trap_state", delete_all=False)
        self.dd.clear_markers("trap_action", delete_all=False)
        self.dd.clear_markers("action", delete_all=False)
        self.dd.clear_markers("rollouts", delete_all=False)

    def _run_experiment(self):
        self.last_run_cost = []
        obs, info = self._reset_sim()
        if self.ctrl.goal is not None:
            self.dd.draw_goal(self.ctrl.goal)
        traj = [obs]
        u = []
        infos = [info]
        pred_cls = []
        pred_traj = []
        model_error = []
        model_error_normalized = []

        if isinstance(self.ctrl, online_controller.OnlineMPC):
            XU, _, _ = self.ctrl.ds.training_set(original=True)
            mean_z = XU[:, 2].mean().item()

        for simTime in range(self.num_frames - 1):
            self.dd.draw_text("step", "{}".format(simTime), 1)
            start = time.perf_counter()

            if isinstance(self.ctrl, online_controller.OnlineMPC):
                # set the observed z value to the average of training data due to problems from preprocessing
                obs[2] = mean_z

            action = self.ctrl.command(obs, info)

            # visualization before taking action
            if isinstance(self.ctrl, online_controller.OnlineMPC):
                pred_cls.append(self.ctrl.dynamics_class)
                self.dd.draw_text("dynamics class", "dyn cls {}".format(self.ctrl.dynamics_class), 2)

                if isinstance(self.ctrl, online_controller.TAMPC):
                    mode_text = "recovery" if self.ctrl.autonomous_recovery_mode else (
                        "local" if self.ctrl.using_local_model_for_nonnominal_dynamics else "nominal")
                    self.dd.draw_text("control mode", mode_text, 3)
                if self.ctrl.trap_set is not None:
                    self.dd.draw_trap_set(self.ctrl.trap_set)

                # print current state; the model prediction error is last time step's prediction about the current state
                model_pred_error = 0
                if self.ctrl.diff_predicted is not None:
                    model_pred_error = self.ctrl.diff_predicted.norm().item()
                self.dd.draw_state(obs, simTime, model_pred_error, action=action)
                model_error_normalized.append(model_pred_error)

                rollouts = self.ctrl.get_rollouts(obs)
                self.dd.draw_rollouts(rollouts)

            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.info("%d cost %-5.2f took %.3fs done %d action %-12s obs %s", simTime, cost,
                        time.perf_counter() - start, done,
                        np.round(action, 2), np.round(obs, 3))

            self.last_run_cost.append(cost)
            u.append(action)
            traj.append(obs)
            infos.append(info)

            if self._predicts_state():
                pred_traj.append(self.ctrl.predicted_next_state)
                # model error from the previous prediction step (can only evaluate it at the current step)
                model_error.append(self.ctrl.prediction_error(obs))

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                break

        self.traj = np.stack(traj)
        if len(pred_traj):
            self.pred_traj = np.stack(pred_traj)
            self.pred_cls = np.stack(pred_cls)
        else:
            self.pred_traj = np.array([])
            self.pred_cls = np.array([])
        u.append(np.zeros(self.env.nu))
        self.u = np.stack(u)
        # make same length as state trajectory by appending 0 action
        self.info = np.stack(infos)
        if len(model_error):
            self.model_error = np.stack(model_error)
            self.model_error_normalized = np.stack(model_error_normalized)
        else:
            self.model_error = np.array([])
            self.model_error_normalized = np.array([])

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        assert len(self.last_run_cost) == self.u.shape[0]

        # if we're done (close to goal), use some local techniques to find hole
        if done and isinstance(self.ctrl, online_controller.OnlineMPC):
            logger.info("done, using local controller to insert peg")
            if self.spiral_explore:
                ctrl = SpiralController()
                obs, rew, done, info = self.env.step([0, 0], -self.env.MAX_PUSH_DIST * 0.1)
                stable_z = obs[2]
                found_hole = False
                while not found_hole:
                    for i in range(75):
                        action = ctrl.command(obs)
                        # account for mechanical drift
                        action = np.array(action)
                        mag = np.linalg.norm(action)
                        action[0] -= mag * 0.05
                        obs, rew, done, info = self.env.step(action, -self.env.MAX_PUSH_DIST * 0.1)
                        simTime += 1
                        self.dd.draw_state(obs, simTime, 0, action=action)
                        logger.info("%d stable z %f z %f rz %f", i, np.round(stable_z, 3), np.round(obs[2], 3),
                                    np.round(info[5], 3))
                        if abs(obs[2] - stable_z) > self.env.MAX_PUSH_DIST * 0.05:
                            found_hole = True
                            break
                    ctrl.t = 0
            else:
                P_TERM = 20
                cost = 100
                while cost > 0.0001:
                    dxy = [self.ctrl.goal[0].item() - obs[0], self.ctrl.goal[1].item() - obs[1]]
                    action = np.array(dxy) * P_TERM
                    if np.linalg.norm(action) < 0.2:
                        action *= 0.2 / np.linalg.norm(action)
                    obs, rew, done, info = self.env.step(action)
                    cost = -rew
                    logger.info("dxy %s cost %f", np.round(dxy, 3), cost)

            self.env.srv_action([0, 0, -self.env.MAX_PUSH_DIST * 0.6])

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
                'model error normalized': self.model_error_normalized, 'mask': mask.reshape(-1, 1)}

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


class SpiralController:
    def __init__(self, start_t=1.5, scale=0.018, growth=4):
        self.t = start_t
        self.scale = scale
        self.growth = np.pi / 20 * growth

    def command(self, obs):
        t = self.t
        x = (self.scale * t) * cos(t)
        y = (self.scale * t) * sin(t)

        self.t += self.growth
        return [x, y]


class PegRealDataSource(EnvDataSource):
    loader_map = {RealPegEnv: PegRealLoader}

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        return PegRealDataSource.loader_map.get(env_type, None)
