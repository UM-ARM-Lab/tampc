import time
import torch

import numpy as np
from arm_pytorch_utilities import controller
from arm_pytorch_utilities import simulation
from matplotlib import pyplot as plt
from stucco import cfg
from stucco.env.pybullet_env import PybulletEnv, logger


class PybulletSim(simulation.Simulation):
    def __init__(self, env: PybulletEnv, ctrl: controller.Controller, num_frames=1000, save_dir="base",
                 terminal_cost_multiplier=1, stop_when_done=True,
                 visualize_rollouts=True,
                 visualize_action_sample=False,
                 visualize_prediction_error=False,
                 reaction_dim=2,
                 **kwargs):

        super().__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        env.verify_dims()
        self.mode = env.mode
        self.stop_when_done = stop_when_done
        self.visualize_rollouts = visualize_rollouts
        self.visualize_action_sample = visualize_action_sample
        self.visualize_prediction_error = visualize_prediction_error
        self.reaction_dim = reaction_dim

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
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, self.env.nx))
        self.pred_traj = np.zeros_like(self.traj)
        self.u = np.zeros((self.num_frames, self.env.nu))
        self.model_error = np.zeros_like(self.traj)
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        self.pred_cls = np.zeros((self.num_frames,))
        self.info = {}
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.model_error, self.time, self.pred_cls = (
            data[:frame] for data
            in
            (self.traj, self.u,
             self.model_error,
             self.time, self.pred_cls))

    def visualize_before_action(self, simTime, obs):
        contact_set = getattr(self.ctrl, 'contact_set', None)
        if contact_set is not None:
            self.env.visualize_contact_set(contact_set)

    def visualize_after_action(self, simTime, obs):
        pass

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        info = None

        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            self.env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

            start = time.perf_counter()

            action = self.ctrl.command(obs, info)

            self.visualize_before_action(simTime, obs)

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
            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs

            for name, value in info.items():
                if name not in self.info:
                    self.info[name] = []
                self.info[name].append(value)

            self.visualize_after_action(simTime, obs)

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

        stacked_info = {}
        for name, values in self.info.items():
            # info to offset by 1 with 0 in front
            if name in ['reaction', 'torque']:
                values = [np.zeros(len(values[0]))] + values
            values = np.stack(values)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            stacked_info[name] = np.stack(values)
        data = {'X': X, 'U': self.u,
                'model error': self.model_error,
                'scaled model error': scaled_model_error,
                'mask': mask.reshape(-1, 1), 'predicted dynamics_class': self.pred_cls.reshape(-1, 1)}
        data.update(stacked_info)

        return data

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

        # plot of other info
        self._start_plot_other_info()

        for i in range(state_dim):
            self.axes[i].set_ylabel(axis_name[i])
        for i in range(ctrl_dim):
            self.au[i].set_ylabel('$u_{}$'.format(i))

        self.fig.tight_layout()
        self.fu.tight_layout()

        plt.ion()
        plt.show()

    def _start_plot_other_info(self):
        pass

    def _do_plot_other_info(self):
        pass

    def _plot_data(self):
        if self.fig is None:
            self.start_plot_runs()
            plt.pause(0.0001)

        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i], label='true')
        self.axes[0].legend()

        self._do_plot_other_info()

        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        plt.pause(0.0001)

    def _reset_sim(self):
        return self.env.reset()
