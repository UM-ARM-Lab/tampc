import time
import torch

import numpy as np
from arm_pytorch_utilities import simulation, math_utils, rand
from matplotlib import pyplot as plt
from tampc import cfg, cost as control_cost
from tampc.controller import controller, online_controller
from tampc.env.pybullet_env import PybulletEnv, logger


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
        self.reaction_force = np.zeros((self.num_frames, self.reaction_dim))
        self.wall_contact = np.zeros((self.num_frames,))
        self.contact_id = np.ones((self.num_frames,), dtype=np.intc) * -1
        self.model_error = np.zeros_like(self.traj)
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        self.pred_cls = np.zeros_like(self.wall_contact)
        self.object_poses = {}
        self.object_distances = {}
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.reaction_force, self.contact_id, self.wall_contact, self.model_error, self.time, self.pred_cls = (
            data[:frame] for data
            in
            (self.traj, self.u,
             self.reaction_force,
             self.contact_id,
             self.wall_contact,
             self.model_error,
             self.time, self.pred_cls))

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _has_recovery_policy(self):
        return isinstance(self.ctrl, online_controller.TAMPC)

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        info = None

        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            self.env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

            start = time.perf_counter()

            action = self.ctrl.command(obs, info)

            contact_set = getattr(self.ctrl, 'contact_set', None)
            if contact_set is not None:
                self.env.visualize_contact_set(contact_set)

            # visualizations before taking action
            if self._predicts_dynamics_cls():
                if self.ctrl.projected_x is not None:
                    self.env.draw_user_text(
                        "x {} projected to {}".format(obs.round(decimals=2),
                                                      self.ctrl.projected_x.cpu().numpy().round(decimals=2)),
                        xy=(-1., -0., -1))

                self.pred_cls[simTime] = self.ctrl.dynamics_class
                self.env.draw_user_text("dyn cls {}".format(self.ctrl.dynamics_class), location_index=2,
                                        xy=(0.5, 0.6, -1))

                # if self.ctrl.dynamics_class != 0:
                #     self.ctrl.dynamics.nominal_model.plot_dynamics_at_state(
                #         torch.tensor(obs, dtype=self.ctrl.dtype, device=self.ctrl.d),
                #         'update at t={} ({})'.format(simTime, obs))

                if self.ctrl.trap_set and self.ctrl.trap_cost is not None:
                    self.env.visualize_trap_set(self.ctrl.trap_set)

                if self._has_recovery_policy() and self.ctrl.autonomous_recovery is online_controller.AutonomousRecovery.MAB:
                    mode_text = "recovery" if self.ctrl.autonomous_recovery_mode else (
                        "local" if self.ctrl.using_local_model_for_nonnominal_dynamics else "")
                    self.env.draw_user_text(mode_text, location_index=3, xy=(0.5, 0.5, -1))
                    if self.ctrl.recovery_cost and isinstance(self.ctrl.recovery_cost,
                                                              (control_cost.GoalSetCost, control_cost.CostQRSet)):
                        # plot goal set
                        self.env.visualize_goal_set(self.ctrl.recovery_cost.goal_set)

                    # for i in range(self.ctrl.num_costs):
                    #     self.env.draw_user_text(
                    #         "a{} {:.2f} ({:.2f})".format(i, self.ctrl.mab._mean[i], self.ctrl.mab._cov[i, i]),
                    #         4 + i)
                    # if self.ctrl.last_arm_pulled is not None:
                    #     text = ["a"]
                    #     for value in self.ctrl.recovery_cost_weight():
                    #         text.append("{:.2f}".format(value))
                    #     self.env.draw_user_text(" ".join(text), 4 + self.ctrl.num_costs,
                    #                             left_offset=1 - (self.ctrl.num_costs - 3) * 0.1)

            with rand.SavedRNG():
                if self.visualize_action_sample and isinstance(self.ctrl, controller.MPPI_MPC):
                    self._plot_action_sample(self.ctrl.mpc.perturbed_action)
                if self.visualize_rollouts:
                    rollouts = self.ctrl.get_rollouts(obs)
                    self.env.visualize_rollouts(rollouts)

            # with rand.SavedRNG():
            #     nom_actions = self.ctrl.mpc.U
            #     can_actions = torch.zeros_like(nom_actions)
            #     cur_state = torch.tensor(obs, dtype=nom_actions.dtype, device=nom_actions.device)
            #     scale = self.env.MAX_PUSH_DIST
            #     for t in range(nom_actions.shape[0]):
            #         d = self.ctrl.goal - cur_state
            #         straight_action = d[:2] / scale
            #         straight_action = math_utils.clip(straight_action, self.ctrl.u_min, self.ctrl.u_max)
            #         cur_state[:2] += straight_action * scale
            #         can_actions[t] = straight_action
            #     actions = torch.stack((nom_actions, can_actions))
            #     cost_total, states, _, center_points = self.ctrl.mpc._compute_rollout_costs(actions)
            #     colors = ['copper', 'cool', 'spring']
            #     visualized = 0
            #     # if self.visualize_rollouts:
            #     #     self.env.visualize_rollouts(states[0, 1].cpu().numpy(), state_cmap='summer')
            #     if center_points[0] is not None:
            #         # only consider the first sample (m = 0)
            #         center_points = [pt[:, 0] for pt in center_points]
            #         center_points = torch.stack(center_points)
            #         num_objs = center_points.shape[1]
            #         for j in range(num_objs):
            #             visualized += 1
            #             rollout = center_points[:, j]
            #             c = colors[j % len(colors)]
            #             self.env.visualize_rollouts(rollout.cpu().numpy(), state_cmap=c)
            #     if self.visualize_rollouts:
            #         for j in range(visualized, len(colors)):
            #             self.env.visualize_rollouts([], state_cmap=colors[j % len(colors)])
            #     self.env.draw_user_text(
            #         "straight cost {:.2f} sampled cost {:.2f}".format(cost_total[1], cost_total[0]),
            #         location_index=5, xy=(-0.5, 0.3, -1))

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
            # reaction force felt as we apply this action, as observed at the start of the next time step
            self.reaction_force[simTime + 1, :] = info['reaction']
            self.wall_contact[simTime + 1] = info['wall_contact']
            self.contact_id[simTime] = info['contact_id']
            object_poses = info.get('object_poses', None)
            if object_poses is not None:
                for obj_id, pose in object_poses.items():
                    if obj_id not in self.object_poses:
                        self.object_poses[obj_id] = []
                    self.object_poses[obj_id].append(pose)
            object_distances = info.get('object_distances', None)
            if object_distances is not None:
                for obj_id, distance in object_distances.items():
                    if obj_id not in self.object_distances:
                        self.object_distances[obj_id] = []
                    self.object_distances[obj_id].append(distance)

            if self._predicts_state():
                self.pred_traj[simTime + 1, :] = self.ctrl.predicted_next_state
                # model error from the previous prediction step (can only evaluate it at the current step)
                self.model_error[simTime, :] = self.ctrl.prediction_error(obs)
                if self.visualize_prediction_error:
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

        data = {'X': X, 'U': self.u, 'reaction': self.reaction_force, 'model error': self.model_error,
                'scaled model error': scaled_model_error, 'wall contact': self.wall_contact.reshape(-1, 1),
                'contact_id': self.contact_id.reshape(-1, 1),
                'mask': mask.reshape(-1, 1), 'predicted dynamics_class': self.pred_cls.reshape(-1, 1)}

        for obj_id, pose_list in self.object_poses.items():
            poses = np.stack(pose_list)
            data[f"obj{obj_id}pose"] = poses

        for obj_id, distance_list in self.object_distances.items():
            distances = np.stack(distance_list)
            data[f"obj{obj_id}distance"] = distances

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
