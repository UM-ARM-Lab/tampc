from arm_pytorch_utilities import rand
from stucco.env import cost as control_cost
from tampc.controller import controller, online_controller

from stucco.env.pybullet_sim import PybulletSim


class TAMPCPybulletSim(PybulletSim):
    def visualize_before_action(self, simTime, obs):
        super(TAMPCPybulletSim, self).visualize_before_action(simTime, obs)

        if isinstance(self.ctrl, online_controller.TAMPC):
            if self.ctrl.projected_x is not None:
                self.env.draw_user_text(
                    "x {} projected to {}".format(obs.round(decimals=2),
                                                  self.ctrl.projected_x.cpu().numpy().round(decimals=2)),
                    xy=(-1., -0., -1))

            self.pred_cls[simTime] = self.ctrl.dynamics_class
            self.env.draw_user_text("dyn cls {}".format(self.ctrl.dynamics_class), location_index=2,
                                    xy=(0.5, 0.6, -1))

            if self.ctrl.trap_set and self.ctrl.trap_cost is not None:
                self.env.visualize_trap_set(self.ctrl.trap_set)

            if self.ctrl.autonomous_recovery is online_controller.AutonomousRecovery.MAB:
                mode_text = "recovery" if self.ctrl.autonomous_recovery_mode else (
                    "local" if self.ctrl.using_local_model_for_nonnominal_dynamics else "")
                self.env.draw_user_text(mode_text, location_index=3, xy=(0.5, 0.5, -1))
                if self.ctrl.recovery_cost and isinstance(self.ctrl.recovery_cost,
                                                          (control_cost.GoalSetCost, control_cost.CostQRSet)):
                    # plot goal set
                    self.env.visualize_goal_set(self.ctrl.recovery_cost.goal_set)

        with rand.SavedRNG():
            if self.visualize_action_sample and isinstance(self.ctrl, controller.MPPI_MPC):
                self._plot_action_sample(self.ctrl.mpc.perturbed_action)
            if self.visualize_rollouts:
                rollouts = self.ctrl.get_rollouts(obs)
                self.env.visualize_rollouts(rollouts)

    def visualize_after_action(self, simTime, obs):
        if isinstance(self.ctrl, controller.ControllerWithModelPrediction):
            self.pred_traj[simTime + 1, :] = self.ctrl.predicted_next_state
            # model error from the previous prediction step (can only evaluate it at the current step)
            self.model_error[simTime, :] = self.ctrl.prediction_error(obs)
            if self.visualize_prediction_error:
                self.env.visualize_prediction_error(self.ctrl.predicted_next_state.reshape(-1))
