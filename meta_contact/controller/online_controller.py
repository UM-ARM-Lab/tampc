import abc
import logging
import enum

import numpy as np
import torch

from arm_pytorch_utilities import math_utils, linalg, tensor_utils
from meta_contact.dynamics import online_model, hybrid_model
from meta_contact.controller import controller, gating_function
from meta_contact import cost

logger = logging.getLogger(__name__)


def noop_constrain(state):
    return state


class OnlineMPC(controller.MPC):
    """
    Online controller with a pytorch based MPC method (CEM, MPPI)
    """

    def __init__(self, ds, *args, constrain_state=noop_constrain,
                 gating: gating_function.GatingFunction = gating_function.AlwaysSelectNominal(),
                 **kwargs):
        self.ds = ds
        self.constrain_state = constrain_state
        self.mpc = None
        self.gating = gating
        self.dynamics_class = 0
        self.dynamics_class_prediction = {}
        self.dynamics_class_history = []
        self.mpc_cost_history = []
        super().__init__(*args, **kwargs)
        assert isinstance(self.dynamics, hybrid_model.HybridDynamicsModel)

    def update_prior(self, prior):
        self.dynamics.prior = prior

    def reset(self):
        super(OnlineMPC, self).reset()
        self.dynamics_class_prediction = {}
        self.dynamics_class_history = []
        self.mpc_cost_history = []

    def predict_next_state(self, state, control):
        # we'll temporarily ensure usage of the original nominal model for predicting the next state
        current_model = self.dynamics.nominal_model
        self.dynamics.nominal_model = self.dynamics._original_nominal_model

        next_state = self.dynamics(state.view(1, -1), control.view(1, -1),
                                   torch.tensor(gating_function.DynamicsClass.NOMINAL).view(-1)).cpu().numpy()

        self.dynamics.nominal_model = current_model
        return next_state

    @tensor_utils.ensure_2d_input
    def _apply_dynamics(self, state, u, t=0):
        # TODO this is a hack for handling when dynamics blows up
        bad_states = state.norm(dim=1) > 1e5
        x = state
        x[bad_states] = 0
        u[bad_states] = 0

        cls = self.gating.sample_class(x, u)
        cls[bad_states] = -1

        self.dynamics_class_prediction[t] = cls

        # hybrid dynamics
        next_state = self.dynamics(x, u, cls)

        next_state = self._adjust_next_state(next_state, u, t)
        next_state = self.constrain_state(next_state)
        next_state[bad_states] = state[bad_states]

        return next_state

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-1], self.u_history[-1], x)

        self.mpc_cost_history.append(self.mpc.running_cost(x.view(1, -1), None))
        u = self._compute_action(x)

        return u

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class AutonomousRecovery(enum.IntEnum):
    NONE = 0
    RANDOM = 1
    RETURN_STATE = 2
    RETURN_LATENT = 3


class OnlineMPPI(OnlineMPC, controller.MPPI_MPC):
    def __init__(self, *args, abs_unrecognized_threshold=2, rel_unrecognized_threshold=5,
                 assume_all_nonnominal_dynamics_are_traps=True, nonnominal_dynamics_penalty_tolerance = 0.5,
                 autonomous_recovery=AutonomousRecovery.RETURN_STATE, reuse_escape_as_demonstration=True, **kwargs):
        super(OnlineMPPI, self).__init__(*args, **kwargs)
        self.recovery_traj_seeder: RecoveryTrajectorySeeder = None
        self.abs_unrecognized_threshold = abs_unrecognized_threshold
        self.rel_unrecognized_threshold = rel_unrecognized_threshold

        self.assume_all_nonnominal_dynamics_are_traps = assume_all_nonnominal_dynamics_are_traps

        self.using_local_model_for_nonnominal_dynamics = False
        self.nonnominal_dynamics_start_index = -1
        self.nonnominal_dynamics_trend_len = 4
        # we have to be decreasing cost at this much compared to before nonnominal dynamics to not be in a trap
        self.nonnominal_dynamics_penalty_tolerance = nonnominal_dynamics_penalty_tolerance

        self.autonomous_recovery_mode = False
        self.autonomous_recovery_start_index = -1
        self.autonomous_recovery_end_index = -1
        self.leave_recovery_num_turns = 3
        self.recovery_cost = None
        self.autonomous_recovery = autonomous_recovery
        self.original_horizon = self.mpc.T
        self.reuse_escape_as_demonstration = reuse_escape_as_demonstration

    def create_recovery_traj_seeder(self, *args, **kwargs):
        self.recovery_traj_seeder = RecoveryTrajectorySeeder(self, *args, **kwargs)

    def _mpc_command(self, obs):
        return OnlineMPC._mpc_command(self, obs)

    def _recovery_running_cost(self, state, action):
        return self.recovery_cost(state, action)

    def _in_non_nominal_dynamics(self):
        return self.diff_predicted is not None and \
               self.dynamics_class == gating_function.DynamicsClass.NOMINAL and \
               self.diff_predicted.norm() > self.abs_unrecognized_threshold and \
               self.diff_relative.norm() > self.rel_unrecognized_threshold and \
               len(self.u_history) > 1 and \
               self.u_history[-1][1] > 0

    def _entering_trap(self):
        # already inside trap
        if self.autonomous_recovery_mode:
            return False

        # not in non-nominal dynamics assume not a trap
        if not self.using_local_model_for_nonnominal_dynamics:
            return False

        # heuristic for determining if this a trap and should we enter autonomous recovery mode
        # TODO kind of hacky but our model predicts poorly when action has 0 magnitude
        if self.autonomous_recovery is not AutonomousRecovery.NONE and \
                len(self.x_history) > 3 and \
                self.u_history[-1][1] > 0:

            if self.assume_all_nonnominal_dynamics_are_traps:
                return True

            # cooldown on entering and leaving traps
            cur_index = len(self.x_history)
            if cur_index - self.autonomous_recovery_end_index < self.leave_recovery_num_turns:
                return False

            # check cost history compared to before we entered non-nominal dyanmics and after we've entered
            # don't include the first state since the reaction forces are not initialized correctly
            before_trend = torch.cat(self.orig_cost_history[
                                     max(1, self.nonnominal_dynamics_start_index - self.nonnominal_dynamics_trend_len):
                                     self.nonnominal_dynamics_start_index])
            current_trend = torch.cat(self.orig_cost_history[-self.nonnominal_dynamics_trend_len:])
            # should be negative
            before_progress_rate = (before_trend[1:] - before_trend[:-1]).mean()
            current_progress_rate = (current_trend[1:] - current_trend[:-1]).mean()
            is_trap = before_progress_rate * self.nonnominal_dynamics_penalty_tolerance < current_progress_rate
            logger.debug("before progress rate %f current progress rate %f trap? %d", before_progress_rate.item(),
                         current_progress_rate.item(), is_trap)
            return is_trap
        return False

    def _left_trap(self):
        # not in a trap to begin with
        if not self.autonomous_recovery_mode:
            return False

        # can leave if we've left non-nominal dynamics for a while
        consecutive_recognized_dynamics_class = 0
        for i in range(-1, -len(self.u_history), -1):
            if self.dynamics_class_history[i] == gating_function.DynamicsClass.UNRECOGNIZED:
                break
            if self.u_history[i][1] > 0:
                consecutive_recognized_dynamics_class += 1
        if consecutive_recognized_dynamics_class >= self.leave_recovery_num_turns:
            return True

        cur_index = len(self.mpc_cost_history) - 1
        if cur_index - self.autonomous_recovery_start_index < self.nonnominal_dynamics_trend_len:
            return False

        # can also leave if we are as close as we can get to previous states
        before_trend = torch.cat(self.mpc_cost_history[self.autonomous_recovery_start_index:
                                                       self.autonomous_recovery_start_index + self.nonnominal_dynamics_trend_len])
        current_trend = torch.cat(self.mpc_cost_history[max(self.autonomous_recovery_start_index,
                                                            cur_index - self.nonnominal_dynamics_trend_len):])

        before_progress_rate = (before_trend[1:] - before_trend[:-1]).mean()
        current_progress_rate = (current_trend[1:] - current_trend[:-1]).mean()
        left_trap = before_progress_rate * 0.1 < current_progress_rate
        logger.debug("before recovery rate %f current recovery rate %f left trap? %d", before_progress_rate.item(),
                     current_progress_rate.item(), left_trap)
        return left_trap

    def _left_local_model(self):
        # not using local model to begin with
        if not self.using_local_model_for_nonnominal_dynamics:
            return False
        consecutive_recognized_dynamics_class = 0
        for i in range(-1, -len(self.u_history), -1):
            if self.dynamics_class_history[i] == gating_function.DynamicsClass.UNRECOGNIZED:
                break
            if self.u_history[i][1] > 0:
                consecutive_recognized_dynamics_class += 1
        return consecutive_recognized_dynamics_class >= self.leave_recovery_num_turns

    def _start_local_model(self, x):
        logger.debug("Entering non nominal dynamics")
        logger.debug(self.diff_predicted)
        logger.debug(self.diff_relative)

        self.using_local_model_for_nonnominal_dynamics = True
        # does not include the current observation
        self.nonnominal_dynamics_start_index = len(self.x_history) + 1

        self.dynamics.use_temp_local_nominal_model()
        # update the local model with the last transition for entering the mode
        self.dynamics.update(self.x_history[-1], self.u_history[-1], x)

    def _start_recovery_mode(self):
        logger.debug("Entering autonomous recovery mode")
        self.autonomous_recovery_mode = True
        self.autonomous_recovery_start_index = len(self.x_history) + 1

        # different strategies for recovery mode
        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.RETURN_LATENT]:
            # change mpc cost
            # TODO parameterize how far back in history to return to
            goal_set = torch.stack(
                self.x_history[max(0, self.nonnominal_dynamics_start_index - 10):self.nonnominal_dynamics_start_index])
            # TODO remove this domain knowledge of what dimensions are import for recovering
            Q = self.Q.clone()
            Q[0, 0] = Q[1, 1] = 1
            Q[2, 2] = 1
            Q[3, 3] = Q[4, 4] = 0
            self.recovery_cost = cost.CostQRGoalSet(goal_set, Q, self.R, self.compare_to_goal, self.ds,
                                                    compare_in_latent_space=self.autonomous_recovery is AutonomousRecovery.RETURN_LATENT)
            self.mpc.running_cost = self._recovery_running_cost
            self.mpc.terminal_state_cost = None
            self.mpc.change_horizon(10)

    def _end_recovery_mode(self):
        logger.debug("Leaving autonomous recovery mode")
        logger.debug(torch.tensor(self.dynamics_class_history[-self.leave_recovery_num_turns:]))
        self.autonomous_recovery_mode = False
        self.autonomous_recovery_end_index = len(self.x_history) + 1

        # if we're sure that we've left an unrecognized class, save as recovery
        if self.reuse_escape_as_demonstration:
            # TODO filter out moves? / weight later points more?
            x_recovery = []
            u_recovery = []
            for i in range(self.autonomous_recovery_start_index, len(self.x_history)):
                if self.u_history[i][1] > 0:
                    x_recovery.append(self.x_history[i])
                    u_recovery.append(self.u_history[i])
            x_recovery = torch.stack(x_recovery)
            u_recovery = torch.stack(u_recovery)
            logger.info("Using data from index %d with len %d for local model",
                        self.autonomous_recovery_start_index, x_recovery.shape[0])
            self.dynamics.create_local_model(x_recovery, u_recovery)
            self.gating = self.dynamics.get_gating()
            self.recovery_traj_seeder.update_data(self.dynamics.dss)

        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.RETURN_LATENT]:
            # restore cost functions
            self.mpc.running_cost = self._running_cost
            self.mpc.terminal_state_cost = self._terminal_cost
            self.mpc.change_horizon(self.original_horizon)

    def _end_local_model(self):
        logger.debug("Leaving local model")
        self.dynamics.use_normal_nominal_model()
        self.using_local_model_for_nonnominal_dynamics = False

    def _compute_action(self, x):
        assert self.recovery_traj_seeder is not None
        # use only state for dynamics_class selection; this way we can get dynamics_class before calculating action
        a = torch.zeros((1, self.nu), device=self.d, dtype=x.dtype)
        self.dynamics_class = self.gating.sample_class(x.view(1, -1), a).item()

        # in non-nominal dynamics
        if self._in_non_nominal_dynamics():
            self.dynamics_class = gating_function.DynamicsClass.UNRECOGNIZED

            if not self.using_local_model_for_nonnominal_dynamics:
                self._start_local_model(x)
        else:
            # TODO check correctness; only update nominal trajectory if we're not in autonomous recovery mode
            if not self.autonomous_recovery_mode:
                self.recovery_traj_seeder.update_nominal_trajectory(self.dynamics_class, x)

        self.dynamics_class_history.append(self.dynamics_class)

        if self._entering_trap():
            self._start_recovery_mode()

        if self._left_trap():
            self._end_recovery_mode()

        if self._left_local_model():
            self._end_local_model()

        if self.autonomous_recovery_mode and self.autonomous_recovery is AutonomousRecovery.RANDOM:
            u = torch.rand(self.nu, device=self.d).cuda() * (self.u_max - self.u_min) + self.u_min
        else:
            u = self.mpc.command(x)

        return u


class NominalTrajFrom(enum.IntEnum):
    RANDOM = 0
    ROLLOUT_FROM_RECOVERY_STATES = 1
    RECOVERY_ACTIONS = 2
    NO_ADJUSTMENT = 3


class RecoveryTrajectorySeeder:
    def __init__(self, ctrl: OnlineMPPI, dss, fixed_recovery_nominal_traj=True, lookup_traj_start=True,
                 nom_traj_from=NominalTrajFrom.RECOVERY_ACTIONS):
        """
        :param ctrl:
        :param dss:
        :param fixed_recovery_nominal_traj: if false, only set nominal traj when entering dynamics_class, otherwise set every step
        """
        self.ctrl = ctrl
        self.fixed_recovery_nominal_traj = fixed_recovery_nominal_traj
        self.last_class = 0
        self.nom_traj_from = nom_traj_from
        self.lookup_traj_start = lookup_traj_start
        self.ds_nominal = dss[0]

        # local trajectories
        self.train_i = {}
        self.Z_train = {}
        self.U_train = {}
        self.update_data(dss)

    def update_data(self, dss):
        self.train_i = {}
        self.Z_train = {}
        self.U_train = {}
        for dyn_cls, ds_local in enumerate(dss):
            if dyn_cls == gating_function.DynamicsClass.NOMINAL:
                continue

            max_rollout_steps = 10
            XU, Y, info = ds_local.training_set(original=True)
            X_train, self.U_train[dyn_cls] = torch.split(XU, self.ds_nominal.original_config().nx, dim=1)
            self.train_i[dyn_cls] = len(X_train) - 1

            if self.nom_traj_from is NominalTrajFrom.ROLLOUT_FROM_RECOVERY_STATES:
                for ii in range(max(0, self.train_i[dyn_cls] - max_rollout_steps), self.train_i[dyn_cls]):
                    self.ctrl.command(X_train[ii].cpu().numpy())
                self.U_train[dyn_cls] = self.ctrl.mpc.U.clone()
            U_zero = torch.zeros_like(self.U_train[dyn_cls])
            self.Z_train[dyn_cls] = self.ds_nominal.preprocessor.transform_x(torch.cat((X_train, U_zero), dim=1))

    def update_nominal_trajectory(self, dyn_cls, state):
        if dyn_cls == gating_function.DynamicsClass.NOMINAL or self.nom_traj_from is NominalTrajFrom.NO_ADJUSTMENT:
            return False
        adjusted_trajectory = False
        # start with random noise
        U = self.ctrl.mpc.noise_dist.sample((self.ctrl.mpc.T,))
        if self.nom_traj_from is NominalTrajFrom.RANDOM:
            adjusted_trajectory = True
        else:
            if dyn_cls not in self.U_train:
                raise RuntimeError(
                    "Unrecgonized dynamics class {} (known {})".format(dyn_cls, list(self.U_train.keys())))

            # if we're not using the fixed recovery nom, then we set it if we're entering from another dynamics_class
            if self.fixed_recovery_nominal_traj or self.last_class != dyn_cls:
                adjusted_trajectory = True

                train_i = self.train_i[dyn_cls]
                # option to select where to start in training automatically from data
                if self.lookup_traj_start:
                    xu = torch.cat(
                        (state.view(1, -1), torch.zeros((1, self.ctrl.nu), device=state.device, dtype=state.dtype)),
                        dim=1)
                    z = self.ds_nominal.preprocessor.transform_x(xu)
                    dists = (self.Z_train[dyn_cls] - z).norm(dim=1)
                    # TODO prioritize points closer to the escape?
                    train_i = dists.argmin()

                ctrl_rollout_steps = min(len(self.U_train[dyn_cls]) - train_i, self.ctrl.mpc.T - 1)
                U[1:1 + ctrl_rollout_steps] = self.U_train[dyn_cls][train_i:train_i + ctrl_rollout_steps]

        if adjusted_trajectory:
            self.ctrl.mpc.U = U
        self.last_class = dyn_cls
        return adjusted_trajectory
