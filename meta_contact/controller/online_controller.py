import abc
import logging
from enum import Enum

import numpy as np
import torch

from arm_pytorch_utilities import math_utils, linalg, tensor_utils
from meta_contact.dynamics import online_model, hybrid_model
from meta_contact.controller import controller, gating_function
from meta_contact import cost

logger = logging.getLogger(__name__)


class OnlineController(controller.MPC):
    """
    Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf

    External API is in numpy ndarrays, but internally keeps tensors, and interacts with any models using tensors
    """

    def update_prior(self, prior):
        self.dynamics.prior = prior

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-1], self.u_history[-1], x)

        u = self._compute_action(x)

        return u

    @abc.abstractmethod
    def _compute_action(self, x):
        """
        Compute nu-dimensional action from current policy
        """


def noop_constrain(state):
    return state


class OnlineMPC(OnlineController):
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
        super().__init__(*args, **kwargs)
        assert isinstance(self.dynamics, hybrid_model.HybridDynamicsModel)

    def reset(self):
        super(OnlineMPC, self).reset()
        self.dynamics_class_prediction = {}
        self.dynamics_class_history = []

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

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class OnlineMPPI(OnlineMPC, controller.MPPI_MPC):
    def __init__(self, *args, abs_unrecognized_threshold=1, rel_unrecognized_threshold=5, compare_in_latent_space=False,
                 **kwargs):
        super(OnlineMPPI, self).__init__(*args, **kwargs)
        self.recovery_traj_seeder = None
        self.abs_unrecognized_threshold = abs_unrecognized_threshold
        self.rel_unrecognized_threshold = rel_unrecognized_threshold
        self.autonomous_recovery_mode = False
        self.leave_recovery_num_turns = 3
        self.recovery_cost = None
        self.compare_in_latent_space = compare_in_latent_space

    def create_recovery_traj_seeder(self, *args, **kwargs):
        self.recovery_traj_seeder = RecoveryTrajectorySeeder(self, *args, **kwargs)

    def _mpc_command(self, obs):
        return OnlineMPC._mpc_command(self, obs)

    def _recovery_running_cost(self, state, action):
        return self.recovery_cost(state, action)

    def _recovery_terminal_cost(self, state, action):
        # extract the last state; assume if given 3 dimensions then it's (B x T x nx) or (M x B x T x nx)
        if len(state.shape) is 3:
            state = state[:, -1, :]
        elif len(state.shape) is 4:
            state = state[:, :, -1, :]
        state_loss = self.terminal_cost_multiplier * self.recovery_cost(state, action, terminal=True)
        return state_loss

    def _compute_action(self, x):
        assert self.recovery_traj_seeder is not None
        # use only state for dynamics_class selection; this way we can get dynamics_class before calculating action
        a = torch.zeros((1, self.nu), device=self.d, dtype=x.dtype)
        self.dynamics_class = self.gating.sample_class(x.view(1, -1), a).item()

        if self.diff_predicted is not None:
            logger.debug("abs err %f rel err %f full %s %s", self.diff_predicted.norm(), self.diff_relative.norm(),
                         self.diff_predicted.cpu().numpy(), self.diff_relative.cpu().numpy())
        # it's unrecognized if we don't recognize it as any local model (gating function thinks its the nominal model)
        # but we still have high model error
        if self.diff_predicted is not None and self.dynamics_class == gating_function.DynamicsClass.NOMINAL and (
                self.diff_predicted.norm() > self.abs_unrecognized_threshold and self.diff_relative.norm() > self.rel_unrecognized_threshold):
            self.dynamics_class = gating_function.DynamicsClass.UNRECOGNIZED

            # TODO if we're sure that we've entered an unrecognized class (consistent dynamics not working)
            if not self.autonomous_recovery_mode and self.dynamics_class_history[-1] != self.dynamics_class:
                logger.debug("Entering autonomous recovery mode")
                self.autonomous_recovery_mode = True
                # change mpc cost
                # TODO parameterize this
                goal_set = self.x_history[-10:-3]
                self.recovery_cost = cost.CostQRGoalSet(goal_set, self.Q, self.R, self.compare_to_goal, self.ds,
                                                        compare_in_latent_space=self.compare_in_latent_space)
                self.mpc.running_cost = self._recovery_running_cost
                self.mpc.terminal_state_cost = self._recovery_terminal_cost
        else:
            if self.autonomous_recovery_mode and torch.all(torch.tensor(self.dynamics_class_history[
                                                                        -self.leave_recovery_num_turns:] != gating_function.DynamicsClass.UNRECOGNIZED)):
                logger.debug("Leaving autonomous recovery mode")
                self.autonomous_recovery_mode = False
                # restore cost functions
                self.mpc.running_cost = self._running_cost
                self.mpc.terminal_state_cost = self._terminal_cost
                # TODO if we're sure that we've left an unrecognized class, save as recovery
                # TODO filter out moves

            self.recovery_traj_seeder.update_nominal_trajectory(self.dynamics_class, x)

        self.dynamics_class_history.append(self.dynamics_class)

        u = self.mpc.command(x)
        return u


class NominalTrajFrom(Enum):
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
        for dyn_cls, ds_local in enumerate(dss):
            if dyn_cls == gating_function.DynamicsClass.NOMINAL:
                continue

            self.train_i[dyn_cls] = 14
            max_rollout_steps = 10
            XU, Y, info = ds_local.training_set(original=True)
            X_train, self.U_train[dyn_cls] = torch.split(XU, self.ds_nominal.original_config().nx, dim=1)
            assert self.train_i[dyn_cls] < len(X_train)

            if nom_traj_from is NominalTrajFrom.ROLLOUT_FROM_RECOVERY_STATES:
                for ii in range(max(0, self.train_i[dyn_cls] - max_rollout_steps), self.train_i[dyn_cls]):
                    ctrl.command(X_train[ii].cpu().numpy())
                self.U_train[dyn_cls] = ctrl.mpc.U.clone()
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
                    train_i = (self.Z_train[dyn_cls] - z).norm(dim=1).argmin()

                ctrl_rollout_steps = min(len(self.U_train[dyn_cls]) - train_i, self.ctrl.mpc.T)
                U[1:1 + ctrl_rollout_steps] = self.U_train[dyn_cls][train_i:train_i + ctrl_rollout_steps]

        if adjusted_trajectory:
            self.ctrl.mpc.U = U
        self.last_class = dyn_cls
        return adjusted_trajectory
