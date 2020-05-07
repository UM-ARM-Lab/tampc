import abc
import logging
from enum import Enum

import numpy as np
import torch

from arm_pytorch_utilities import math_utils, linalg, tensor_utils
from meta_contact.dynamics import online_model
from meta_contact.controller import controller, gating_function

logger = logging.getLogger(__name__)


class OnlineController(controller.MPC):
    """
    Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf

    External API is in numpy ndarrays, but internally keeps tensors, and interacts with any models using tensors
    """

    def __init__(self, online_dynamics: online_model.OnlineDynamicsModel, config, **kwargs):
        super().__init__(online_dynamics, config, **kwargs)
        self.u_history = []

    def reset(self):
        self.u_history = []
        self.dynamics.reset()
        super(OnlineController, self).reset()

    def update_prior(self, prior):
        self.dynamics.prior = prior

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.prev_x, self.prev_u, x)

        u = self._compute_action(x)
        if self.u_max is not None:
            u = math_utils.clip(u, self.u_min, self.u_max)

        if isinstance(self.dynamics, online_model.OnlineLinearizeMixing):
            self.dynamics.evaluate_error(self.prev_x, self.prev_u, x, u)

        self.u_history.append(u)

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

    def __init__(self, *args, constrain_state=noop_constrain,
                 gating: gating_function.GatingFunction = gating_function.AlwaysSelectNominal(),
                 **kwargs):
        self.constrain_state = constrain_state
        self.mpc = None
        self.gating = gating
        self.dynamics_class = 0
        self.dynamics_class_history = {}
        super().__init__(*args, **kwargs)

    def reset(self):
        super(OnlineMPC, self).reset()
        self.dynamics_class_history = {}

    @tensor_utils.ensure_2d_input
    def _apply_dynamics(self, state, u, t=0):
        # TODO this is a hack for handling when dynamics blows up
        bad_states = state.norm(dim=1) > 1e5
        x = state
        x[bad_states] = 0
        u[bad_states] = 0

        cls = self.gating.sample_class(x, u)
        cls[bad_states] = -1

        self.dynamics_class_history[t] = cls
        next_state = torch.zeros_like(state)
        # TODO we should generalize to more than 2 modes
        nominal_cls = cls == 0
        local_cls = cls == 1
        if torch.any(nominal_cls):
            next_state[nominal_cls] = self.dynamics.prior.dyn_net.predict(
                torch.cat((state[nominal_cls], u[nominal_cls]), dim=1))
        if torch.any(local_cls):
            next_state[local_cls] = self.dynamics.predict(None, None, state[local_cls], u[local_cls])

        next_state = self._adjust_next_state(next_state, u, t)
        next_state = self.constrain_state(next_state)
        next_state[bad_states] = state[bad_states]

        return next_state

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class OnlineMPPI(OnlineMPC, controller.MPPI_MPC):
    def __init__(self, *args, **kwargs):
        super(OnlineMPPI, self).__init__(*args, **kwargs)
        self.recovery_traj_seeder = None

    def create_recovery_traj_seeder(self, *args, **kwargs):
        self.recovery_traj_seeder = RecoveryTrajectorySeeder(self, *args, **kwargs)

    def _mpc_command(self, obs):
        return OnlineMPC._mpc_command(self, obs)

    def _compute_action(self, x):
        assert self.recovery_traj_seeder is not None
        # use only state for dynamics_class selection; this way we can get dynamics_class before calculating action
        a = torch.zeros((1, self.nu), device=self.d, dtype=x.dtype)
        self.dynamics_class = self.gating.sample_class(x.view(1, -1), a).item()
        self.recovery_traj_seeder.update_nominal_trajectory(self.dynamics_class, x)
        # TODO change mpc cost if we're outside the nominal dynamics_class
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
