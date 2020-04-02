import abc
import logging
import numpy as np
import torch

from arm_pytorch_utilities import math_utils, linalg
from meta_contact.dynamics import online_model
from meta_contact.controller import controller, mode_selector

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
                 mode_select: mode_selector.ModeSelector = mode_selector.AlwaysSelectNominal(),
                 **kwargs):
        self.constrain_state = constrain_state
        self.mpc = None
        self.mode_select = mode_select
        super().__init__(*args, **kwargs)

    def _apply_dynamics(self, state, u, t=0):
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)

        dynamics_mode = self.mode_select.sample_mode(state, u)
        next_state = torch.zeros_like(state)
        # TODO we should generalize to more than 2 modes
        nominal_mode = dynamics_mode == 0
        local_mode = dynamics_mode == 1
        if torch.any(nominal_mode):
            next_state[nominal_mode] = self.dynamics.prior.dyn_net.predict(
                torch.cat((state[nominal_mode], u[nominal_mode]), dim=1))
        if torch.any(local_mode):
            next_state[local_mode] = self.dynamics.predict(None, None, state[local_mode], u[local_mode])

        next_state = self._adjust_next_state(next_state, u, t)
        next_state = self.constrain_state(next_state)

        return next_state

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class OnlineMPPI(OnlineMPC, controller.MPPI):
    def _mpc_command(self, obs):
        return OnlineMPC._mpc_command(self, obs)

    def _mpc_opts(self):
        # TODO move variance usage in biasing control sampling rather than as a loss function
        opts = super()._mpc_opts()
        # use variance cost if possible (when not sampling)
        if isinstance(self.dynamics, online_model.OnlineGPMixing) and self.dynamics.sample_dynamics is False:
            q = 10000
            self.Q_variance = torch.diag(torch.tensor([q, q, q, q, 0, 0], device=self.d, dtype=self.dtype))
            opts['dynamics_variance'] = self._dynamics_variance
            opts['running_cost_variance'] = self._running_cost_variance
        return opts

    def _dynamics_variance(self, next_state):
        import gpytorch
        if self.dynamics.last_prediction is not None:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return self.dynamics.last_prediction.variance
        return None

    def _running_cost_variance(self, variance):
        if variance is not None:
            c = linalg.batch_quadratic_product(variance, self.Q_variance)
            return -c
        return 0
