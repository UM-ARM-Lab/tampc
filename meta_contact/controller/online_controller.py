import abc
import logging
import numpy as np
import torch

from arm_pytorch_utilities import math_utils, linalg
from meta_contact import online_model
from meta_contact.controller import controller

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

    def _command(self, obs):
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

    def __init__(self, *args, constrain_state=noop_constrain, always_use_online_model=False, **kwargs):
        self.constrain_state = constrain_state
        self.mpc = None
        self.always_use_online_model = always_use_online_model
        super().__init__(*args, **kwargs)

    def _apply_dynamics(self, state, u, t=0):
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)

        # TODO select model in a smarter way; currently we have a in-contact local model and otherwise use nominal model
        # TODO the MPC method doesn't give dynamics px and pu (different from our prevx and prevu)
        use_local_model = self.always_use_online_model
        if self.context[0] is not None:
            r = np.linalg.norm(self.context[0]['reaction'])
            if r > 200:
                use_local_model = True
        if use_local_model:
            next_state = self.dynamics.predict(None, None, state, u)
        else:
            next_state = self.dynamics.prior.dyn_net.predict(torch.cat((state, u), dim=1))

        next_state = self._adjust_next_state(next_state, u, t)
        next_state = self.constrain_state(next_state)

        return next_state

    def _compute_action(self, x):
        return self.mpc.command(x)


class OnlineMPPI(OnlineMPC, controller.MPPI):
    def _command(self, obs):
        return OnlineMPC._command(self, obs)

    def _mpc_opts(self):
        opts = super()._mpc_opts()
        # use variance cost if possible (when not sampling)
        if isinstance(self.dynamics, online_model.OnlineGPMixing) and self.dynamics.sample_dynamics is False:
            self.Q_variance = torch.eye(self.nx, device=self.d, dtype=self.dtype) * 2.
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
            return c
        return 0
