import abc
import logging

from arm_pytorch_utilities import math_utils
from meta_contact import online_model
from meta_contact.controller import controller

LOGGER = logging.getLogger(__name__)


class OnlineController(controller.MPC):
    """
    Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf

    External API is in numpy ndarrays, but internally keeps tensors, and interacts with any models using tensors
    """

    def __init__(self, online_dynamics: online_model.OnlineDynamicsModel, config, **kwargs):
        super().__init__(online_dynamics, config, **kwargs)
        self.prev_u = None
        self.u_history = []

    def reset(self):
        self.prev_u = None
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

        self.dynamics.evaluate_error(self.prev_x, self.prev_u, x, u)
        # if self.prevu is not None:  # smooth
        #    u = 0.5*u+0.5*self.prevu
        self.prev_u = u
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

    def __init__(self, *args, constrain_state=noop_constrain, **kwargs):
        self.constrain_state = constrain_state
        self.mpc = None
        super().__init__(*args, **kwargs)

    def _apply_dynamics(self, state, u):
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)

        # TODO the MPC method doesn't give dynamics px and pu (different from our prevx and prevu)
        # verified against non-batch calculations
        next_state = self.dynamics.predict(None, None, state, u)

        next_state = self.constrain_state(next_state)
        return next_state

    def _compute_action(self, x):
        return self.mpc.command(x)


class OnlineMPPI(OnlineMPC, controller.MPPI):
    def _command(self, obs):
        return OnlineMPC._command(self, obs)
