import abc
import logging

import numpy as np
import torch
from arm_pytorch_utilities import trajectory, math_utils
from arm_pytorch_utilities.policy.lin_gauss import LinearGaussianPolicy
from meta_contact import cost
from meta_contact import online_model
from meta_contact.controller.controller import Controller
from pytorch_cem import cem
from pytorch_mppi import mppi

LOGGER = logging.getLogger(__name__)


class OnlineController(Controller):
    """Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf"""

    def __init__(self, online_dynamics: online_model.OnlineDynamicsModel, config,
                 compare_to_goal=np.subtract, u_min=None, u_max=None, Q=1, R=1):
        super().__init__(compare_to_goal)
        self.nx = config.nx
        self.nu = config.nu
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)

        self.dynamics = online_dynamics

        # Init objects
        if np.isscalar(Q):
            self.Q = np.eye(self.nx) * Q
        else:
            self.Q = Q
            assert self.Q.shape[0] == self.nx

        self.weight_u = np.ones(self.nu) * R
        self.R = np.diag(self.weight_u)
        self.cost = cost.CostQROnline(self.goal, self.Q, self.R, self.compare_to_goal)

        self.prevx = None
        self.prevu = None
        self.u_history = []

    def reset(self):
        self.prevx = None
        self.prevu = None
        self.u_history = []

    def set_goal(self, goal):
        super().set_goal(goal)
        self.cost.eetgt = goal

    def update_prior(self, prior):
        self.dynamics.prior = prior

    def command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.prevx, self.prevu, x)

        self.update_policy(t, x)

        u = self.compute_action(x)
        if self.u_max is not None:
            u = np.clip(u, self.u_min, self.u_max)

        self.dynamics.evaluate_error(self.prevx, self.prevu, x, u)
        # if self.prevu is not None:  # smooth
        #    u = 0.5*u+0.5*self.prevu
        self.prevx = x
        self.prevu = u
        self.u_history.append(u)

        return u

    @abc.abstractmethod
    def update_policy(self, t, x):
        """
        Update the controller state if necessary
        :param t: current time step
        :param x: current state
        :return: nothing
        """

    @abc.abstractmethod
    def compute_action(self, x):
        """
        Compute nu-dimensional action from current policy
        """


def noop_constrain(state):
    return state


class OnlineMPC(OnlineController):
    """
    Online controller with a pytorch based MPC method (CEM, MPPI)
    """

    def __init__(self, *args, constrain_state=noop_constrain, dtype=torch.double, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain_state = constrain_state
        self.mpc = None
        self.dtype = dtype

    def apply_dynamics(self, state, u):
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)

        # TODO the MPC method doesn't give dynamics px and pu (different from our prevx and prevu)
        # verified against non-batch calculations
        next_state = self.dynamics.predict(None, None, state, u)

        next_state = self.constrain_state(next_state)
        return next_state

    def get_running_cost(self, state, u):
        return torch.from_numpy(self.cost(state.numpy(), u.numpy()))

    def update_policy(self, t, x):
        pass

    def compute_action(self, x):
        return self.mpc.command(torch.from_numpy(x)).numpy()


class OnlineCEM(OnlineMPC):
    def __init__(self, *args, mpc_opts=None, **kwargs):
        super().__init__(*args, **kwargs)
        if mpc_opts is None:
            mpc_opts = {}
        self.mpc = cem.CEM(self.apply_dynamics, self.get_running_cost, self.nx, self.nu,
                           u_min=torch.tensor(self.u_min, dtype=self.dtype),
                           u_max=torch.tensor(self.u_max, dtype=self.dtype),
                           **mpc_opts)


class OnlineMPPI(OnlineMPC):
    def __init__(self, *args, mpc_opts=None, **kwargs):
        super().__init__(*args, **kwargs)
        if mpc_opts is None:
            mpc_opts = {}
        noise_sigma = mpc_opts.pop('noise_sigma', None)
        if noise_sigma is None:
            if torch.is_tensor(self.u_max):
                noise_sigma = torch.diag(self.u_max)
            else:
                noise_mult = self.u_max or 1
                noise_sigma = torch.eye(self.nu, dtype=self.dtype) * noise_mult
        self.mpc = mppi.MPPI(self.apply_dynamics, self.get_running_cost, self.nx, noise_sigma=noise_sigma, **mpc_opts)


class OnlineLQR(OnlineController):
    def __init__(self, *args, max_timestep=100, horizon=15, lqr_iter=1, u_noise=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = horizon
        self.maxT = max_timestep

        self.u_noise = u_noise

        # LQR options
        self.min_mu = 1e-6
        self.del0 = 2
        self.lqr_discount = 0.9
        self.lqr_iter = lqr_iter

        self.policy = None

    def reset(self):
        super(OnlineLQR, self).reset()
        self.policy = None

    def update_policy(self, t, x):
        if t == 0:
            self.policy = self.initial_policy()
        else:
            self.policy = self.run_lqr(t, x, self.policy)

    def compute_action(self, x):
        """
        Compute nu-dimensional action from a
        time-varying LG policy's first timestep (and add noise)
        """
        # Only the first timestep of the policy is used
        lgpolicy = self.policy
        u = lgpolicy.K[0] @ x + lgpolicy.k[0]
        if self.u_noise is not None:
            u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(self.nu))
        return u

    def initial_policy(self):
        """Return LinearGaussianPolicy for timestep 0"""
        H, nu, nx = self.H, self.nu, self.nx
        # use infinite horizon LQR
        xu = slice(nx + nu)
        xp = slice(nx + nu, nx + nu + nx)

        # Mix and add regularization (equation 1)
        sigma, mun = self.dynamics.sigma, self.dynamics.mu

        sigma[xu, xu] = sigma[xu, xu]  # + self.sigreg * np.eye(nx + nu)
        sigma_inv = np.linalg.inv(sigma[xu, xu])

        Fm = sigma[xu, xp].T @ sigma_inv
        fv = mun[xp] - Fm @ mun[xu]

        init_noise = self.u_noise
        tile = (H, 1, 1)
        cholPSig = np.tile(np.sqrt(init_noise) * np.eye(nu), tile)
        PSig = np.tile(init_noise * np.eye(nu), tile)
        invPSig = np.tile(1 / init_noise * np.eye(nu), tile)

        # ignoring the affine part
        A = Fm[:, :self.nx]
        B = Fm[:, self.nx:]
        K, _, _ = trajectory.dlqr(A, B, self.Q, self.R)

        return LinearGaussianPolicy(np.tile(-K, tile), np.zeros((H, nu)), PSig, cholPSig, invPSig)

    def run_lqr(self, t, x, prev_policy, jacobian=None):
        """
        Compute a new policy given new state

        Returns:
            LinearGaussianPolicy: An updated policy
        """
        horizon = min(self.H, self.maxT - t)
        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.lqr_iter):
            # This is plain LQR
            lgpolicy, reg_mu, reg_del = trajectory.lqr(self.cost, prev_policy, self.dynamics,
                                                       horizon, t, x, self.prevx, self.prevu,
                                                       reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                                                       jacobian=jacobian,
                                                       max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy
