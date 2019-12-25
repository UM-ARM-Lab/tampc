from meta_contact.controller.controller import Controller
from meta_contact.cost import CostFKOnline
from meta_contact.online_dynamics import OnlineDynamics
from meta_contact.prior import gaussian_params_from_dataset
import logging
import numpy as np
from arm_pytorch_utilities.policy.lin_gauss import LinearGaussianPolicy
from arm_pytorch_utilities.trajectory import lqr

LOGGER = logging.getLogger(__name__)
CLIP_U = 0.03


class OnlineController(Controller):
    """Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf"""

    def __init__(self, prior, ds=None, max_timestep=100, R=1, horizon=15, lqr_iter=1, init_gamma = 0.1):
        super().__init__()
        self.dX = 5
        self.dU = 2
        self.H = horizon
        self.wu = np.array([1., 1.]) * R
        self.maxT = max_timestep
        self.init_gamma = init_gamma
        self.gamma = self.init_gamma
        self.u_noise = 0.001
        # self.block_idx = slice(0, 2)
        self.block_idx = slice(2, 4)

        if ds is not None:
            sigma, mu = gaussian_params_from_dataset(ds)
            self.dyn_init_sig = sigma
            self.dyn_init_mu = mu
        else:
            self.dyn_init_mu = np.zeros(self.dX * 2 + self.dU)
            self.dyn_init_sig = np.eye(self.dX * 2 + self.dU)

        # LQR options
        self.min_mu = 1e-6
        self.del0 = 2
        self.lqr_discount = 0.9
        self.lqr_iter = lqr_iter

        # Init objects
        self.cost = CostFKOnline(self.goal, wu=self.wu, ee_idx=self.block_idx, maxT=self.maxT,
                                 use_jacobian=False)
        self.prior = prior
        self.dynamics = OnlineDynamics(self.gamma, self.prior, self.dyn_init_mu, self.dyn_init_sig, self.dX, self.dU)

        self.prevx = None
        self.prevu = None
        self.prev_policy = None
        self.u_history = []

    def set_goal(self, goal):
        super().set_goal(goal)
        self.cost.eetgt = goal

    def command(self, obs):
        return self.act(obs, obs, len(self.u_history))

    def act(self, x, obs, t, noise=None, sample=None):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        if t == 0:
            lgpolicy = self.initial_policy()
        else:
            self.dynamics.update(self.prevx, self.prevu, x)
            lgpolicy = self.run_lqr(t, x, self.prev_policy)

        u = self.compute_action(lgpolicy, x)
        # if self.prevu is not None:  # smooth
        #    u = 0.5*u+0.5*self.prevu
        self.prev_policy = lgpolicy
        self.prevx = x
        self.prevu = u
        self.u_history.append(u)

        return u

    def initial_policy(self):
        """Return LinearGaussianPolicy for timestep 0"""
        dU, dX = self.dU, self.dX
        H = self.H
        K = np.zeros((H, dU, dX))
        k = np.zeros((H, dU))
        # K = self.offline_K[:H]  # np.zeros((H, dU, dX))
        # k = self.offline_k[:H]  # np.zeros((H, dU))
        init_noise = 1
        self.gamma = self.init_gamma
        cholPSig = np.tile(np.sqrt(init_noise) * np.eye(dU), [H, 1, 1])
        PSig = np.tile(init_noise * np.eye(dU), [H, 1, 1])
        invPSig = np.tile(1 / init_noise * np.eye(dU), [H, 1, 1])
        return LinearGaussianPolicy(K, k, PSig, cholPSig,
                                    invPSig)

    def compute_action(self, lgpolicy, x, add_noise=True):
        """
        Compute dU-dimensional action from a
        time-varying LG policy's first timestep (and add noise)
        """
        # Only the first timestep of the policy is used
        u = lgpolicy.K[0].dot(x) + lgpolicy.k[0]
        if add_noise:
            u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(self.dU))
        u = np.clip(u, -CLIP_U, CLIP_U)
        return u

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
            lgpolicy, reg_mu, reg_del = lqr(self.cost, prev_policy, self.dynamics,
                                            horizon, t, x, self.prevx, self.prevu,
                                            reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                                            jacobian=jacobian,
                                            max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy
