from meta_contact.controller.controller import Controller
from meta_contact import cost
from meta_contact.online_dynamics import OnlineDynamics
from meta_contact.prior import gaussian_params_from_dataset
import logging
import numpy as np
from arm_pytorch_utilities.policy.lin_gauss import LinearGaussianPolicy
from arm_pytorch_utilities import trajectory, math_utils

LOGGER = logging.getLogger(__name__)


class OnlineController(Controller):
    """Controller mixing locally linear model with prior model from https://arxiv.org/pdf/1509.06841.pdf"""

    def __init__(self, prior, ds, max_timestep=100, Q=1, R=1, horizon=15, lqr_iter=1, init_gamma=0.1,
                 compare_to_goal=np.subtract, u_min=None, u_max=None):
        super().__init__(compare_to_goal)
        self.dX = ds.config.nx
        self.dU = ds.config.nu
        self.H = horizon
        self.maxT = max_timestep
        self.init_gamma = init_gamma
        self.gamma = self.init_gamma
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)

        self.u_noise = 0.001
        # TODO get rid of these environment specific parameters
        # self.block_idx = slice(0, 2)
        # self.block_idx = slice(2, 4)

        self.ds = ds
        sigma, mu = gaussian_params_from_dataset(ds)
        self.dyn_init_sig = sigma
        self.dyn_init_mu = mu

        # LQR options
        self.min_mu = 1e-6
        self.del0 = 2
        self.lqr_discount = 0.9
        self.lqr_iter = lqr_iter

        # Init objects
        if np.isscalar(Q):
            self.Q = np.eye(self.dX) * Q
        else:
            self.Q = Q
            assert self.Q.shape[0] == self.dX
        self.weight_u = np.ones(self.dU) * R
        self.R = np.diag(self.weight_u)
        # self.cost = cost.CostFKOnline(self.goal, wu=self.weight_u, ee_idx=self.block_idx, maxT=self.maxT,
        #                               use_jacobian=False)
        self.cost = cost.CostQROnline(self.goal, self.Q, self.R, self.compare_to_goal)
        self.dynamics = OnlineDynamics(self.gamma, prior, self.dyn_init_mu, self.dyn_init_sig, self.dX, self.dU)

        self.prevx = None
        self.prevu = None
        self.prev_policy = None
        self.u_history = []

    def update_prior(self, prior):
        self.dynamics.prior = prior

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

        add_noise = noise is not None
        u = self.compute_action(lgpolicy, x, add_noise)
        # if self.prevu is not None:  # smooth
        #    u = 0.5*u+0.5*self.prevu
        self.prev_policy = lgpolicy
        self.prevx = x
        self.prevu = u
        self.u_history.append(u)

        return u

    def initial_policy(self):
        """Return LinearGaussianPolicy for timestep 0"""
        H, dU, dX = self.H, self.dU, self.dX
        # use infinite horizon LQR
        xu = slice(dX + dU)
        xp = slice(dX + dU, dX + dU + dX)

        # Mix and add regularization (equation 1)
        sigma, mun = self.dyn_init_sig, self.dyn_init_mu

        XU, Y, _ = self.ds.training_set()
        XU, Y = XU.numpy(), Y.numpy()
        N = XU.shape[0]

        # XUY = np.concatenate((XU,Y), axis=1)
        # mun = np.mean(XUY, axis=0)
        # sigma = np.cov(XUY, rowvar=False)

        sigma[xu, xu] = sigma[xu, xu]  # + self.sigreg * np.eye(dX + dU)
        sigma_inv = np.linalg.inv(sigma[xu, xu])
        # Solve normal equations to get dynamics. (equation 2)
        # Fm = sigma_inv.dot(sigma[xu, xp]).T  # f_xu
        # fv = mun[xp] - Fm.dot(mun[xu])  # f_c

        Fm = sigma[xu, xp].T @ sigma_inv
        fv = mun[xp] - Fm @ mun[xu]

        init_noise = self.u_noise
        tile = (H, 1, 1)
        cholPSig = np.tile(np.sqrt(init_noise) * np.eye(dU), tile)
        PSig = np.tile(init_noise * np.eye(dU), tile)
        invPSig = np.tile(1 / init_noise * np.eye(dU), tile)

        # ignoring the affine part
        A = Fm[:, :self.dX]
        B = Fm[:, self.dX:]
        K, _, _ = trajectory.dlqr(A, B, self.Q, self.R)

        # # compare against ordinary least squares
        # params, res, rank, _ = np.linalg.lstsq(XU, Y)
        # # compare against affine
        # params_a, res_a, rank_a, _ = np.linalg.lstsq(np.concatenate((XU, np.ones((N, 1))), axis=1), Y)
        # Fm_a = params_a[:7, :]
        # fv_a = params_a[7, :]
        # yhat = XU @ Fm.T + np.tile(fv, (XU.shape[0], 1))
        # yhat_ls = XU @ params
        # yhat_a = XU @ Fm_a + np.tile(fv_a, (N, 1))
        #
        # e = np.linalg.norm(yhat - Y)
        # e_ls = np.linalg.norm(yhat_ls - Y)
        # e_a = np.linalg.norm(yhat_a - Y)

        return LinearGaussianPolicy(np.tile(-K, tile), np.zeros((H, dU)), PSig, cholPSig, invPSig)

    def compute_action(self, lgpolicy, x, add_noise=True):
        """
        Compute dU-dimensional action from a
        time-varying LG policy's first timestep (and add noise)
        """
        # Only the first timestep of the policy is used
        u = lgpolicy.K[0] @ x + lgpolicy.k[0]
        if add_noise:
            u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(self.dU))
        if self.u_max:
            u = np.clip(u, -self.u_max, self.u_max)
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
            lgpolicy, reg_mu, reg_del = trajectory.lqr(self.cost, prev_policy, self.dynamics,
                                                       horizon, t, x, self.prevx, self.prevu,
                                                       reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                                                       jacobian=jacobian,
                                                       max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy
