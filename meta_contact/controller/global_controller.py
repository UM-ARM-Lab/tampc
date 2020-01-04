import numpy as np
from meta_contact.controller.controller import Controller
from meta_contact import model
from pytorch_mppi import mppi
from pytorch_cem import cem
from arm_pytorch_utilities import linalg
import torch
import abc

import logging

from arm_pytorch_utilities.trajectory import dlqr

logger = logging.getLogger(__name__)


class GlobalLQRController(Controller):
    def __init__(self, ds, Q=1, R=1, u_max=None, goal_slice=None):
        super().__init__()
        # load data and create LQR controller
        self.nu = ds.config.nu
        self.nx = ds.config.nx
        self.u_max = u_max
        if goal_slice is None:
            self.goal_slice = slice(self.nx)

        if np.isscalar(Q):
            self.Q = np.eye(self.nx) * Q
        else:
            self.Q = Q
            assert self.Q.shape[0] == self.nx
        self.R = np.diag([R for _ in range(self.nu)])

        self.A, self.B, self.K = None, None, None
        self.update_model(ds)

        # confirm in MATLAB
        # import os
        # from meta_contact import cfg
        # scipy.io.savemat(os.path.join(cfg.DATA_DIR, 'sys.mat'), {'A': self.A, 'B': self.B})

        # self.Q = np.diag([1, 1])
        # self.K, S, E = dlqr(self.A[:2, :2], self.B[:2], self.Q, self.R)
        # K = np.zeros((2, 5))
        # K[:, :2] = self.K
        # self.K = K

        # hand designed K works
        # self.K = -np.array([[0, 0, -0.05, 0, 0], [0, 0, 0, 0.05, 0]])

    def update_model(self, ds):
        self.A, self.B = model.linear_model_from_ds(ds)
        if ds.config.predict_difference:
            self.A += np.eye(self.nx)

        self.K, S, E = dlqr(self.A, self.B, self.Q, self.R)

    def command(self, obs):
        # remove the goal from xb yb
        x = np.array(obs)
        x[self.goal_slice] -= self.goal[self.goal_slice]
        u = -self.K @ x

        if self.u_max:
            u = np.clip(u, -self.u_max, self.u_max)
        return u


class QRCostOptimalController(Controller):
    def __init__(self, ds, Q=1, R=1, u_max=None):
        super().__init__()

        self.nu = ds.config.nu
        self.nx = ds.config.nx
        self.u_max = u_max
        self.dtype = torch.double

        if torch.is_tensor(Q):
            self.Q = Q
            assert self.Q.shape[0] == self.nx
        else:
            self.Q = np.eye(self.nx) * Q
        self.R = torch.eye(self.nu, dtype=self.dtype) * R

    def _running_cost(self, state, action):
        state = state - torch.tensor(self.goal, dtype=state.dtype, device=state.device)
        cost = linalg.batch_quadratic_product(state, self.Q) + linalg.batch_quadratic_product(action, self.R)
        return cost

    @abc.abstractmethod
    def _mpc_command(self, obs):
        pass

    def command(self, obs):
        # use learn_mpc's Cross Entropy
        u = self._mpc_command(torch.tensor(obs))
        if self.u_max:
            u = torch.clamp(u, -self.u_max, self.u_max)
        return u


class GlobalCEMController(QRCostOptimalController):
    def __init__(self, dynamics, ds, Q=1, R=1, u_max=None, **kwargs):
        super().__init__(ds, Q=Q, R=R, u_max=u_max)
        self.mpc = cem.CEM(dynamics, self._running_cost, self.nx, self.nu, init_cov_diag=self.u_max,
                           ctrl_max_mag=self.u_max, **kwargs)

    def _mpc_command(self, obs):
        return self.mpc.command(obs)


class GlobalMPPIController(QRCostOptimalController):
    def __init__(self, dynamics, ds, Q=1, R=1, u_max=None, **kwargs):
        super().__init__(ds, Q=Q, R=R, u_max=u_max)
        noise_mult = self.u_max or 1
        noise_sigma = torch.eye(self.nu, dtype=self.dtype) * noise_mult
        self.mpc = mppi.MPPI(dynamics, self._running_cost, self.nx, noise_sigma=noise_sigma, **kwargs)

    def _mpc_command(self, obs):
        return self.mpc.command(obs)
