import numpy as np
from meta_contact.controller.controller import Controller
from meta_contact.experiment import interactive_block_pushing as exp
from pytorch_mppi import mppi
from pytorch_cem import cem
from arm_pytorch_utilities import linalg
import torch
import abc

import logging

from arm_pytorch_utilities.trajectory import dlqr

logger = logging.getLogger(__name__)


class GlobalLQRController(Controller):
    def __init__(self, R=1, predict_difference=False):
        super().__init__()
        # load data and create LQR controller
        ds = exp.RawPushDataset(predict_difference=predict_difference)
        n = ds.XU.shape[1]
        self.nu = 2
        self.nx = n - self.nu

        XU = ds.XU.numpy()
        Y = ds.Y.numpy()
        # get dynamics
        params, res, rank, _ = np.linalg.lstsq(XU, Y)
        if predict_difference:
            # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
            self.A = np.diag([1., 1., 1., 1., 1.])
            self.B = np.zeros((self.nx, self.nu))
            self.A[2:, :] += params[:self.nx, :].T
            self.B[0, 0] = 1
            self.B[1, 1] = 1
            self.B[2:, :] += params[self.nx:, :].T
        else:
            # predict dynamics rather than difference
            self.A = params[:self.nx, :].T
            self.B = params[self.nx:, :].T

        # TODO increase Q for yaw later (and when that becomes part of goal)
        # self.Q = np.diag([0, 0, 0.1, 0.1, 0])
        # self.Q = np.diag([1, 1, 0, 0, 0])
        self.Q = np.diag([0., 0., 1, 1, 0])
        self.R = np.diag([R for _ in range(self.nu)])

        # confirm in MATLAB
        # import os
        # from meta_contact import cfg
        # scipy.io.savemat(os.path.join(cfg.DATA_DIR, 'sys.mat'), {'A': self.A, 'B': self.B})
        self.K, S, E = dlqr(self.A, self.B, self.Q, self.R)

        # self.Q = np.diag([1, 1])
        # self.K, S, E = dlqr(self.A[:2, :2], self.B[:2], self.Q, self.R)
        # K = np.zeros((2, 5))
        # K[:, :2] = self.K
        # self.K = K

        # hand designed K works
        # self.K = -np.array([[0, 0, -0.05, 0, 0], [0, 0, 0, 0.05, 0]])

    def command(self, obs):
        # remove the goal from xb yb
        x = np.array(obs)
        # x[0:2] -= self.goal[2:4]
        x[2:4] -= self.goal[2:4]
        u = -self.K @ x.reshape((self.nx, 1))

        n = np.linalg.norm(u)
        if n > 0.04:
            u = u / n * 0.04
        print(x)
        return u


class QRCostOptimalController(Controller):
    def __init__(self, R=1):
        super().__init__()

        self.nu = 2
        self.nx = 5
        self.max_push_mag = 0.03
        self.dtype = torch.double

        self.Q = torch.diag(torch.tensor([0, 0, 1, 1, 0], dtype=self.dtype))
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
        n = torch.norm(u)
        if n > self.max_push_mag:
            u = u / n * self.max_push_mag
        return u


class GlobalCEMController(QRCostOptimalController):
    def __init__(self, dynamics, R=1, **kwargs):
        super().__init__(R=R)
        self.mpc = cem.CEM(dynamics, self._running_cost, self.nx, self.nu, init_cov_diag=self.max_push_mag,
                           ctrl_max_mag=self.max_push_mag, **kwargs)

    def _mpc_command(self, obs):
        return self.mpc.command(obs)


class GlobalMPPIController(QRCostOptimalController):
    def __init__(self, dynamics, R=1, **kwargs):
        super().__init__(R=R)
        noise_sigma = torch.eye(self.nu, dtype=self.dtype) * self.max_push_mag
        self.mpc = mppi.MPPI(dynamics, self._running_cost, self.nx, noise_sigma=noise_sigma, **kwargs)

    def _mpc_command(self, obs):
        return self.mpc.command(obs)
