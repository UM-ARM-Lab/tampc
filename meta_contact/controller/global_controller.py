import abc
import logging

import numpy as np
import torch
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities.trajectory import dlqr
from meta_contact import model
from meta_contact.controller.controller import Controller
from pytorch_cem import cem
from pytorch_mppi import mppi

logger = logging.getLogger(__name__)


class GlobalLQRController(Controller):
    def __init__(self, ds, Q=1, R=1, compare_to_goal=np.subtract, u_min=None, u_max=None):
        super().__init__(compare_to_goal)
        # load data and create LQR controller
        self.nu = ds.config.nu
        self.nx = ds.config.nx
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)

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
        x = self.compare_to_goal(x, self.goal)
        u = -self.K @ x

        if self.u_max is not None:
            u = np.clip(u, self.u_min, self.u_max)
        return u


class QRCostOptimalController(Controller):
    def __init__(self, dynamics, config, Q=1, R=1, compare_to_goal=torch.sub, u_min=None, u_max=None, device='cpu'):
        super().__init__(compare_to_goal)

        self.nu = config.nu
        self.nx = config.nx
        self.dtype = torch.double
        self.d = device
        self.u_min, self.u_max = tensor_utils.ensure_tensor(self.d, self.dtype, *math_utils.get_bounds(u_min, u_max))
        self.dynamics = dynamics

        if torch.is_tensor(Q):
            self.Q = Q.to(device=self.d)
            assert self.Q.shape[0] == self.nx
        else:
            self.Q = np.eye(self.nx, dtype=self.dtype, device=self.d) * Q
        if torch.is_tensor(R):
            self.R = R.to(device=self.d)
            assert self.R.shape[0] == self.nu
        else:
            self.R = torch.eye(self.nu, dtype=self.dtype, device=self.d) * R

    def _running_cost(self, state, action):
        diff = self.compare_to_goal(state, torch.tensor(self.goal, dtype=state.dtype, device=state.device))
        cost = linalg.batch_quadratic_product(diff, self.Q) + linalg.batch_quadratic_product(action, self.R)
        return cost

    @abc.abstractmethod
    def _mpc_command(self, obs):
        pass

    def command(self, obs):
        # use learn_mpc's Cross Entropy
        u = self._mpc_command(tensor_utils.ensure_tensor(self.d, self.dtype, obs))
        if self.u_max is not None:
            u = math_utils.clip(u, self.u_min, self.u_max)
        return u


class GlobalCEMController(QRCostOptimalController):
    def __init__(self, *args, mpc_opts=None, **kwargs):
        if mpc_opts is None:
            mpc_opts = {}
        super().__init__(*args, **kwargs)
        self.mpc = cem.CEM(self.dynamics, self._running_cost, self.nx, self.nu, u_min=self.u_min, u_max=self.u_max,
                           device=self.d, **mpc_opts)

    def _mpc_command(self, obs):
        return self.mpc.command(obs)


class GlobalMPPIController(QRCostOptimalController):
    def __init__(self, *args, use_bounds=True, mpc_opts=None, **kwargs):
        if mpc_opts is None:
            mpc_opts = {}
        super().__init__(*args, **kwargs)
        # if not given we give it a default value
        noise_sigma = mpc_opts.pop('noise_sigma', None)
        if noise_sigma is None:
            if torch.is_tensor(self.u_max):
                noise_sigma = torch.diag(self.u_max)
            else:
                noise_mult = self.u_max if self.u_max is not None else 1
                noise_sigma = torch.eye(self.nu, dtype=self.dtype) * noise_mult
        # there's interesting behaviour for MPPI if we don't pass in bounds - it'll be optimistic and try to exploit
        # regions in the dynamics where we don't know the effects of control
        if use_bounds:
            u_min, u_max = self.u_min, self.u_max
        else:
            u_min, u_max = None, None
        self.mpc = mppi.MPPI(self.dynamics, self._running_cost, self.nx, u_min=u_min, u_max=u_max,
                             noise_sigma=noise_sigma, device=self.d, **mpc_opts)

    def reset(self):
        self.mpc.reset()

    def _mpc_command(self, obs):
        return self.mpc.command(obs)
