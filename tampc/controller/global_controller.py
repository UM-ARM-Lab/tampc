import logging

import numpy as np
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities.trajectory import dlqr
from tampc.dynamics import model
from tampc.controller.controller import Controller

logger = logging.getLogger(__name__)


class GlobalLQRController(Controller):
    def __init__(self, ds, Q=1, R=1, compare_to_goal=np.subtract, u_min=None, u_max=None):
        super().__init__(compare_to_goal)
        # load data and create LQR controller
        config = ds.original_config()
        self.nu = config.nu
        self.nx = config.nx
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)

        self.Q = tensor_utils.ensure_diagonal(Q, self.nx).cpu().numpy()
        self.R = tensor_utils.ensure_diagonal(R, self.nu).cpu().numpy()

        self.A, self.B, self.K = None, None, None
        self.update_model(ds)

        # confirm in MATLAB
        # import os
        # from tampc import cfg
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
        u = -self.K @ x.reshape(-1)

        if self.u_max is not None:
            u = np.clip(u, self.u_min, self.u_max)
        return u
