import numpy as np
from meta_contact.controller.controller import Controller
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior
from learn_hybrid_mpc import mpc, evaluation

import logging

from meta_contact.util import dlqr

logger = logging.getLogger(__name__)


class GlobalLQRController(Controller):
    def __init__(self, R=1):
        super().__init__()
        # load data and create LQR controller
        ds = exp.RawPushDataset()
        n = ds.XU.shape[1]
        self.nu = 2
        self.nx = n - self.nu
        # get dynamics
        params, res, rank, _ = np.linalg.lstsq(ds.XU.numpy(), ds.Y.numpy())
        # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
        self.A = np.diag([1., 1., 1., 1., 1.])
        self.B = np.zeros((self.nx, self.nu))
        # self.A[2:, :] += params[:self.nx, :].T
        self.B[0, 0] = 1
        self.B[1, 1] = 1
        # self.B[2:, :] += params[self.nx:, :].T
        # TODO increase Q for yaw later (and when that becomes part of goal)
        # self.Q = np.diag([0, 0, 0.1, 0.1, 0])
        self.Q = np.diag([1, 1, 0, 0, 0])
        self.R = np.diag([R for _ in range(self.nu)])

        # confirm in MATLAB
        # import os
        # from meta_contact import cfg
        # scipy.io.savemat(os.path.join(cfg.DATA_DIR, 'sys.mat'), {'A': self.A, 'B': self.B})
        self.K, S, E = dlqr(self.A, self.B, self.Q, self.R)
        # self.K, S, E = dlqr(self.A[:2,:2], self.B[:2], self.Q, self.R)
        # K = np.zeros((2,5))
        # K[:,:2] = self.K
        # self.K = K

    def command(self, obs):
        # remove the goal from xb yb
        x = np.array(obs)
        x[0:2] -= self.goal
        # x[2:4] -= self.goal
        u = -self.K @ x.reshape((self.nx, 1))
        print(x)
        return u


class GlobalNetworkCrossEntropyController(Controller):
    def __init__(self, model, name='', R=1, checkpoint=None, **kwargs):
        super().__init__()
        ds = exp.PushDataset(data_dir='pushing/touching.mat', **kwargs)
        self.prior = prior.Prior(model, name, ds, 1e-3, 1e-5)
        # learn prior model on data
        # load data if we already have some, otherwise train from scratch
        if checkpoint and self.prior.load(checkpoint):
            logger.info("loaded checkpoint %s", checkpoint)
        else:
            self.prior.learn_model(100)

        # freeze network
        for param in self.prior.model.parameters():
            param.requires_grad = False

        nu = 2
        self.Q = np.diag([0, 0, 1, 1, 0])
        self.R = np.diag([R for _ in range(nu)])
        self.cost = evaluation.QREvaluation(self.Q, self.R, self.Q, self.get_goal)
        max_push_mag = 0.03
        self.ce = mpc.CrossEntropy(self.prior, self.cost, 10, 175, nu, 7, 3, init_cov_diag=max_push_mag,
                                   ctrl_max_mag=max_push_mag)

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        # assume goal is xb yb
        self.goal = np.array([0, 0, goal[0], goal[1], 0])

    def command(self, obs):
        # use learn_mpc's Cross Entropy
        u = self.ce.action(np.array(obs))
        u = u / np.linalg.norm(u) * 0.04
        return u
