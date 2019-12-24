# TODO deprecated, look at online_dynamics
import numpy as np
import scipy.linalg
from meta_contact.controller.controller import Controller
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior
from meta_contact import util

import logging

logger = logging.getLogger(__name__)


class LocallyLinearLQRController(Controller):
    # TODO make beta dependent on relative accuracy of local prediction
    def __init__(self, beta=0.95):
        super().__init__()
        self.beta = beta
        # initialize mean and covariance based on a dataset
        ds = exp.RawPushDataset()
        N, n = ds.XU.shape
        self.nu = 2
        self.nx = n - self.nu

        XU = ds.XU.numpy()
        Y = ds.Y.numpy()

        NX = XU[1:, :5]
        P = np.column_stack((XU[:-1], NX))
        self.mu = np.mean(P, 0)
        # delta - mu*mu' = cov
        self.delta = np.cov(P, rowvar=False) + self.mu.reshape(-1, 1) @ self.mu.reshape(1, -1)

        self.last_xu = None

        # TODO increase Q for yaw later (and when that becomes part of goal)
        self.Q = np.diag([0, 0, 1, 1, 0])
        # self.Q = np.diag([1, 1, 0, 0, 0])
        self.R = np.diag([10 for _ in range(self.nu)])

        # sigma_hat = self.delta - (1 - self.beta) * self.mu.reshape(-1, 1) @ self.mu.reshape(1, -1)
        # # extract parameters
        # n = self.nx + self.nu
        # fxu, res, rank, eigs = np.linalg.lstsq(sigma_hat[:n, :n], sigma_hat[:n, n:])
        # A = fxu[:self.nx].T
        # B = fxu[self.nx:].T
        #
        # params, res, rank, _ = np.linalg.lstsq(XU[:-1], NX)
        # # AA = np.diag([1., 1., 1., 1., 1.])
        # AA = params[:self.nx, :].T
        # BB = params[self.nx:].T
        # print('compare')

    def set_goal(self, goal):
        # assume goal is xb yb
        self.goal = np.array([0, 0, goal[0], goal[1], 0])
        # self.goal = np.array([goal[0], goal[1], 0, 0, 0])

    def command(self, obs):
        xn = np.array(obs)
        # TODO update our estimate of local dynamics
        if self.last_xu is not None:
            p = np.concatenate((self.last_xu, xn))
            self.mu = self.beta * self.mu + (1 - self.beta) * p
            self.delta = self.beta * self.delta + (1 - self.beta) * p.reshape(-1, 1) @ p.reshape(1, -1)

        sigma_hat = self.delta - (1 - self.beta) * self.mu.reshape(-1, 1) @ self.mu.reshape(1, -1)
        # TODO evaluate prior here

        # extract parameters
        n = self.nx + self.nu
        fxu, res, rank, eigs = np.linalg.lstsq(sigma_hat[:n, :n], sigma_hat[:n, n:])
        A = fxu[:self.nx].T
        B = fxu[self.nx:].T
        # LQR with new dynamics
        K, S, E = util.dlqr(A,B,self.Q,self.R)
        u = -K @ (xn - self.goal).reshape(-1, 1)
        u = np.squeeze(np.asarray(u))

        u = u / np.linalg.norm(u) * 0.04
        self.last_xu = np.concatenate((xn, u))
        return u
