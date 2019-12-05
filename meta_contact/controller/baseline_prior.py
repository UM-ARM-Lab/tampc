import numpy as np
import scipy.linalg
from meta_contact.controller.controller import Controller
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior
import logging

logger = logging.getLogger(__name__)


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """

    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


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
    def __init__(self, checkpoint=None):
        super().__init__()
        ds = exp.PushDataset()
        self.prior = prior.Prior('first', ds, 1e-3, 1e-5)
        # learn prior model on data
        # load data if we already have some, otherwise train from scratch
        if checkpoint and self.prior.load(checkpoint):
            logger.info("loaded checkpoint %s", checkpoint)
        else:
            self.prior.learn_model(500)

    def command(self, obs):
        # TODO use learn_mpc's Cross Entropy class here
        u = (np.random.random((2,)) - 0.5)
        return u
