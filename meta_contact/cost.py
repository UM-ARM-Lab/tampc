""" This file defines utility classes and functions for costs from GPS. """
import torch
from arm_pytorch_utilities import linalg


class CostQROnlineTorch:
    def __init__(self, target, Q, R, compare_to_goal):
        self.Q = Q
        self.R = R
        self.eetgt = target
        self.compare_to_goal = compare_to_goal
        self.final_penalty = 1.0  # weight = sum of remaining weight * final penalty

    def __call__(self, X, U):
        X = self.compare_to_goal(X, self.eetgt)
        l = linalg.batch_quadratic_product(X, self.Q) + linalg.batch_quadratic_product(U, self.R)
        return l

    def eval(self, X, U, t, jac=None):
        """
        Get cost and up to its second order derivatives wrt X and U (to approximate a quadratic cost)

        :param X:
        :param U:
        :param t:
        :param jac:
        :return:
        """
        nx = X.shape[1]
        nu = U.shape[1]
        T = X.shape[0]

        l = self.__call__(X, U)
        X = self.compare_to_goal(X, self.eetgt)
        lu = 2 * U @ self.R
        lx = 2 * X @ self.Q
        luu = 2 * self.R.repeat(T, 1, 1)
        lxx = 2 * self.Q.repeat(T, 1, 1)
        lux = torch.zeros((T, nu, nx), dtype=X.dtype, device=X.device)

        return l, lx, lu, lxx, luu, lux
