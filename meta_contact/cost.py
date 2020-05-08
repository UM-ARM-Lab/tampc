""" This file defines utility classes and functions for costs from GPS. """
import torch
from arm_pytorch_utilities import linalg, tensor_utils


def qr_cost(diff_function, X, X_goal, Q, R, U=None, terminal=False):
    X = diff_function(X, X_goal)
    c = linalg.batch_quadratic_product(X, Q)
    if not terminal:
        c += linalg.batch_quadratic_product(U, R)
    return c


class CostQROnlineTorch:
    def __init__(self, target, Q, R, compare_to_goal):
        self.Q = Q
        self.R = R
        self.eetgt = target
        self.compare_to_goal = compare_to_goal

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        return qr_cost(self.compare_to_goal, X, self.eetgt, self.Q, self.R, U, terminal)

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


class CostQRGoalSet:
    """If goal is not just a single target but a set of possible goals"""

    def __init__(self, goal_set, Q, R, compare_to_goal, ds, goal_weights=None, compare_in_latent_space=False):
        self.ds = ds
        self.Q = Q
        self.R = R
        self.goal_set = goal_set
        self.goal_weights = goal_weights
        self.compare_to_goal = compare_to_goal
        self.compare_in_latent_space = compare_in_latent_space

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []
        for i, goal in enumerate(self.goal_set):
            # TODO handle comparison in latent space (need to change Q)
            c = qr_cost(self.compare_to_goal, X, goal, self.Q, self.R, U, terminal)
            if self.goal_weights is not None:
                c *= self.goal_weights[i]
            costs.append(c)

        costs = torch.stack(costs, dim=0)
        # take the minimum to any of the goals
        costs = costs.min(dim=0).values
        return costs
