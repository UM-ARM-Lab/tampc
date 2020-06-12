""" This file defines utility classes and functions for costs from GPS. """
import torch
import logging
import abc
from arm_pytorch_utilities import linalg, tensor_utils

logger = logging.getLogger(__name__)


def qr_cost(diff_function, X, X_goal, Q, R, U=None, U_goal=None, terminal=False):
    X = diff_function(X, X_goal)
    c = linalg.batch_quadratic_product(X, Q)
    if U is not None and not terminal:
        if U_goal is not None:
            U = U - U_goal
        c += linalg.batch_quadratic_product(U, R)
    return c


class Cost(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, U=None, terminal=False):
        """Get cost of state and control"""


class CostQROnlineTorch(Cost):
    def __init__(self, target, Q, R, compare_to_goal):
        self.Q = Q
        self.R = R
        self.goal = target
        self.compare_to_goal = compare_to_goal

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        return qr_cost(self.compare_to_goal, X, self.goal, self.Q, self.R, U=U, U_goal=None, terminal=terminal)

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
        X = self.compare_to_goal(X, self.goal)
        lu = 2 * U @ self.R
        lx = 2 * X @ self.Q
        luu = 2 * self.R.repeat(T, 1, 1)
        lxx = 2 * self.Q.repeat(T, 1, 1)
        lux = torch.zeros((T, nu, nx), dtype=X.dtype, device=X.device)

        return l, lx, lu, lxx, luu, lux


def min_cost(costs):
    return costs.min(dim=0).values


class CostQRSet(Cost):
    """If cost is not dependent on a single target but a set of targets"""

    def __init__(self, goal_set, Q, R, compare_to_goal, reduce=min_cost, goal_weights=None):
        self.Q = Q
        self.R = R
        self.goal_set = goal_set
        self.goal_weights = goal_weights
        self.compare_to_goal = compare_to_goal
        self.reduce = reduce

        logger.debug("state set\n%s", goal_set)

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []

        for i, goal in enumerate(self.goal_set):
            if type(goal) is tuple:
                goal_x, goal_u = goal
                c = qr_cost(self.compare_to_goal, X, goal_x, self.Q, self.R, U=U, U_goal=goal_u, terminal=terminal)
            else:
                c = qr_cost(self.compare_to_goal, X, goal, self.Q, self.R, U=U, terminal=terminal)
            if self.goal_weights is not None:
                c *= self.goal_weights[i]
            costs.append(c)

        if len(costs):
            costs = torch.stack(costs, dim=0)
            costs = self.reduce(costs)
        else:
            costs = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        return costs


class ComposeCost(Cost):
    def __init__(self, costs):
        self.fn = costs

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = [cost_fn(X, U, terminal) for cost_fn in self.fn]

        # _debug_cost_ratio = costs[1] / costs[0]
        # _debug_cost_ratio = torch.sort(_debug_cost_ratio).values
        # if _debug_cost_ratio[0] > 0:
        #     logger.info("sampled cost ratios %s", _debug_cost_ratio)

        costs = torch.stack(costs, dim=0)

        costs = costs.sum(dim=0)
        return costs
