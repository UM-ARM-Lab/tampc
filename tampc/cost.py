""" This file defines utility classes and functions for costs from GPS. """
import torch
import logging
import abc
import numpy as np
from arm_pytorch_utilities import linalg, tensor_utils, math_utils

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
        numpy API for LQR

        :param X:
        :param U:
        :param t:
        :param jac:
        :return:
        """
        # Constants.
        nx = X.shape[1]
        nu = U.shape[1]
        T = X.shape[0]

        X = self.compare_to_goal(X, self.goal)
        l = 0.5 * (np.einsum('ij,kj,ik->i', X, self.Q, X) + np.einsum('ij,kj,ik->i', U, self.R, U))
        lu = U @ self.R
        lx = X @ self.Q
        luu = np.tile(self.R, (T, 1, 1))
        lxx = np.tile(self.Q, (T, 1, 1))
        lux = np.zeros((T, nu, nx))

        return l, lx, lu, lxx, luu, lux


def min_cost(costs):
    return costs.min(dim=0).values


def default_similarity(U, goal_u):
    return torch.cosine_similarity(U, goal_u, dim=-1).clamp(0, 1)


def identity(cost):
    return cost


class CostQRSet(Cost):
    """If cost is not dependent on a single target but a set of targets"""

    def __init__(self, goal_set, Q, R, compare_to_goal, u_similarity=None, reduce=min_cost,
                 process_cost=identity, goal_weights=None):
        self.Q = Q
        self.R = R
        self.goal_set = goal_set
        self.goal_weights = goal_weights
        self.compare_to_goal = compare_to_goal
        self.reduce = reduce
        self.u_similarity = u_similarity if u_similarity else default_similarity
        self.process_cost = process_cost

        logger.debug("state set\n%s", goal_set)

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []
        if U is not None:
            assert X.shape[:-1] == U.shape[:-1]

        for i, goal in enumerate(self.goal_set):
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


class GoalSetCost(Cost):
    """If cost is not dependent on a single target but a set of targets"""

    def __init__(self, goal_set, compare_to_goal, state_dist, reduce, scale=1, goal_weights=None):
        self.goal_set = goal_set
        self.goal_weights = goal_weights
        self.compare_to_goal = compare_to_goal
        self.state_dist = state_dist
        self.reduce = reduce
        self.scale = scale

        logger.debug("state set\n%s", goal_set)

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []
        if U is not None:
            assert X.shape[:-1] == U.shape[:-1]

        for i, goal in enumerate(self.goal_set):
            c = self.compare_to_goal(X, goal)
            c = self.state_dist(c).square() * self.scale

            if self.goal_weights is not None:
                c *= self.goal_weights[i]
            costs.append(c)

        if len(costs):
            costs = torch.stack(costs, dim=0)
            costs = self.reduce(costs)
        else:
            costs = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        return costs


# not really important, just has to be high enough to avoid saturating everywhere
TRAP_MAX_COST = 100000


class TrapSetCost(Cost):
    """Avoid re-entering traps"""

    def __init__(self, goal_set, compare_to_goal, state_dist, u_similarity, reduce, goal_weights=None):
        self.goal_set = goal_set
        self.goal_weights = goal_weights
        self.compare_to_goal = compare_to_goal
        self.state_dist = state_dist
        self.reduce = reduce
        self.u_similarity = u_similarity

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []
        if U is not None:
            assert X.shape[:-1] == U.shape[:-1]

        for i, goal in enumerate(self.goal_set):
            # have to have both high cost in x and u (multiplicative rather than additive)
            goal_x, goal_u = goal
            c_x = self.compare_to_goal(X, goal_x)
            c_x = self.state_dist(c_x).square()
            c_x = torch.clamp((1 / c_x), 0, TRAP_MAX_COST)
            c_u = 0 if terminal else (1 if U is None else self.u_similarity(U, goal_u))
            c = c_x * c_u
            if self.goal_weights is not None:
                c *= self.goal_weights[i]
            costs.append(c)

        if len(costs):
            costs = torch.stack(costs, dim=0)
            costs = self.reduce(costs)
        else:
            costs = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        return costs


class AvoidContactAtGoalCost(Cost):
    """Avoid bringing contact set close to goal"""

    def __init__(self, goal, scale=1):
        self.goal = goal
        self.scale = scale

    @tensor_utils.handle_batch_input
    def __call__(self, contact_set, contact_data, U=None, terminal=False):
        return contact_set.goal_cost(self.goal, contact_data) * self.scale


class CostLeaveState(Cost):
    """Reward for distance away from current state"""

    def __init__(self, X, Q, R, compare_to_goal, max_cost):
        self.x = X
        self.Q = Q
        self.R = R
        self.compare_to_goal = compare_to_goal
        self.max_cost = max_cost

    def update_state(self, x):
        self.x = x

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        dist_cost = qr_cost(self.compare_to_goal, X, self.x, self.Q, self.R, U=U, terminal=terminal)
        # reward for having high dist (invert cost)
        costs = torch.clamp((1 / dist_cost), 0, self.max_cost)
        return costs


class ComposeCost(Cost):
    def __init__(self, costs, weights=None):
        self.fn = costs
        self.weights = weights

    def add(self, cost_fn):
        self.fn.append(cost_fn)

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = [cost_fn(X, U, terminal) for cost_fn in self.fn]

        # _debug_cost_ratio = costs[1] / costs[0]
        # _debug_cost_ratio = torch.sort(_debug_cost_ratio).values
        # if _debug_cost_ratio[0] > 0:
        #     logger.info("sampled cost ratios %s", _debug_cost_ratio)

        costs = torch.stack(costs, dim=0)
        if self.weights is not None:
            if callable(self.weights):
                costs *= self.weights().view(-1, 1)
            else:
                costs *= self.weights.view(-1, 1)

        costs = costs.sum(dim=0)
        return costs


class ArtificialRepulsionCost(Cost):
    """Cost for artificial potential fields of repulsion"""

    def __init__(self, state_set, compare_to_goal, state_dist, max_dist_influence, gain=1):
        self.state_set = state_set
        self.compare_to_goal = compare_to_goal
        self.state_dist = state_dist
        self.max_dist_influence = max_dist_influence
        self.gain = gain

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        costs = []
        if U is not None:
            assert X.shape[:-1] == U.shape[:-1]

        for i, state in enumerate(self.state_set):
            rho = self.state_dist(self.compare_to_goal(X, state))
            inside_range = rho <= self.max_dist_influence
            c = torch.zeros_like(rho)
            c[inside_range] = (1 / rho[inside_range] - 1 / self.max_dist_influence).square()
            c = torch.clamp(c, 0, TRAP_MAX_COST)
            costs.append(c)

        if len(costs):
            costs = torch.stack(costs, dim=0)
            costs = costs.sum(dim=0)
        else:
            costs = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        return costs * self.gain


class APFHelicoid2D(Cost):
    """Helicoid potential field in 2D"""

    def __init__(self, rxy, oxy, clockwise=False):
        # robot and obstacle xy
        self.rxy = rxy
        self.oxy = oxy
        self.clockwise = clockwise

    @tensor_utils.handle_batch_input
    def __call__(self, X, U=None, terminal=False):
        original_angle = self.oxy - self.rxy
        original_angle = torch.atan2(original_angle[1], original_angle[0])
        Xxy = X[:, :2]
        d = self.oxy - Xxy
        angles = torch.atan2(d[:, 1], d[:, 0])

        angular_diffs = math_utils.angular_diff_batch(original_angle, angles) * torch.norm(d, dim=1)
        return angular_diffs * (1 if self.clockwise else -1)
