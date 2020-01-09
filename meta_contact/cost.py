""" This file defines utility classes and functions for costs from GPS. """
import numpy as np

RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4


def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0):
    """
    Return a time-varying multiplier.
    Returns:
        A (T,) float vector containing weights for each time step.
    """
    if ramp_option == RAMP_CONSTANT:
        wpm = np.ones(T)
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32) + 1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32) + 1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros(T)
        wpm[T - 1] = 1.0
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
        np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (
            dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1
    )
    d2 = l1 * (
            (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
             (np.expand_dims(wp ** 2, axis=1) / psq)) -
            ((np.expand_dims(dscls, axis=1) *
              np.expand_dims(dscls, axis=2)) / psq ** 3)
    )
    d2 += l2 * (
            np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx


def evall1l2term_fast(wp, d, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.

    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))

    Args:
        wp:
            T x D matrix containing weights for each dimension and timestep
        d:
            T x D states to evaluate norm on
        l1: l1 loss weight
        l2: l2 loss weight
        alpha:

    Returns:
        l: T, Evaluated loss
        lx: T x Dx First derivative
        lxx: T x Dx x Dx Second derivative
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 \
        + np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = d1

    # Second order terms.
    psq = np.expand_dims(np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    d2 = l1 * ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) -
               ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 3))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    lxx = d2

    return l, lx, lxx


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
        0.5 * np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1
    # First order derivative terms.
    d1 = dscl * l2 + (
            dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        alpha + np.sum(dscl ** 2, axis=1, keepdims=True), axis=1
    )
    # TODO: Need * 2.0 somewhere in following line, or * 0.0 which is
    #      wrong but better.
    d2 = l1 * (
            (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
             (np.expand_dims(wp ** 2, axis=1) / psq)) -
            ((np.expand_dims(dscls, axis=1) *
              np.expand_dims(dscls, axis=2)) / psq ** 2)
    )
    d2 += l2 * (
            np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx


def evallogl2term_fast(wp, d, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty. (0 jacobians)

    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))

    Args:
        wp:
            T x D matrix containing weights for each dimension and timestep
        d:
            T x D states to evaluate norm on
        l1: l1 loss weight
        l2: l2 loss weight
        alpha:

    Returns:
        l: T, Evaluated loss
        lx: T x Dx First derivative
        lxx: T x Dx x Dx Second derivative
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 \
        + 0.5 * np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = d1

    # Second order terms.
    psq = np.expand_dims((alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    d2 = l1 * ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) -
               ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 2))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    lxx = d2

    return l, lx, lxx


class CostFKOnline(object):
    """Forward kinematic cost adopted from the GPS code base.

    Assumes only parts of the state space is important for reaching the goal (termed ee - end effector)
    """

    def __init__(self, ee_target, wu=None, jnt_tgt=None, jnt_wp=None, ee_idx=None, jnt_idx=None, use_jacobian=True,
                 maxT=None):
        self.dim_ee = ee_idx.stop - ee_idx.start
        # weights penalizing ee to target
        self.wp = np.ones(self.dim_ee)
        self.eetgt = ee_target
        self.ee_idx = ee_idx
        self.jnt_idx = jnt_idx
        self.jnt_tgt = jnt_tgt
        self.jnt_wp = jnt_wp
        self.use_jacobian = use_jacobian

        self.final_penalty = 1.0  # weight = sum of remaining weight * final penalty
        self.ramp_option = RAMP_CONSTANT
        self.l1 = 0.01
        self.l2 = 1.0
        self.alpha = 1e-5
        self.wu = wu

        ramp_len = maxT
        self.wpm = get_ramp_multiplier(self.ramp_option, ramp_len, wp_final_multiplier=1.0)

    def eval(self, X, U, t, jac=None):
        # Constants.
        dX = X.shape[1]
        dU = U.shape[1]
        T = X.shape[0]

        wp = self.wp * np.expand_dims(self.wpm[t:t + T], axis=-1)
        wp[-1, :] *= self.final_penalty

        l = 0.5 * np.sum(self.wu * (U ** 2), axis=1)
        lu = self.wu * U
        lx = np.zeros((T, dX))
        luu = np.tile(np.diag(self.wu), [T, 1, 1])
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # in case the goal is specified in terms of the full state
        if self.eetgt.shape[0] != self.dim_ee:
            self.eetgt = self.eetgt[self.ee_idx]
        dist = X[:, self.ee_idx] - self.eetgt

        # Derivatives w.r.t. EE dimensions
        l_ee, lx_ee, lxx_ee = evallogl2term_fast(wp, dist, self.l1, self.l2, self.alpha)
        lx[:, self.ee_idx] = lx_ee
        lxx[:, self.ee_idx, self.ee_idx] = lxx_ee

        if self.jnt_tgt is not None:
            jwp = self.jnt_wp * np.expand_dims(self.wpm[t:t + T], axis=-1)
            jwp[-1, :] *= self.final_penalty
            jdist = X[:, self.jnt_idx] - self.jnt_tgt
            l_j, lx_j, lxx_j = evallogl2term_fast(jwp, jdist, self.l1, self.l2, self.alpha)
            lx[:, self.jnt_idx] += lx_j
            lxx[:, self.jnt_idx, self.jnt_idx] += lxx_j

        # Derivatives w.r.t. Joint dimensions
        # dist = dist[:,0:3]
        # Jd = Jd[:,0:3,:]
        # wp = wp[:,0:3]
        # if self.use_jacobian:
        #     Jdd = np.zeros((T, self.dim_ee, self.dim_jnt, self.dim_jnt))
        #     l_fk, lx_fk, lxx_fk = evallogl2term(wp, dist, Jd, Jdd, self.l1, self.l2, self.alpha)
        #     l += l_fk
        #     lx[:, self.jnt_idx] += lx_fk
        #     lxx[:, self.jnt_idx, self.jnt_idx] += lxx_fk

        # TODO: Add derivatives for the actual end-effector dimensions of state
        # Right now only derivatives w.r.t. joints are considered
        return l, lx, lu, lxx, luu, lux


class CostQROnline(object):
    """QR cost adopted from the GPS code base."""

    def __init__(self, target, Q, R, compare_to_goal):
        self.Q = Q
        self.R = R
        self.eetgt = target
        self.compare_to_goal = compare_to_goal
        self.final_penalty = 1.0  # weight = sum of remaining weight * final penalty

    def __call__(self, X, U):
        X = self.compare_to_goal(X, self.eetgt)
        l = 0.5 * (np.einsum('ij,kj,ik->i', X, self.Q, X) + np.einsum('ij,kj,ik->i', U, self.R, U))
        return l

    def eval(self, X, U, t, jac=None):
        # Constants.
        nx = X.shape[1]
        nu = U.shape[1]
        T = X.shape[0]

        l = self.__call__(X, U)
        X = self.compare_to_goal(X, self.eetgt)
        lu = U @ self.R
        lx = X @ self.Q
        luu = np.tile(self.R, (T, 1, 1))
        lxx = np.tile(self.Q, (T, 1, 1))
        lux = np.zeros((T, nu, nx))

        return l, lx, lu, lxx, luu, lux
