import numpy as np
from arm_pytorch_utilities.trajectory import invert_psd
from meta_contact import prior


class OnlineDynamics(object):
    """ Moving average estimate of locally linear dynamics from https://arxiv.org/pdf/1509.06841.pdf

    Note gamma here is (1-gamma) described in the paper, so high gamma forgets quicker
    """

    def __init__(self, gamma, prior: prior.OnlineDynamicsPrior, init_mu, init_sigma, dX, dU, N=1, sigreg=1e-5):
        self.gamma = gamma
        self.prior = prior
        self.sigreg = sigreg  # Covariance regularization (adds sigreg*eye(N))
        self.dX = dX
        self.dU = dU
        self.empsig_N = N
        self.emp_error = None
        self.prior_error = None

        self.prior_trust_coefficient = 0.1  # the lower it is the more we trust the prior; 0 means only ever use prior
        # TODO can track only mu and xxt, generate sigma when asking for dynamics?
        # Initial values
        self.mu = init_mu
        self.sigma = init_sigma
        self.xxt = init_sigma + np.outer(self.mu, self.mu)

    def evaluate_error(self, px, pu, cx, cu):
        """After updating dynamics and using that dynamics to plan an action,
        evaluate the error of empirical and prior dynamics"""
        # can't evaluate if it's the first action
        if px is None:
            self.emp_error = self.prior_error = None
            return
        xu, pxu, xux = concatenate_state_control(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_params(self.dX, self.dU, xu, pxu, xux)
        # evaluate the accuracy of empirical and prior dynamics on (xux')
        Fe, fe, _ = conditioned_dynamics(self.dX, self.dU, self.sigma, self.mu, sigreg=self.sigreg)
        emp_x = evaluate_dynamics(px, pu, Fe, fe)
        # prior dynamics
        Fp, fp, _ = conditioned_dynamics(self.dX, self.dU, Phi / n0, mu0, sigreg=self.sigreg)
        prior_x = evaluate_dynamics(px, pu, Fp, fp)

        # compare against actual x'
        self.emp_error = np.linalg.norm(emp_x - cx)
        self.prior_error = np.linalg.norm(prior_x - cx)
        # TODO update gamma based on the relative error of these dynamics
        # rho = self.emp_error / self.prior_error
        # # high gamma means to trust empirical model (paper uses 1-rho, but this makes more sense)
        # self.gamma = self.prior_trust_coefficient / rho

    def update(self, px, pu, cx):
        """ Perform a moving average update on the current dynamics """
        xux = np.r_[px, pu, cx].astype(np.float32)

        gamma = self.gamma
        # Do a moving average update (equations 3,4)
        self.mu = self.mu * (1 - gamma) + xux * (gamma)
        self.xxt = self.xxt * (1 - gamma) + np.outer(xux, xux) * (gamma)
        self.xxt = 0.5 * (self.xxt + self.xxt.T)
        self.sigma = self.xxt - np.outer(self.mu, self.mu)

    def get_dynamics(self, t, px, pu, cx, cu):
        """
        Compute F, f - the linear dynamics where next_x = F*[curx, curu] + f
        """
        # prior parameters
        xu, pxu, xux = concatenate_state_control(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_params(self.dX, self.dU, xu, pxu, xux)

        # mix prior and empirical distribution
        sigma, mu = prior.mix_distributions(self.sigma, self.mu, self.empsig_N, Phi, mu0, m, n0)
        return conditioned_dynamics(self.dX, self.dU, sigma, mu, self.sigreg)


def concatenate_state_control(px, pu, cx, cu):
    # TODO use faster way of constructing xu and pxu (concatenate)
    xu = np.r_[cx, cu].astype(np.float32)
    pxu = np.r_[px, pu]
    xux = np.r_[px, pu, cx]
    return xu, pxu, xux


def get_locally_linear_dynamics(nx, nu, px, pu, cx, cu, N, emp_mu, emp_sigma, prior, sigreg=1e-5):
    xu, pxu, xux = concatenate_state_control(px, pu, cx, cu)

    # Mix and add regularization (equation 1)
    sigma, mun = prior.mix(nx, nu, xu, pxu, xux, emp_sigma, emp_mu, N)
    return conditioned_dynamics(nx, nu, sigma, mun, sigreg)


def conditioned_dynamics(nx, nu, sigma, mu, sigreg=1e-5):
    it = slice(nx + nu)
    ip = slice(nx + nu, nx + nu + nx)
    sigma[it, it] = sigma[it, it] + sigreg * np.eye(nx + nu)
    sigma_inv = invert_psd(sigma[it, it])

    # Solve normal equations to get dynamics. (equation 2)
    Fm = sigma_inv.dot(sigma[it, ip]).T  # f_xu
    fv = mu[ip] - Fm.dot(mu[it])  # f_c
    dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)  # F
    dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)  # Guarantee symmetric

    return Fm, fv, dyn_covar


def evaluate_dynamics(x, u, F, f, cov=None):
    # TODO sample from multivariate normal if covariance is given
    xp = F @ np.concatenate((x, u)) + f
    return xp
