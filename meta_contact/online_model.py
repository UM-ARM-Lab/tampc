import numpy as np
import torch
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities.trajectory import invert_psd
from meta_contact import model
from meta_contact import prior


class OnlineDynamicsModel(object):
    """ Moving average estimate of locally linear dynamics from https://arxiv.org/pdf/1509.06841.pdf

    Note gamma here is (1-gamma) described in the paper, so high gamma forgets quicker.

    All dynamics public API takes input and returns output in original xu space,
    while internally (functions starting with underscore) they all operate in transformed space.
    """

    def __init__(self, gamma, online_prior: prior.OnlineDynamicsPrior, ds, N=1, sigreg=1e-5):
        self.gamma = gamma
        self.prior = online_prior
        self.sigreg = sigreg  # Covariance regularization (adds sigreg*eye(N))
        sigma, mu = prior.gaussian_params_from_dataset(ds)
        self.ds = ds
        self.advance = model.advance_state(ds.config, use_np=False)

        self.nx = ds.config.nx
        self.nu = ds.config.nu
        self.empsig_N = N
        self.emp_error = None
        self.prior_error = None

        self.prior_trust_coefficient = 0.1  # the lower it is the more we trust the prior; 0 means only ever use prior
        # TODO can track only mu and xxt, generate sigma when asking for dynamics?
        # Initial values
        self.mu = mu
        self.sigma = sigma
        self.xxt = sigma + np.outer(self.mu, self.mu)

    def evaluate_error(self, px, pu, cx, cu):
        """After updating dynamics and using that dynamics to plan an action,
        evaluate the error of empirical and prior dynamics"""
        # can't evaluate if it's the first action
        if px is None:
            self.emp_error = self.prior_error = None
            return
        xu, pxu, xux = _concatenate_state_control(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_params(self.nx, self.nu, xu, pxu, xux)
        # evaluate the accuracy of empirical and prior dynamics on (xux')
        Fe, fe, _ = conditioned_dynamics(self.nx, self.nu, self.sigma, self.mu, sigreg=self.sigreg)
        emp_x = evaluate_dynamics(px, pu, Fe, fe)
        # prior dynamics
        Fp, fp, _ = conditioned_dynamics(self.nx, self.nu, Phi / n0, mu0, sigreg=self.sigreg)
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
        # TODO make this private? It doesn't handle transforms
        # prior parameters
        xu, pxu, xux = _concatenate_state_control(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_params(self.nx, self.nu, xu, pxu, xux)

        # mix prior and empirical distribution
        sigma, mu = prior.mix_distributions(self.sigma, self.mu, self.empsig_N, Phi, mu0, m, n0)
        return conditioned_dynamics(self.nx, self.nu, sigma, mu, self.sigreg)

    def _get_batch_dynamics(self, px, pu, cx, cu):
        """
        Compute F, f - the linear dynamics where either dx or next_x = F*[curx, curu] + f
        The semantics depends on the data source the prior was trained on and that this was initialized on
        """
        # prior parameters
        xu = torch.cat((cx, cu), 1)
        pxu = torch.cat((px, pu), 1) if px is not None else None
        xux = torch.cat((px, pu, cx), 1) if px is not None else None
        Phi, mu0, m, n0 = self.prior.get_batch_params(self.nx, self.nu, xu, pxu, xux)

        # mix prior and empirical distribution
        sigma, mu = prior.batch_mix_distribution(torch.from_numpy(self.sigma), torch.from_numpy(self.mu), self.empsig_N,
                                                 Phi, mu0, m, n0)
        return _batch_conditioned_dynamics(self.nx, self.nu, sigma, mu, self.sigreg)

    def _apply_transform(self, x, u):
        xu = torch.cat((x, u), dim=1)
        xu = self.ds.preprocessor.transform_x(xu)
        x = xu[:, :self.nx]
        u = xu[:, self.nx:]
        return x, u

    def predict(self, px, pu, cx, cu, already_transformed=False):
        """
        Predict next state; will return with the same dimensions as cx
        :return: B x N x nx or N x nx next states
        """
        ocx = cx  # original state
        # transform if necessary (ensure dynamics is evaluated only in transformed space)
        if self.ds.preprocessor and not already_transformed:
            cx, cu = self._apply_transform(cx, cu)
            px, pu = self._apply_transform(px, pu)

        params = self._get_batch_dynamics(px, pu, cx, cu)
        y = _batch_evaluate_dynamics(cx, cu, *params)

        if self.ds.preprocessor:
            y = self.ds.preprocessor.invert_transform(y, ocx)

        next_state = self.advance(ocx, y)

        return next_state


def _concatenate_state_control(px, pu, cx, cu):
    # TODO use faster way of constructing xu and pxu (concatenate)
    xu = np.r_[cx, cu].astype(np.float32)
    pxu = np.r_[px, pu]
    xux = np.r_[px, pu, cx]
    return xu, pxu, xux


def conditioned_dynamics(nx, nu, sigma, mu, sigreg=1e-5):
    it = slice(nx + nu)
    ip = slice(nx + nu, nx + nu + nx)
    sigma[it, it] += sigreg * np.eye(nx + nu)
    sigma_inv = invert_psd(sigma[it, it])

    # Solve normal equations to get dynamics. (equation 2)
    Fm = sigma_inv.dot(sigma[it, ip]).T  # f_xu
    fv = mu[ip] - Fm.dot(mu[it])  # f_c
    dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)  # F
    dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)  # Guarantee symmetric

    return Fm, fv, dyn_covar


def _batch_conditioned_dynamics(nx, nu, sigma, mu, sigreg=1e-5):
    it = slice(nx + nu)
    ip = slice(nx + nu, nx + nu + nx)
    # guarantee symmetric positive definite with regularization
    sigma[:, it, it] += sigreg * torch.eye(nx + nu, dtype=sigma.dtype, device=sigma.device)
    u = torch.cholesky(sigma[:, it, it])
    # equivalent to inv * sigma
    # Solve normal equations to get dynamics. (equation 2)
    Fm = torch.cholesky_solve(sigma[:, it, ip], u).transpose(-1, -2)
    fv = mu[:, ip] - linalg.batch_batch_product(mu[:, it], Fm)

    # TODO calculate dyn_covar
    return Fm, fv, None


def evaluate_dynamics(x, u, F, f, cov=None):
    # TODO sample from multivariate normal if covariance is given
    xp = F @ np.concatenate((x, u)) + f
    return xp


def _batch_evaluate_dynamics(x, u, F, f, cov=None):
    xu = torch.cat((x, u), 1)
    xp = linalg.batch_batch_product(xu, F) + f
    return xp
