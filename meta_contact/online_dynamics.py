import numpy as np
from arm_pytorch_utilities.trajectory import invert_psd


class OnlineDynamics(object):
    """ Moving average estimate of locally linear dynamics from https://arxiv.org/pdf/1509.06841.pdf

    Note gamma here is (1-gamma) described in the paper, so high gamma forgets quicker
    """

    def __init__(self, gamma, prior, init_mu, init_sigma, dX, dU, N=1, sigreg=1e-5):
        self.gamma = gamma
        self.prior = prior
        self.sigreg = sigreg  # Covariance regularization (adds sigreg*eye(N))
        self.dX = dX
        self.dU = dU
        self.empsig_N = N

        # TODO can track only mu and xxt, generate sigma when asking for dynamics?
        # Initial values
        self.mu = init_mu
        self.sigma = init_sigma
        self.xxt = init_sigma + np.outer(self.mu, self.mu)

    def update(self, prevx, prevu, curx):
        """ Perform a moving average update on the current dynamics """
        xux = np.r_[prevx, prevu, curx].astype(np.float32)

        gamma = self.gamma
        # Do a moving average update (equations 3,4)
        self.mu = self.mu * (1 - gamma) + xux * (gamma)
        self.xxt = self.xxt * (1 - gamma) + np.outer(xux, xux) * (gamma)
        self.xxt = 0.5 * (self.xxt + self.xxt.T)
        self.sigma = self.xxt - np.outer(self.mu, self.mu)

    def get_dynamics(self, t, prevx, prevu, curx, curu):
        """
        Compute F, f - the linear dynamics where next_x = F*[curx, curu] + f
        """
        dX = self.dX
        dU = self.dU

        it = slice(dX + dU)
        ip = slice(dX + dU, dX + dU + dX)

        N = self.empsig_N

        xu = np.r_[curx, curu].astype(np.float32)
        pxu = np.r_[prevx, prevu]
        xux = np.r_[prevx, prevu, curx]

        # Mix and add regularization (equation 1)
        sigma, mun = self.prior.mix(dX, dU, xu, pxu, xux, self.sigma, self.mu, N)
        sigma[it, it] = sigma[it, it] + self.sigreg * np.eye(dX + dU)
        sigma_inv = invert_psd(sigma[it, it])

        # Solve normal equations to get dynamics. (equation 2)
        Fm = sigma_inv.dot(sigma[it, ip]).T  # f_xu
        fv = mun[ip] - Fm.dot(mun[it])  # f_c
        dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)  # F
        dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)  # Guarantee symmetric

        return Fm, fv, dyn_covar
