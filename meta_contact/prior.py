import torch
import logging
from arm_pytorch_utilities.gmm import GMM
import numpy as np
from arm_pytorch_utilities import linalg
from meta_contact import model
from arm_pytorch_utilities import grad

logger = logging.getLogger(__name__)


def xux_from_dataset(ds):
    XU, Y, _ = ds.training_set()
    if ds.config.expanded_input:
        XU = XU[:, :ds.config.nx + ds.config.nu]
    if ds.config.predict_difference:
        XUX = torch.cat((XU, XU[:, :ds.config.nx] + Y), dim=1)
    else:
        XUX = torch.cat((XU, Y), dim=1)
    return XUX


def gaussian_params_from_dataset(ds):
    XUX = xux_from_dataset(ds)
    mu = XUX.mean(0)
    sigma = linalg.cov(XUX, rowvar=False)
    assert sigma.shape[0] == XUX.shape[1]
    return sigma.numpy(), mu.numpy()


class OnlineDynamicsPrior:
    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        """Mix global and local dynamics returning MAP covariance and mean (local gaussian dynamics)"""
        return empsig, mun


def mix_prior(dX, dU, nnF, nnf, xu, sigma_x=None, strength=1.0, dyn_init_sig=None,
              use_least_squares=False, full_calculation=False):
    """
    Provide a covariance/bias term for mixing NN with least squares model.
    """
    it = slice(dX + dU)

    if use_least_squares:
        sigX = dyn_init_sig[it, it]
    else:
        # \bar{Sigma}_xu,xu from section V.B, strength is alpha
        sigX = np.eye(dX + dU) * strength

    # lower left corner, nnF.T is df/dxu
    sigXK = sigX.dot(nnF.T)
    if full_calculation:
        nn_Phi = np.r_[np.c_[sigX, sigXK],
                       np.c_[sigXK.T, nnF.dot(sigX).dot(nnF.T) + sigma_x]]
        # nnf is f(xu)
        nn_mu = np.r_[xu, nnF.dot(xu) + nnf]
    else:
        # \bar{Sigma}, ignoring lower right
        nn_Phi = np.r_[np.c_[sigX, sigXK],
                       np.c_[sigXK.T, np.zeros((dX, dX))]]  # Lower right square is unused
        nn_mu = None  # Unused

    return nn_Phi, nn_mu


class NNPrior(OnlineDynamicsPrior):
    @classmethod
    def from_data(cls, mw: model.NetworkModelWrapper, checkpoint=None, train_epochs=50, batch_N=500, **kwargs):
        # ensure that we're predicting residuals
        # if not mw.dataset.config.predict_difference:
        #     raise RuntimeError("Network must be predicting residuals")
        # create (pytorch) network, load from checkpoint if given, otherwise train for some iterations
        if checkpoint and mw.load(checkpoint):
            logger.info("loaded checkpoint %s", checkpoint)
        else:
            mw.learn_model(train_epochs, batch_N=batch_N)

        return NNPrior(mw, **kwargs)

    def __init__(self, mw: model.NetworkModelWrapper, mix_strength=1.0):
        self.dyn_net = mw
        self.dyn_net.freeze()
        self.mix_prior_strength = mix_strength
        self.full_context = mw.dataset.config.expanded_input

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        # feed pxu and xu to network (full contextual network)
        full_input = np.concatenate((xu, pxu)) if self.full_context else xu
        full_input = torch.tensor(full_input, dtype=torch.double)
        # jacobian of xu' wrt xu and pxu, need to strip the pxu columns
        F = grad.jacobian(self.dyn_net.predict, full_input)
        # first columns are xu, latter columns are pxu
        F = F[:, :dX + dU].numpy()
        # TODO not sure if this should be + xu since we're predicting residual, but f is currently unused
        xp = self.dyn_net.predict(full_input.view(1, -1))
        xp = xp.view(-1).numpy()
        f = -F @ xu + xp
        # build \bar{Sigma} (nn_Phi) and \bar{mu} (nnf)
        nn_Phi, nnf = mix_prior(dX, dU, F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
        sigma = (N * empsig + nn_Phi) / (N + 1)
        mun = (N * mun + np.r_[xu, xp]) / (N + 1)
        return sigma, mun


class GMMPrior(OnlineDynamicsPrior):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """

    @classmethod
    def from_data(cls, ds, **kwargs):
        XUX = xux_from_dataset(ds)
        prior = cls(**kwargs)
        prior.update_batch(XUX.numpy())
        return prior

    def __init__(self, min_samples_per_cluster=20, max_clusters=50, max_samples=20, strength=1.0):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum samples per cluster.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for
                fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        self.X = None
        self.U = None
        self.gmm = GMM()
        self._min_samp = min_samples_per_cluster
        self._max_samples = max_samples
        self._max_clusters = max_clusters
        self._strength = strength

    def initial_state(self):
        """ Return dynamics prior for initial time step. """
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update_batch(self, XUX):
        self.gmm.update(XUX, self._max_clusters, max_iterations=15)

    def update(self, X, U):
        """
        Update prior with additional data.
        Args:
            X: A N x T x dX matrix of sequential state data.
            U: A N x T x dU matrix of sequential control data.
        """
        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  # TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T + 1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))
        logger.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        self.gmm.update(xux, K)

    def eval(self, Dx, Du, pts):
        """
        Evaluate prior.
        Args:
            pts: A N x Dx+Du+Dx matrix.
        """
        # Construct query data point by rearranging entries and adding
        # in reference.
        assert pts.shape[1] == Dx + Du + Dx

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm.inference(pts)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        mu0, Phi, m, n0 = self.eval(dX, dU, xux.reshape(1, dX + dU + dX))
        m = m
        mun = (N * mun + mu0 * m) / (N + m)  # Use bias
        sigma = (N * empsig + Phi + ((N * m) / (N + m)) * np.outer(mun - mu0, mun - mu0)) / (N + n0)
        return sigma, mun


class LSQPrior(OnlineDynamicsPrior):
    @classmethod
    def from_data(cls, ds, **kwargs):
        sigma, mu = gaussian_params_from_dataset(ds)
        prior = cls(sigma, mu, **kwargs)
        return prior

    def __init__(self, init_sigma, init_mu, mix_strength=1.0):
        self.dyn_init_sig = init_sigma
        self.dyn_init_mu = init_mu
        # m in equation 1
        self.mix_prior_strength = mix_strength

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        # equation 1
        mu0, Phi = (self.dyn_init_mu, self.dyn_init_sig)
        mun = (N * mun + mu0 * self.mix_prior_strength) / (N + self.mix_prior_strength)
        sigma = (N * empsig + self.mix_prior_strength * Phi) / (
                N + self.mix_prior_strength)  # + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
        return sigma, mun
