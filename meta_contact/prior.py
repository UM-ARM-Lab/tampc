import abc
import logging

import numpy as np
import torch
from arm_pytorch_utilities import grad
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities.gmm import GMM
from meta_contact import model

logger = logging.getLogger(__name__)


def xux_from_dataset(ds):
    XU, Y, _ = ds.training_set()
    if ds.config.expanded_input:
        XU = XU[:, :ds.config.nx + ds.config.nu]
    # if the output is not in state space (such as if xu is transformed), then just treat them as XUY
    if not ds.config.y_in_x_space:
        XUX = torch.cat((XU, Y), dim=1)
    else:
        if ds.config.predict_difference:
            XUX = torch.cat((XU, XU[:, :ds.config.nx] + Y), dim=1)
        else:
            XUX = torch.cat((XU, Y), dim=1)
    return XUX


def gaussian_params_from_datasource(ds):
    """Take what we're given in terms of XU,Y, and let the dynamics model handle how that translates to next state"""
    XU, Y, _ = ds.training_set()
    data = torch.cat((XU, Y), dim=1)
    mu = data.mean(0)
    sigma = linalg.cov(data, rowvar=False)
    assert sigma.shape[0] == data.shape[1]
    return sigma.numpy(), mu.numpy()


class OnlineDynamicsPrior:
    """
    Dynamics model priors. May model dx = f(x,u) or x' = f(x,u) depending on data.
    Assumes all input are in transformed space and doesn't do any transformation itself.
    """

    @abc.abstractmethod
    def get_params(self, nx, nu, xu, pxu, xux):
        """Get normal inverse-Wishart prior parameters (Phi, mu0, m, n0) evaluated at this (pxu,xu)"""

    @abc.abstractmethod
    def get_batch_params(self, nx, nu, xu, pxu, xux):
        """Get normal inverse-Wishart prior parameters for batch of data (first dimension is batch)"""


def mix_prior(dX, dU, nnF, nnf, xu, sigma_x=None, strength=1.0, dyn_init_sig=None,
              use_least_squares=False, full_calculation=False):
    """
    Provide a covariance/bias term for mixing NN with least squares model.
    """
    it = slice(dX + dU)
    ny = nnF.shape[0]

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
                       np.c_[sigXK.T, np.zeros((ny, ny))]]  # Lower right square is unused
        nn_mu = None  # Unused

    return nn_Phi, nn_mu


def batch_mix_prior(nx, nu, nnF, strength=1.0):
    """
    Mix prior but with batch nnF and expect pytorch tensor instead of np
    """
    N, ny, nxnu = nnF.shape
    # \bar{Sigma}_xu,xu from section V.B, strength is alpha
    sigX = torch.eye(nxnu, dtype=nnF.dtype).repeat(N, 1, 1) * strength
    # lower left corner, nnF.T is df/dxu
    sigXK = sigX @ nnF.transpose(1, 2)
    # \bar{Sigma}, ignoring lower right
    top = torch.cat((sigX, sigXK), dim=2)
    bot = torch.cat((sigXK.transpose(1, 2), torch.zeros((N, ny, ny), dtype=nnF.dtype)), dim=2)
    nn_Phi = torch.cat((top, bot), dim=1)
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
            pass
        else:
            mw.learn_model(train_epochs, batch_N=batch_N)

        return NNPrior(mw, **kwargs)

    def __init__(self, mw: model.NetworkModelWrapper, mix_strength=1.0):
        self.dyn_net = mw
        self.dyn_net.freeze()
        self.mix_prior_strength = mix_strength
        self.full_context = mw.ds.config.expanded_input

    def get_params(self, nx, nu, xu, pxu, xux):
        # feed pxu and xu to network (full contextual network)
        full_input = np.concatenate((xu, pxu)) if self.full_context else xu
        full_input = torch.tensor(full_input, dtype=torch.double)
        # jacobian of xu' wrt xu and pxu, need to strip the pxu columns
        F = grad.jacobian(self._predict, full_input)
        # first columns are xu, latter columns are pxu
        F = F[:, :nx + nu].numpy()
        # TODO not sure if this should be + xu since we're predicting residual, but f is currently unused
        # TODO check correctness of linearization when our input isn't xu (transformed z)
        xp = self._predict(full_input.view(1, -1))
        xp = xp.view(-1).numpy()
        f = -F @ xu + xp
        # build \bar{Sigma} (nn_Phi) and \bar{mu} (nnf)
        nn_Phi, nnf = mix_prior(nx, nu, F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
        # NOTE nnf is not used
        mu0 = np.r_[xu, xp]
        # m and n0 are 1 (mix prior strength already scaled nn_Phi)
        return nn_Phi, mu0, 1, 1

    def _predict(self, *args):
        # should use private method because everything is already transformed
        return self.dyn_net._batch_apply_model(*args)

    def get_batch_params(self, nx, nu, xu, pxu, xux):
        # feed pxu and xu to network (full contextual network)
        full_input = torch.cat((xu, pxu), 1) if self.full_context else xu
        # jacobian of xu' wrt xu and pxu, need to strip the pxu columns
        F = grad.batch_jacobian(self._predict, full_input)
        # first columns are xu, latter columns are pxu
        F = F[:, :, :nx + nu]
        xp = self._predict(full_input)
        # build \bar{Sigma} (nn_Phi) and \bar{mu} (nnf)
        nn_Phi, nnf = batch_mix_prior(nx, nu, F, strength=self.mix_prior_strength)
        # NOTE nnf is not used
        mu0 = torch.cat((xu, xp), 1)
        # m and n0 are 1 (mix prior strength already scaled nn_Phi)
        return nn_Phi, mu0, 1, 1


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
        # TODO remove need for xux (just take what the ds gives)
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
            X: A N x T x nx matrix of sequential state data.
            U: A N x T x nu matrix of sequential control data.
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
        return Phi, mu0, m, n0

    def get_params(self, nx, nu, xu, pxu, xux):
        return self.eval(nx, nu, xux.reshape(1, nx + nu + nx))

    def get_batch_params(self, nx, nu, xu, pxu, xux):
        N = xu.shape[0]
        nxux = 2 * nx + nu
        Phi = torch.zeros((N, nxux, nxux), dtype=xu.dtype)
        mu0 = torch.zeros((N, nxux), dtype=xu.dtype)
        m = 1
        n0 = 1
        for i in range(N):
            Phi_i, mu0_i, m, n0 = self.get_params(nx, nu, xu[i].numpy(), pxu[i].numpy(), xux[i].numpy())
            Phi[i] = torch.from_numpy(Phi_i)
            mu0[i] = torch.from_numpy(mu0_i)

        return Phi, mu0, m, n0


class LSQPrior(OnlineDynamicsPrior):
    @classmethod
    def from_data(cls, ds, **kwargs):
        sigma, mu = gaussian_params_from_datasource(ds)
        prior = cls(sigma, mu, **kwargs)
        return prior

    def __init__(self, init_sigma, init_mu, mix_strength=1.0):
        self.dyn_init_sig = init_sigma
        self.dyn_init_mu = init_mu
        # m in equation 1
        self.mix_prior_strength = mix_strength

    def get_params(self, nx, nu, xu, pxu, xux):
        return self.dyn_init_sig, self.dyn_init_mu, self.mix_prior_strength, self.mix_prior_strength

    def get_batch_params(self, nx, nu, xu, pxu, xux):
        N = xu.shape[0]
        return torch.from_numpy(self.dyn_init_sig).repeat(N, 1, 1), torch.from_numpy(self.dyn_init_mu).repeat(N, 1), \
               self.mix_prior_strength, self.mix_prior_strength


def mix_distributions(emp_sigma, emp_mu, N, Phi, mu0, m, n0):
    """Mix empirical normal with normal inverse-Wishart prior, giving a normal distribution"""
    mu = (N * emp_mu + mu0 * m) / (N + m)
    sigma = (N * emp_sigma + Phi + ((N * m) / (N + m)) * np.outer(emp_mu - mu0, emp_mu - mu0)) / (N + n0)
    return sigma, mu


def batch_mix_distribution(emp_sigma, emp_mu, N, Phi, mu0, m, n0):
    """Mix current empirical normal with batch prior"""
    mu = (N * emp_mu + mu0 * m) / (N + m)
    d = emp_mu - mu0
    sigma = (N * emp_sigma + Phi + ((N * m) / (N + m)) * linalg.batch_outer_product(d, d)) / (N + n0)
    return sigma, mu
