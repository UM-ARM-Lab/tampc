import torch
import logging
from arm_pytorch_utilities.gmm import GMM
import numpy as np
from arm_pytorch_utilities import linalg
from meta_contact import model
from arm_pytorch_utilities import grad

logger = logging.getLogger(__name__)


def xux_from_dataset(ds, nx=5):
    XU, Y, _ = ds.training_set()
    if not ds.config.predict_difference:
        XUX = torch.cat((XU, Y), dim=1)
    else:
        XUX = torch.cat((XU[:-1], XU[1:, :nx]), dim=1)
    return XUX


def gaussian_params_from_dataset(ds):
    XUX = xux_from_dataset(ds)
    mu = XUX.mean(0)
    sigma = linalg.cov(XUX)
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
    def from_data(cls, mw: model.NetworkModelWrapper, checkpoint=None, train_epochs=50, **kwargs):
        # ensure that we're predicting residuals
        if not mw.dataset.config.predict_difference:
            raise RuntimeError("Network must be predicting residuals")
        # create (pytorch) network, load from checkpoint if given, otherwise train for some iterations
        if checkpoint and mw.load(checkpoint):
            logger.info("loaded checkpoint %s", checkpoint)
        else:
            mw.learn_model(train_epochs)

        return NNPrior(mw, **kwargs)

    def __init__(self, mw: model.NetworkModelWrapper, full_context=False, mix_strength=1.0):
        # TODO ensure that the data matches the full context flag
        self.full_context = full_context
        self.dyn_net = mw
        self.dyn_net.freeze()
        self.mix_prior_strength = mix_strength

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        # feed pxu and xu to network (full contextual network)
        full_input = torch.tensor(np.concatenate((xu, pxu), axis=1))
        F = grad.jacobian(self.dyn_net.model, full_input)
        net_fwd = self.dyn_net.model(full_input)
        f = -F.dot(xu) + net_fwd
        # build \bar{Sigma} (nn_Phi) and \bar{mu} (nnf)
        nn_Phi, nnf = mix_prior(dX, dU, F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
        sigma = (N * empsig + nn_Phi) / (N + 1)
        mun = (N * mun + np.r_[xu, F.dot(xu) + f]) / (N + 1)
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


class LinearPrior:
    def __init__(self, ds):
        self.dataset = ds
        XU, Y, _ = ds.training_set()
        n = XU.shape[1]
        self.nu = 2
        self.nx = n - self.nu
        # get dynamics
        params, res, rank, _ = np.linalg.lstsq(XU.numpy(), Y.numpy())
        # our call is setup to handle residual dynamics, so need to make sure that's the case
        if not ds.config.predict_difference:
            raise RuntimeError("Dynamics is set up to only handle residual dynamics")
        # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
        self.A = np.zeros((self.nx, self.nx))
        self.A[2:, :] += params[:self.nx, :].T
        self.B = np.zeros((self.nx, self.nu))
        self.B[0, 0] = 1
        self.B[1, 1] = 1
        self.B[2:, :] += params[self.nx:, :].T

    def __call__(self, x, u):
        xu = np.concatenate((x, u))

        if self.dataset.preprocessor:
            xu = self.dataset.preprocessor.transform_x(xu.reshape(1, -1)).numpy().reshape(-1)

        dxb = self.A @ xu[:self.nx] + self.B @ xu[self.nx:]
        dxb = dxb[self.nu:]

        if self.dataset.preprocessor:
            dxb = self.dataset.preprocessor.invert_transform(dxb.reshape(1, -1)).reshape(-1)

        if torch.is_tensor(dxb):
            dxb = dxb.numpy()
        # dxb = self.model(xu)
        # directly move the pusher
        x[:2] += u
        x[2:] += dxb
        return x


class LinearPriorTorch(LinearPrior):
    def __init__(self, ds):
        super().__init__(ds)
        self.A = torch.from_numpy(self.A)
        self.B = torch.from_numpy(self.B)

    def __call__(self, x, u):
        xu = torch.cat((x, u), dim=1)

        if self.dataset.preprocessor:
            xu = self.dataset.preprocessor.transform_x(xu)

        dxb = xu[:, :self.nx] @ self.A.transpose(0, 1) + xu[:, self.nx:] @ self.B.transpose(0, 1)
        # dxb = self.A @ xu[:, :self.nx] + self.B @ xu[:, self.nx:]
        # strip x,y of the pusher, which we add directly;
        dxb = dxb[:, self.nu:]

        if self.dataset.preprocessor:
            dxb = self.dataset.preprocessor.invert_transform(dxb)

        # directly move the pusher
        x[:, :2] += u
        x[:, 2:] += dxb
        return x
