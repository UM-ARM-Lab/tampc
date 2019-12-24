import os
import torch
import logging
from tensorboardX import SummaryWriter
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.optim import Lookahead
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from arm_pytorch_utilities.gmm import GMM
import numpy as np
from meta_contact import cfg
from arm_pytorch_utilities import linalg

logger = logging.getLogger(__name__)


def xux_from_dataset(ds):
    XU, _, _ = ds.training_set()
    XUX = torch.cat((XU[:-1], XU[1:, :5]), dim=1)
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


# TODO NNPrior


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
        # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
        # self.A = np.diag([1., 1., 1., 1., 1.])
        self.A = np.zeros((5, 5))
        self.B = np.zeros((self.nx, self.nu))
        self.A[2:, :] += params[:self.nx, :].T
        # self.B[0, 0] = 1
        # self.B[1, 1] = 1
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


class Prior:
    def __init__(self, model, name, dataset, lr, regularization, lookahead=True):
        self.dataset = dataset
        self.optimizer = None
        self.step = 0
        self.name = name
        # create model architecture
        self.dataset.make_data()
        self.XU, self.Y, self.labels = self.dataset.training_set()
        self.XUv, self.Yv, self.labelsv = self.dataset.validation_set()
        self.model = model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=regularization)
        if lookahead:
            self.optimizer = Lookahead(self.optimizer)

        self.writer = SummaryWriter(flush_secs=20, comment=os.path.basename(name))

    # def _compute_loss(self, XU, Y):
    #     Yhat = self.model(XU)
    #     E = (Y - Yhat).norm(2, dim=1) ** 2
    #     return E

    def _compute_loss(self, XU, Y):
        pi, normal = self.model(XU)
        # compute losses
        # negative log likelihood
        nll = MixtureDensityNetwork.loss(pi, normal, Y)
        return nll

    def _accumulate_stats(self, loss, vloss):
        self.writer.add_scalar('accuracy_loss/training', loss, self.step)
        self.writer.add_scalar('accuracy_loss/validation', vloss, self.step)

    def learn_model(self, max_epoch, batch_N=500):
        ds_train = load_data.SimpleDataset(self.XU, self.Y, self.labels)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_N, shuffle=True)
        self.step = 0

        save_checkpoint_every_n_epochs = max_epoch // 20

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()

            for i_batch, data in enumerate(train_loader):
                self.step += 1

                XU, Y, contacts = data

                self.optimizer.zero_grad()
                accuracy_loss = self._compute_loss(XU, Y)

                # validation and other analysis
                with torch.no_grad():
                    vloss = self._compute_loss(self.XUv, self.Yv)
                    self._accumulate_stats(accuracy_loss.mean(), vloss.mean())

                accuracy_loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

                logger.info("Epoch %d acc loss %f", epoch, accuracy_loss.mean().item())
        # save after training
        self.save()

    def save(self):
        state = {
            'step': self.step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        base_dir = os.path.join(cfg.ROOT_DIR, 'checkpoints')
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        full_name = os.path.join(base_dir, '{}.{}.tar'.format(self.name, self.step))
        torch.save(state, full_name)
        logger.info("saved checkpoint %s", full_name)

    def load(self, filename):
        if not os.path.isfile(filename):
            return False
        checkpoint = torch.load(filename)
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return True

    def __call__(self, x, u):
        xu = torch.tensor(np.concatenate((x, u))).reshape(1, -1)

        if self.dataset.preprocessor:
            xu = self.dataset.preprocessor.transform_x(xu)

        pi, normal = self.model(xu)
        dxb = MixtureDensityNetwork.sample(pi, normal)

        if self.dataset.preprocessor:
            dxb = self.dataset.preprocessor.invert_transform(dxb).reshape(-1)

        if torch.is_tensor(dxb):
            dxb = dxb.numpy()
        # dxb = self.model(xu)
        # directly move the pusher
        x[:2] += u
        x[2:] += dxb
        return x
