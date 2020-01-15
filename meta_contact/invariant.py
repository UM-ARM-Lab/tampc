import abc
import copy
import logging
import os
import pickle

import numpy as np
import torch
from arm_pytorch_utilities import array_utils
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from meta_contact import cfg, model
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class TransformToUse:
    NO_TRANSFORM = 0
    REDUCE_TO_INPUT = 1
    LATENT_SPACE = 2


class InvariantTransform(LearnableParameterizedModel):
    def __init__(self, ds: datasource.DataSource, nz, too_far_for_neighbour=0.3,
                 train_on_continuous_data=False, **kwargs):
        super().__init__(cfg.ROOT_DIR, **kwargs)
        self.ds = ds
        # copy of config in case it gets modified later (such as by preprocessors)
        self.config = copy.deepcopy(ds.config)
        self.neighbourhood = None
        self.neighbourhood_validation = None
        self.too_far_for_neighbour = too_far_for_neighbour
        self.train_on_continuous_data = train_on_continuous_data
        self.nz = nz
        # update name with parameteres
        self.name = '{}_{}_{}'.format(self.name, self.nz, self.config)

    @abc.abstractmethod
    def xu_to_z(self, state, action):
        """
        Transform state and action down to underlying latent input of dynamics.
        This transform should be invariant to certain variations in state action, such as
        translation and rotation.

        For example in the planar pushing problem, dynamics can be described in the body frame
        of the object being pushed, so this function would be the coordinate transform from world frame to
        body frame.
        :param state: N x nx
        :param action: N x nu
        :return: z, N x nz latent space
        """

    @abc.abstractmethod
    def dz_to_dx(self, x, dz):
        """
        Reverse transform from difference in latent space back to difference in observation space at a given
        observation state
        :param x:
        :param dz:
        :return:
        """

    @abc.abstractmethod
    def dx_to_dz(self, dx):
        """
        Transform differences in observation state to latent state for training
        :param dx:
        :return:
        """

    @staticmethod
    @abc.abstractmethod
    def supports_only_direct_z_to_dx():
        """
        Whether this transformation only supports direct linear dynamics from z to dx, or if it
        supports dynamics in z to dz, then a nonlinear transform from dz to dx.
        If false, then dz to dx and dx to dz should not be trivial.
        :return: True or False
        """

    def _record_metrics(self, writer, batch_mse_loss, batch_cov_loss):
        """
        Use summary writer and passed in losses to
        :param writer: SummaryWriter
        :param batch_mse_loss:
        :param batch_cov_loss:
        :return:
        """
        B = len(batch_cov_loss)
        writer.add_scalar('mse_loss', (sum(batch_mse_loss) / B).item(), self.step)
        writer.add_scalar('cov_loss', (sum(batch_cov_loss) / B).item(), self.step)

    def _evaluate_metrics_on_whole_set(self, neighbourhood, tsf):
        with torch.no_grad():
            data_set = self.ds.validation_set() if neighbourhood is self.neighbourhood_validation else \
                self.ds.training_set()
            X, U, Y = self._get_separate_data_columns(data_set)
            N = X.shape[0]
            cov_losses = []
            mse_losses = []
            for i in range(N):
                losses = self._evaluate_neighbour(X, U, Y, neighbourhood, i, tsf)
                if losses is None:
                    continue
                cov_loss, mse_loss = losses
                cov_losses.append(cov_loss.mean())
                mse_losses.append(mse_loss.mean())

            cov_losses = torch.tensor(cov_losses)
            mse_losses = torch.tensor(mse_losses)
            # filter out nan/inf
            math_utils.replace_nan_and_inf(cov_losses, 0)
            return [losses.mean().item() for losses in (cov_losses, mse_losses)]

    def _evaluate_validation_set(self, writer):
        cov_loss, mse_loss = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation,
                                                                 TransformToUse.LATENT_SPACE)
        writer.add_scalar('cov_loss/validation', cov_loss, self.step)
        writer.add_scalar('mse_loss/validation', mse_loss, self.step)

    def _evaluate_no_transform(self, writer):
        cov_loss, mse_loss = self._evaluate_metrics_on_whole_set(self.neighbourhood,
                                                                 TransformToUse.NO_TRANSFORM)
        writer.add_scalar('original_cov_loss', cov_loss, self.step)
        writer.add_scalar('original_mse_loss', mse_loss, self.step)
        cov_loss, mse_loss = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation,
                                                                 TransformToUse.NO_TRANSFORM)
        writer.add_scalar('original_cov_loss/validation', cov_loss, self.step)
        writer.add_scalar('original_mse_loss/validation', mse_loss, self.step)

    def _get_separate_data_columns(self, data_set):
        XU, Y, _ = data_set
        X, U = torch.split(XU, self.config.nx, dim=1)
        return X, U, Y

    def _is_in_neighbourhood(self, cur, candidate):
        return torch.norm(candidate - cur) < self.too_far_for_neighbour

    def calculate_neighbours(self):
        """
        Calculate information about the neighbour of each data point needed for training
        """
        # load and save this information since it's expensive to calculate
        name = "neighbour_info_{}_{}_continuous_{}_{}.pkl".format(self.ds.N, self.too_far_for_neighbour,
                                                                  int(self.train_on_continuous_data), self.config)
        fullname = os.path.join(cfg.DATA_DIR, name)
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                self.neighbourhood, self.neighbourhood_validation = pickle.load(f)
                logger.info("loaded neighbourhood info from %s", fullname)
                return

        self.neighbourhood = self._do_calculate_neighbourhood(*self.ds.training_set(),
                                                              consider_only_continuous=self.train_on_continuous_data)
        self.neighbourhood_validation = self._do_calculate_neighbourhood(*self.ds.validation_set())

        # analysis for neighbourhood size; useful for tuning hyperparameters
        with open(fullname, 'wb') as f:
            pickle.dump((self.neighbourhood, self.neighbourhood_validation), f)
            logger.info("saved neighbourhood info to %s", fullname)

    def _do_calculate_neighbourhood(self, XU, Y, labels, consider_only_continuous=False):
        # train from samples of ds that are close in euclidean space
        X, U = torch.split(XU, self.config.nx, dim=1)
        # can precalculate since XUY won't change during training and it's only dependent on these
        # TODO for now just consider distance in X, later consider U and also Y?
        if consider_only_continuous:
            # assume training set is not shuffled, we can just look at adjacent datapoints sequentially
            N = XU.shape[0]
            neighbourhood = []
            for i in range(N):
                # bounds on neighbourhood (assume continuous)
                li = i
                ri = i
                cur = X[i]
                while li > 0:
                    neighbour = X[li]
                    if not self._is_in_neighbourhood(cur, neighbour):
                        break
                    li -= 1
                    # could also update our cur - will result in chaining neighbours
                    # cur = next
                while ri < N:
                    neighbour = X[ri]
                    if not self._is_in_neighbourhood(cur, neighbour):
                        break
                    ri += 1
                neighbourhood.append(slice(li, ri))
            neighbourhood_size = torch.tensor([s.stop - s.start for s in neighbourhood], dtype=torch.double)
        else:
            # resort to calculating pairwise distance for all data points
            dists = torch.cdist(X, X)
            dd = -(dists - self.too_far_for_neighbour)

            # avoid edge case of multiple elements at kth closest distance causing them to become 0
            dd += 1e-10

            # make neighbours weighted on dist to data (to be used in weighted least squares)
            weights = dd.clamp(min=0)
            neighbourhood = weights
            neighbourhood_size = neighbourhood.sum(1)

        logger.info("min neighbourhood size %d max %d median %d median %f", neighbourhood_size.min(),
                    neighbourhood_size.max(),
                    neighbourhood_size.median(), neighbourhood_size.mean())
        return neighbourhood

    def _evaluate_neighbour(self, X, U, Y, neighbourhood, i, tsf=TransformToUse.LATENT_SPACE):
        neighbours, neighbour_weights, N = array_utils.extract_positive_weights(neighbourhood[i])

        if N < self.config.ny + self.nz:
            return None
        x, u = X[neighbours], U[neighbours]
        y = Y[neighbours]

        if tsf is TransformToUse.LATENT_SPACE:
            z = self.xu_to_z(x, u)
            # TODO can't actually train dx->dz together with xu->z? (can probably do dual gradient descent/alternate)
            y = self.dx_to_dz(y)
        elif tsf is TransformToUse.REDUCE_TO_INPUT:
            z = u
        elif tsf is TransformToUse.NO_TRANSFORM:
            z = torch.cat((x, u), dim=1)
        else:
            raise RuntimeError("Unrecognized option for transform")

        if N < self.ds.config.ny + z.shape[1]:
            return None
        # fit linear model to latent state
        p, cov = linalg.ls_cov(z, y, weights=neighbour_weights)
        # covariance loss
        cov_loss = cov.trace()

        # mse loss
        yhat = z @ p.t()
        mse_loss = torch.norm(yhat - y, dim=1)
        return cov_loss, mse_loss

    def learn_model(self, max_epoch, batch_N=500):
        if self.neighbourhood is None:
            self.calculate_neighbours()

        writer = SummaryWriter(flush_secs=20, comment="{}_batch{}".format(self.name, batch_N))

        X, U, Y = self._get_separate_data_columns(self.ds.training_set())
        N = X.shape[0]

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer.zero_grad()
        batch_cov_loss = None
        batch_mse_loss = None
        self._evaluate_no_transform(writer)
        for epoch in range(max_epoch):
            logger.debug("Start epoch %d", epoch)
            # evaluate on validation at the start of epochs
            self._evaluate_validation_set(writer)
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()
            # randomize the order we're looking at the neighbourhoods
            neighbour_order = np.random.permutation(N)
            # TODO batch process neighbours
            for i in neighbour_order:
                bi = self.step % batch_N
                if bi == 0:
                    # treat previous batch
                    if batch_cov_loss is not None and len(batch_cov_loss):
                        # hold stats
                        # reduced_loss = sum(batch_cov_loss)
                        reduced_loss = sum(batch_mse_loss)
                        reduced_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self._record_metrics(writer, batch_mse_loss, batch_cov_loss)

                    batch_cov_loss = []
                    batch_mse_loss = []

                self.step += 1

                losses = self._evaluate_neighbour(X, U, Y, self.neighbourhood, i)
                if losses is None:
                    continue

                cov_loss, mse_loss = losses
                batch_cov_loss.append(cov_loss.mean())
                batch_mse_loss.append(mse_loss.mean())

        self.save(last=True)
        self._evaluate_no_transform(writer)


class DirectLinearDynamicsTransform(InvariantTransform):
    """
    Assume dynamics is dynamics is directly linear from z to dx, that is we don't need transforms between
    dx and dz; for simpler dynamics this assumption should be good enough
    """

    @staticmethod
    def supports_only_direct_z_to_dx():
        return True

    def dz_to_dx(self, x, dz):
        return dz

    def dx_to_dz(self, dx):
        return dx


class NetworkInvariantTransform(DirectLinearDynamicsTransform):
    def __init__(self, ds, nz, model_opts=None, **kwargs):
        if model_opts is None:
            model_opts = {}
        config = copy.deepcopy(ds.config)
        # output the latent space instead of y
        config.ny = nz
        self.user = model.DeterministicUser(make.make_sequential_network(config, **model_opts))
        super().__init__(ds, nz, **kwargs)

    def xu_to_z(self, state, action):
        xu = torch.cat((state, action), dim=1)
        z = self.user.sample(xu)

        if self.nz is 1:
            z = z.view(-1, 1)
        # TODO see if we need to formulate it as action * z for toy problem (less generalized, but easier, and nz=1)
        # z = action * z
        return z

    def parameters(self):
        return self.user.model.parameters()

    def _model_state_dict(self):
        return self.user.model.state_dict()

    def _load_model_state_dict(self, saved_state_dict):
        self.user.model.load_state_dict(saved_state_dict)


class InvariantPreprocessor(preprocess.Preprocess):
    """
    Use an invariant transform to transform the data when needed, such that the dynamics model learned using
    the processed data source will be in the latent space.
    """

    def __init__(self, tsf: InvariantTransform, **kwargs):
        self.tsf = tsf
        self.tsf.freeze()
        self.model_input_dim = None
        self.model_output_dim = None
        super(InvariantPreprocessor, self).__init__(**kwargs)

    def update_data_config(self, config: load_data.DataConfig):
        if self.model_output_dim is None:
            raise RuntimeError("Fit the preprocessor for it to know what the proper output dim is")
        # enforce our transform's proclaimed support
        assert self.model_output_dim is config.ny if self.tsf.supports_only_direct_z_to_dx() else self.tsf.nz
        # this is not just tsf.nz because the tsf could have an additional structure such as z*u as output
        # TODO for our current transform we go from xu->z instead of x->z, u->v and we can treat this as nu = 0
        config.n_input = self.model_input_dim
        config.nx = self.model_input_dim
        config.nu = 0
        config.ny = self.model_output_dim  # either ny or nz
        # if we're predicting z to dx then our y will not be in z space
        if self.tsf.supports_only_direct_z_to_dx():
            config.y_in_x_space = False
        # if we're predicting z to dz then definitely will be predicting difference, otherwise don't change
        else:
            config.predict_difference = True

    def transform_x(self, XU):
        X = XU[:, :self.tsf.config.nx]
        U = XU[:, self.tsf.config.nx:]
        return self.tsf.xu_to_z(X, U)

    def transform_y(self, Y):
        # no transformation needed our output is already dx
        if self.tsf.supports_only_direct_z_to_dx():
            return Y
        return self.tsf.dx_to_dz(Y)

    def invert_transform(self, Y, X=None):
        """Invert transformation on Y"""
        # no transformation needed our output is already dx
        if self.tsf.supports_only_direct_z_to_dx():
            return Y
        return self.tsf.dz_to_dx(X, Y)

    def _fit_impl(self, XU, Y, labels):
        """Figure out what the transform outputs"""
        x, u = torch.split(XU[0].view(1, -1), self.tsf.config.nx, dim=1)
        z = self.tsf.xu_to_z(x, u)
        self.model_input_dim = z.shape[1]
        dz = self.tsf.dx_to_dz(Y[0].view(1, -1))
        self.model_output_dim = dz.shape[1]
