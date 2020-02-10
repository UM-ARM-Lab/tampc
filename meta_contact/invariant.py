import abc
import copy
import logging
import os
import pickle
from abc import ABC
from typing import Union

import numpy as np
import torch
from arm_pytorch_utilities import array_utils
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import softknn
from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from meta_contact import cfg
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class TransformToUse:
    NO_TRANSFORM = 0
    REDUCE_TO_INPUT = 1
    LATENT_SPACE = 2


class InvariantTransform(LearnableParameterizedModel):
    def __init__(self, ds: datasource.DataSource, nz, nzo, too_far_for_neighbour=0.3,
                 train_on_continuous_data=False, **kwargs):
        super().__init__(cfg.ROOT_DIR, **kwargs)
        self.ds = ds
        # copy of config in case it gets modified later (such as by preprocessors)
        self.config = copy.deepcopy(ds.config)
        self.neighbourhood = None
        self.neighbourhood_validation = None
        self.too_far_for_neighbour = too_far_for_neighbour
        self.train_on_continuous_data = train_on_continuous_data
        # do not assume at this abstraction level the input and output latent space is the same
        self.nz = nz
        self.nzo = nzo
        # update name with parameteres
        self.name = '{}_{}_{}'.format(self.name, self.nz, self.config)

    @abc.abstractmethod
    def xu_to_zi(self, state, action):
        """
        Transform state and action down to underlying latent input of dynamics. h(x,u) = z_i
        This transform should be invariant to certain variations in state action, such as
        translation and rotation.

        For example in the planar pushing problem, dynamics can be described in the body frame
        of the object being pushed, so this function would be the coordinate transform from world frame to
        body frame.
        :param state: N x nx
        :param action: N x nu
        :return: z_i, N x nz input latent space
        """

    @abc.abstractmethod
    def zo_to_dx(self, x, z_o):
        """
        Reverse transform output latent space back to difference in observation space at a given observation state
        :param x: state at which to perform the h^{-1}(z_o)
        :param z_o: output latent state from learned dynamics z_o = bar{f}(z_i)
        :return:
        """

    @abc.abstractmethod
    def get_zo(self, x, dx, z_i):
        """
        Get output latent variable
        :param x:
        :param dx:
        :param z_i:
        :return:
        """

    def _record_metrics(self, writer, losses, suffix='', log=False):
        with torch.no_grad():
            for i, loss_name in enumerate(self.loss_names()):
                name = '{}{}'.format(loss_name, suffix)
                # allow some loss to be None (e.g. when not always used for every batch)
                if losses[i] is None:
                    continue
                value = losses[i].mean().cpu().item()
                writer.add_scalar(name, value, self.step)
                if log:
                    logger.debug("metric %s %f", name, value)

    def _evaluate_metrics_on_whole_set(self, neighbourhood, tsf):
        with torch.no_grad():
            data_set = self.ds.validation_set() if neighbourhood is self.neighbourhood_validation else \
                self.ds.training_set()
            X, U, Y = self._get_separate_data_columns(data_set)
            N = X.shape[0]

            batch_losses = self._init_batch_losses()

            for i in range(N):
                # TODO evaluate validation MSE in original space to allow for comparison across transforms
                # not easy in terms of infrastructure because we also need original neighbourhood x,y to convert back
                losses = self._evaluate_neighbour(X, U, Y, neighbourhood, i, tsf)
                if losses is None:
                    continue

                for i in range(len(batch_losses)):
                    batch_losses[i].append(losses[i].mean())

            batch_losses = [math_utils.replace_nan_and_inf(torch.tensor(losses)) for losses in batch_losses]
            return batch_losses

    def evaluate_validation(self, writer: Union[SummaryWriter, None]):
        """
        Evaluate losses on the validation set and recording them down if given a writer
        :param writer:
        :return: losses on the validation set
        """
        if self.neighbourhood is None:
            self.calculate_neighbours()
        losses = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.LATENT_SPACE)
        if writer is not None:
            self._record_metrics(writer, losses, suffix="/validation", log=True)
        return losses

    def _evaluate_no_transform(self, writer):
        losses = self._evaluate_metrics_on_whole_set(self.neighbourhood, TransformToUse.NO_TRANSFORM)
        self._record_metrics(writer, losses, suffix="_original", log=True)
        losses = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.NO_TRANSFORM)
        self._record_metrics(writer, losses, suffix="_original/validation", log=True)

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
            neighbourhood_size = (neighbourhood > 0).sum(1)

        logger.info("min neighbourhood size %d max %d median %d", neighbourhood_size.min(),
                    neighbourhood_size.max(),
                    neighbourhood_size.median())
        return neighbourhood

    def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
        if tsf is TransformToUse.LATENT_SPACE:
            z = self.xu_to_zi(X, U)
            Y = self.get_zo(X, Y, z)
        elif tsf is TransformToUse.REDUCE_TO_INPUT:
            z = U
        elif tsf is TransformToUse.NO_TRANSFORM:
            z = torch.cat((X, U), dim=1)
        else:
            raise RuntimeError("Unrecognized option for transform")

        if z.shape[0] < self.ds.config.ny + z.shape[1]:
            return None
        # fit linear model to latent state
        p, cov = linalg.ls_cov(z, Y, weights=weights)
        # covariance loss
        cov_loss = cov.trace()

        # mse loss
        yhat = z @ p.t()
        mse_loss = torch.norm(yhat - Y, dim=1)
        return mse_loss, cov_loss

    def _evaluate_neighbour(self, X, U, Y, neighbourhood, i, tsf=TransformToUse.LATENT_SPACE):
        neighbours, neighbour_weights, N = array_utils.extract_positive_weights(neighbourhood[i])

        if N < self.config.ny + self.nz:
            return None
        x, u = X[neighbours], U[neighbours]
        y = Y[neighbours]

        return self._evaluate_batch(x, u, y, weights=neighbour_weights, tsf=tsf)

    @staticmethod
    def loss_names():
        return "mse_loss", "cov_loss"

    def _reduce_losses(self, losses):
        # use mse loss only
        return torch.sum(losses[0])

    def _init_batch_losses(self):
        return [[] for _ in self.loss_names()]

    def learn_model(self, max_epoch, batch_N=500):
        if self.neighbourhood is None:
            self.calculate_neighbours()

        writer = SummaryWriter(flush_secs=20, comment="{}_batch{}".format(self.name, batch_N))

        X, U, Y = self._get_separate_data_columns(self.ds.training_set())
        N = X.shape[0]

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer.zero_grad()
        batch_losses = None
        self._evaluate_no_transform(writer)
        for epoch in range(max_epoch):
            logger.debug("Start epoch %d", epoch)
            # evaluate on validation at the start of epochs
            self.evaluate_validation(writer)
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()
            # randomize the order we're looking at the neighbourhoods
            neighbour_order = np.random.permutation(N)
            # TODO batch process neighbours
            for i in neighbour_order:
                bi = self.step % batch_N
                if bi == 0:
                    # treat previous batch
                    if batch_losses is not None and len(batch_losses[0]):
                        # turn lists into tensors
                        for j in range(len(batch_losses)):
                            batch_losses[j] = torch.stack(batch_losses[j])
                        # hold stats
                        reduced_loss = self._reduce_losses(batch_losses)
                        reduced_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self._record_metrics(writer, batch_losses)

                    batch_losses = self._init_batch_losses()

                self.step += 1

                losses = self._evaluate_neighbour(X, U, Y, self.neighbourhood, i)
                if losses is None:
                    continue

                # TODO consider if l.mean() generalizes to all kinds of losses
                for i in range(len(batch_losses)):
                    batch_losses[i].append(losses[i].mean())

        self.save(last=True)
        self._evaluate_no_transform(writer)


class LearnLinearDynamicsTransform(InvariantTransform):
    """
    Proposal #3 where we learn linear dynamics given zi instead of doing least squares
    """

    def __init__(self, *args, spread_loss_weight=1., **kwargs):
        self.spread_loss_weight = spread_loss_weight
        super(LearnLinearDynamicsTransform, self).__init__(*args, **kwargs)
        self.name = "{}_{}".format(self.name, self._loss_weight_name())

    def _loss_weight_name(self):
        return "spread_{}".format(self.spread_loss_weight)

    def get_zo(self, x, dx, z_i):
        A = self.linear_dynamics(z_i)
        z_o = linalg.batch_batch_product(z_i, A.transpose(-1, -2))
        return z_o

    @abc.abstractmethod
    def linear_dynamics(self, zi):
        """
        Produce linear dynamics matrix A such that z_o = A * z_i
        :param zi: latent input space
        :return: (nzo x nzi) A
        """

    def _evaluate_batch_apply_tsf(self, X, U, tsf):
        assert tsf is TransformToUse.LATENT_SPACE
        z = self.xu_to_zi(X, U)

        if z.shape[0] < self.ds.config.ny + z.shape[1]:
            return None

        # fit linear model to latent state
        A = self.linear_dynamics(z)
        zo = linalg.batch_batch_product(z, A.transpose(-1, -2))

        yhat = self.zo_to_dx(X, zo)
        return z, A, zo, yhat

    def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
        z, A, zo, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)

        # TODO consider using weights somehow (can use on dynamics dispersion cost)
        # add cost on difference of each A (each linear dynamics should be similar)
        dynamics_spread = torch.std(A, dim=0)
        # mse loss
        mse_loss = torch.norm(yhat - Y, dim=1)
        return mse_loss, dynamics_spread

    @staticmethod
    def loss_names():
        return "mse_loss", "spread_loss"

    def _reduce_losses(self, losses):
        return torch.sum(losses[0]) + self.spread_loss_weight * torch.sum(losses[1])

    def _evaluate_no_transform(self, writer):
        pass


class LearnFromBatchTransform(LearnLinearDynamicsTransform, ABC):
    """
    Instead of taking in predefined neighbourhoods, just learn over all batches
    """

    def _evaluate_metrics_on_whole_set(self, neighbourhood, tsf, translation=(0, 0)):
        with torch.no_grad():
            data_set = self.ds.validation_set() if neighbourhood is self.neighbourhood_validation else \
                self.ds.training_set()
            X, U, Y = self._get_separate_data_columns(data_set)

            # TODO this is hacky and specific to the block pushing env (translate state)
            X = torch.cat((X[:, :2] + torch.tensor(translation, device=X.device, dtype=X.dtype), X[:, 2:]), dim=1)

            batch_losses = self._evaluate_batch(X, U, Y, tsf=tsf)
            batch_losses = [math_utils.replace_nan_and_inf(losses) if losses is not None else None for losses in
                            batch_losses]
            return batch_losses

    def calculate_neighbours(self):
        # don't actually calculate any neighbourhoods
        self.neighbourhood = []
        self.neighbourhood_validation = []

    def _evaluate_neighbour(self, X, U, Y, neighbourhood, i, tsf=TransformToUse.LATENT_SPACE):
        raise RuntimeError("This class should only evaluate batches")

    def evaluate_validation(self, writer):
        # evaluate with translation
        losses = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.LATENT_SPACE)
        if writer is not None:
            self._record_metrics(writer, losses, suffix="/validation", log=True)
            for d in [4, 10]:
                for trans in [[1, 1], [-1, 1], [-1, -1]]:
                    dd = (trans[0] * d, trans[1] * d)
                    ls = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.LATENT_SPACE,
                                                             translation=dd)
                    self._record_metrics(writer, ls, suffix="/validation_{}_{}".format(dd[0], dd[1]))

        return losses

    def learn_model(self, max_epoch, batch_N=500):
        writer = SummaryWriter(flush_secs=20, comment="{}_batch{}".format(self.name, batch_N))

        ds_train = load_data.SimpleDataset(*self.ds.training_set())
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_N, shuffle=True)

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer.zero_grad()
        for epoch in range(max_epoch):
            logger.debug("Start epoch %d", epoch)
            # evaluate on validation at the start of epochs
            self.evaluate_validation(writer)
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()

            for i_batch, data in enumerate(train_loader):
                X, U, Y = self._get_separate_data_columns(data)
                losses = self._evaluate_batch(X, U, Y)
                if losses is None:
                    continue

                reduced_loss = self._reduce_losses(losses)
                reduced_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._record_metrics(writer, losses)
                self.step += 1

        self.save(last=True)


class LearnNeighbourhoodCovRegTransform(LearnFromBatchTransform, ABC):
    """
    Instead of taking in predefined neighbourhoods, simultaneously produce dynamics for batches
    and learn to structure z_i such that euclidean distance results in close dynamics
    using covariance loss regularization on local neighbourhoods.

    Doesn't do much it seems
    """

    def __init__(self, *args, expected_neighbourhood_size=12, **kwargs):
        self.nbrs = softknn.SoftKNN(min_k=expected_neighbourhood_size, normalization=1)
        # this calculation is just too slow so we're only going to randomly do it
        self.cov_loss_usage = 0.05
        LearnFromBatchTransform.__init__(self, *args, **kwargs)

    def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
        z, A, zo, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)

        cov_loss = None
        if np.random.rand() > (1 - self.cov_loss_usage):
            cov_loss = []
            # regularize by making sure nearest neighbours in z leads to good linear fits
            weights = self.nbrs(z)
            for i, w in enumerate(weights):
                neighbours, nw, N = array_utils.extract_positive_weights(w)

                nz = z[neighbours]
                nzo = zo[neighbours]
                _, sigma = linalg.ls_cov(nz, nzo, nw)
                cov_loss.append(sigma.trace())
            cov_loss = torch.stack(cov_loss)

        # mse loss
        mse_loss = torch.norm(yhat - Y, dim=1)
        return mse_loss, cov_loss

    def _reduce_losses(self, losses):
        loss = torch.sum(losses[0])
        if losses[1] is not None:
            loss += torch.sum(losses[1])
        return loss

    @staticmethod
    def loss_names():
        return "mse_loss", "cov_loss"


class InvariantTransformer(preprocess.Transformer):
    """
    Use an invariant transform to transform the data when needed, such that the dynamics model learned using
    the processed data source will be in the latent space.
    """

    def __init__(self, tsf: InvariantTransform):
        self.tsf = tsf
        self.tsf.freeze()
        self.model_input_dim = None
        self.model_output_dim = None
        super(InvariantTransformer, self).__init__()

    def update_data_config(self, config: load_data.DataConfig):
        if self.model_output_dim is None:
            raise RuntimeError("Fit the preprocessor for it to know what the proper output dim is")
        # this is not just tsf.nz because the tsf could have an additional structure such as z*u as output
        # TODO for our current transform we go from xu->z instead of x->z, u->v and we can treat this as nu = 0
        config.n_input = self.model_input_dim
        config.nx = self.model_input_dim
        config.nu = 0
        config.ny = self.model_output_dim  # either ny or nz
        # in general z_o and z_i are different spaces
        config.y_in_x_space = False
        # not sure if below is necessary
        # config.predict_difference = True

    def transform(self, XU, Y, labels=None):
        # these transforms potentially require x to transform y and back, so can't just use them separately
        X = XU[:, :self.tsf.config.nx]
        z_i = self.transform_x(XU)
        z_o = self.tsf.get_zo(X, Y, z_i)
        return z_i, z_o, labels

    def transform_x(self, XU):
        X = XU[:, :self.tsf.config.nx]
        U = XU[:, self.tsf.config.nx:]
        z_i = self.tsf.xu_to_zi(X, U)
        return z_i

    def transform_y(self, Y):
        raise RuntimeError("Should not attempt to transform Y directly; instead must be done with both X and Y")

    def invert_transform(self, Y, X=None):
        """Invert transformation on Y"""
        return self.tsf.zo_to_dx(X, Y)

    def _fit_impl(self, XU, Y, labels):
        """Figure out what the transform outputs"""
        self.model_input_dim = self.tsf.nz
        self.model_output_dim = self.tsf.nzo
