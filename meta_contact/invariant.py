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
    def dx_to_zo(self, x, dx):
        """
        Transform differences in observation state to output latent state for training
        :param x:
        :param dx:
        :return:
        """

    @staticmethod
    @abc.abstractmethod
    def supports_only_direct_zi_to_dx():
        """
        Whether this transformation only supports direct linear dynamics from z to dx, or if it
        supports dynamics in z to dz, then a nonlinear transform from dz to dx.
        If false, then dz to dx and dx to dz should not be trivial.
        :return: True or False
        """

    def _record_metrics(self, writer, losses, suffix='', log=False):
        """
        Use summary writer and passed in losses to
        :param writer: SummaryWriter
        :return:
        """
        with torch.no_grad():
            for i, loss_name in enumerate(self._loss_names()):
                name = '{}{}'.format(loss_name, suffix)
                value = losses[i].mean().cpu().item()
                writer.add_scalar(name, value, self.step)
                if log:
                    logger.debug("metric %s %f", name, value)

    def _evaluate_metrics_on_whole_set(self, neighbourhood, tsf):
        # TODO do evaluation in original space to allow for comparison across transforms
        with torch.no_grad():
            data_set = self.ds.validation_set() if neighbourhood is self.neighbourhood_validation else \
                self.ds.training_set()
            X, U, Y = self._get_separate_data_columns(data_set)
            N = X.shape[0]

            batch_losses = self._init_batch_losses()

            for i in range(N):
                losses = self._evaluate_neighbour(X, U, Y, neighbourhood, i, tsf)
                if losses is None:
                    continue

                for i in range(len(batch_losses)):
                    batch_losses[i].append(losses[i].mean())

            batch_losses = [math_utils.replace_nan_and_inf(torch.tensor(losses)) for losses in batch_losses]
            return batch_losses

    def _evaluate_validation_set(self, writer):
        losses = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.LATENT_SPACE)
        self._record_metrics(writer, losses, suffix="/validation", log=True)

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
            neighbourhood_size = (neighbourhood > 0).sum(1)

        logger.info("min neighbourhood size %d max %d median %d", neighbourhood_size.min(),
                    neighbourhood_size.max(),
                    neighbourhood_size.median())
        return neighbourhood

    def _evaluate_neighbour(self, X, U, Y, neighbourhood, i, tsf=TransformToUse.LATENT_SPACE):
        neighbours, neighbour_weights, N = array_utils.extract_positive_weights(neighbourhood[i])

        if N < self.config.ny + self.nz:
            return None
        x, u = X[neighbours], U[neighbours]
        y = Y[neighbours]

        if tsf is TransformToUse.LATENT_SPACE:
            z = self.xu_to_zi(x, u)
            y = self.dx_to_zo(x, y)
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
        return mse_loss, cov_loss

    @staticmethod
    def _loss_names():
        return "mse_loss", "cov_loss"

    @staticmethod
    def _reduce_losses(losses):
        # use mse loss only
        return torch.sum(losses[0])

    def _init_batch_losses(self):
        return [[] for _ in self._loss_names()]

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def dx_to_zo(self, x, dx):
        raise RuntimeError("Learning linear dynamics instead of doing least squares")

    @abc.abstractmethod
    def linear_dynamics(self, zi):
        """
        Produce linear dynamics matrix A such that z_o = A * z_i
        :param zi: latent input space
        :return: (nzo x nzi) A
        """

    def _evaluate_neighbour(self, X, U, Y, neighbourhood, i, tsf=TransformToUse.LATENT_SPACE):
        neighbours, neighbour_weights, N = array_utils.extract_positive_weights(neighbourhood[i])

        if N < self.config.ny + self.nz:
            return None
        x, u = X[neighbours], U[neighbours]
        y = Y[neighbours]

        assert tsf is TransformToUse.LATENT_SPACE
        z = self.xu_to_zi(x, u)

        if N < self.ds.config.ny + z.shape[1]:
            return None

        # fit linear model to latent state
        # TODO consider using weights somehow
        A = self.linear_dynamics(z)

        zo = z @ A.t()
        yhat = self.zo_to_dx(x, zo)

        # mse loss
        mse_loss = torch.norm(yhat - y, dim=1)
        return mse_loss

    @staticmethod
    def _loss_names():
        return tuple("mse_loss", )

    def _evaluate_no_transform(self, writer):
        pass


class DirectLinearDynamicsTransform(InvariantTransform):
    """
    Assume dynamics is dynamics is directly linear from z to dx, that is we don't need transforms between
    dx and dz; for simpler dynamics this assumption should be good enough
    """

    @staticmethod
    def supports_only_direct_zi_to_dx():
        return True

    def zo_to_dx(self, x, z_o):
        return z_o

    def dx_to_zo(self, x, dx):
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

    def xu_to_zi(self, state, action):
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
        # if we're predicting z to dx then our y will not be in z space
        if self.tsf.supports_only_direct_zi_to_dx():
            config.y_in_x_space = False
        # if we're predicting z to dz then definitely will be predicting difference, otherwise don't change
        else:
            config.predict_difference = True

    def transform(self, XU, Y, labels=None):
        # these transforms potentially require x to transform y and back, so can't just use them separately
        X = XU[:, :self.tsf.config.nx]
        z_i = self.transform_x(XU)
        # no transformation needed our output is already zo
        z_o = Y if self.tsf.supports_only_direct_zi_to_dx() else self.tsf.dx_to_zo(X, Y)
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
        # no transformation needed our output is already dx
        if self.tsf.supports_only_direct_zi_to_dx():
            return Y
        return self.tsf.zo_to_dx(X, Y)

    def _fit_impl(self, XU, Y, labels):
        """Figure out what the transform outputs"""
        x, u = torch.split(XU, self.tsf.config.nx, dim=1)
        zi = self.tsf.xu_to_zi(x, u)
        self.model_input_dim = zi.shape[1]
        zo = self.tsf.dx_to_zo(x, Y)
        self.model_output_dim = zo.shape[1]
