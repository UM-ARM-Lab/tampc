import abc
import copy
import logging
import os
import pickle

import numpy as np
import torch
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from meta_contact import cfg, model
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class InvariantTransform(LearnableParameterizedModel):
    def __init__(self, ds: datasource.DataSource, nz, too_far_for_neighbour=1., **kwargs):
        super().__init__(cfg.ROOT_DIR, **kwargs)

        self.ds = ds
        self.neighbourhood = None
        self.too_far_for_neighbour = too_far_for_neighbour
        self.nz = nz

    @abc.abstractmethod
    def __call__(self, state, action):
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

    def _is_in_neighbourhood(self, cur, candidate):
        return torch.norm(candidate - cur) < self.too_far_for_neighbour

    def calculate_neighbours(self):
        """
        Calculate information about the neighbour of each data point needed for training
        """
        # load and save this information since it's expensive to calculate
        name = "neighbour_info_{}_{}.pkl".format(self.ds.N, self.ds.config)
        fullname = os.path.join(cfg.DATA_DIR, name)
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                self.neighbourhood = pickle.load(f)
                logger.info("loaded neighbourhood info from %s", fullname)
        # train from samples of ds that are close in euclidean space
        XU, Y, _ = self.ds.training_set()
        X, U = torch.split(XU, self.ds.config.nx, dim=1)
        N = XU.shape[0]
        neighbourhood = []
        # can precalculate since XUY won't change during training and it's only dependent on these
        # assume training set is not shuffled, we can just look at adjacent datapoints sequentially
        # TODO remove this assumption by doing (expensive) pairwise-distance, or sampling?
        # for each datapoint we look at other datapoints close to it
        for i in range(N):
            # TODO for now just consider distance in X, later consider U and also Y?
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

        self.neighbourhood = neighbourhood
        # analysis for neighbourhood size; useful for tuning hyperparameters
        neighbourhood_size = [s.stop - s.start for s in neighbourhood]
        logger.info("min neighbourhood size %d max %d median %d median %f", np.min(neighbourhood_size),
                    np.max(neighbourhood_size),
                    np.median(neighbourhood_size), np.mean(neighbourhood_size))
        with open(fullname, 'wb') as f:
            pickle.dump(self.neighbourhood, f)
            logger.info("saved neighbourhood info to %s", fullname)

    def learn_model(self, max_epoch, batch_N=500):
        if self.neighbourhood is None:
            self.calculate_neighbours()

        writer = SummaryWriter(flush_secs=20, comment="{}_batch{}".format(self.name, batch_N))

        XU, Y, _ = self.ds.training_set()
        X, U = torch.split(XU, self.ds.config.nx, dim=1)
        N = XU.shape[0]

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer.zero_grad()
        batch_cov_loss = None
        batch_mse_loss = None
        for epoch in range(max_epoch):
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

                neighbours = self.neighbourhood[i]
                # reject neighbourhoods that are too small
                if neighbours.stop - neighbours.start < self.ds.config.ny + 1:  # dimension of latent space is 1
                    continue
                x, u = X[neighbours], U[neighbours]
                y = Y[neighbours]
                # TODO this is formulation 1 where y is directly linear in z; consider formulation 2
                z = self.__call__(x, u)

                # fit linear model to latent state
                p, cov = linalg.ls_cov(z, y)
                # covariance loss
                cov_loss = cov.trace()

                # mse loss
                yhat = z @ p.t()
                mse_loss = torch.norm(yhat - y, dim=1)

                batch_cov_loss.append(cov_loss.mean())
                batch_mse_loss.append(mse_loss.mean())

                self.step += 1

        self.save(last=True)


class NetworkInvariantTransform(InvariantTransform):
    def __init__(self, ds, nz, model_opts=None, **kwargs):
        if model_opts is None:
            model_opts = {}
        config = copy.deepcopy(ds.config)
        # output the latent space instead of y
        config.ny = nz
        self.user = model.DeterministicUser(make.make_sequential_network(config, **model_opts))
        super().__init__(ds, nz, **kwargs)

    def __call__(self, state, action):
        xu = torch.cat((state, action), dim=1)
        z = self.user.sample(xu)

        # TODO see if we need to formulate it as action * z for toy problem (less generalized, but easier, and nz=1)
        # z = action * z
        return z

    def parameters(self):
        return self.user.model.parameters()

    def _model_state_dict(self):
        return self.user.model.state_dict()

    def _load_model_state_dict(self, saved_state_dict):
        self.user.model.load_state_dict(saved_state_dict)
