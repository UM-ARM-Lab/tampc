import abc
import logging
import os

import numpy as np
import torch
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from meta_contact import cfg
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class ModelUser(abc.ABC):
    """Ways of computing loss and sampling from a model; interface to NetworkModelWrapper"""

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def compute_loss(self, XU, Y):
        """Compute the training loss on this batch"""

    @abc.abstractmethod
    def sample(self, xu):
        """Sample y from inputs xu"""


class MDNUser(ModelUser):
    def compute_loss(self, XU, Y):
        pi, normal = self.model(XU)
        # compute losses
        # negative log likelihood
        nll = MixtureDensityNetwork.loss(pi, normal, Y)
        return nll

    def sample(self, xu):
        pi, normal = self.model(xu)
        y = MixtureDensityNetwork.sample(pi, normal)
        return y


class DeterministicUser(ModelUser):
    def compute_loss(self, XU, Y):
        Yhat = self.model(XU)
        E = (Y - Yhat).norm(2, dim=1)
        return E

    def sample(self, xu):
        y = self.model(xu)
        return y


def advance_state(config: load_data.DataConfig, use_np=True):
    if use_np:
        def cat(seq):
            return np.column_stack(seq)
    else:
        def cat(seq):
            return torch.cat(seq, dim=1)

    if config.predict_difference:
        if config.predict_all_dims:
            def advance(xu, dx):
                x = xu[:, :config.nx]
                return x + dx
        else:
            # directly move the pusher
            def advance(xu, dxb):
                x = xu[:, :config.nx]
                u = xu[:, config.nx:config.nx + config.nu]
                return x + cat((u, dxb))
    else:
        if config.predict_all_dims:
            def advance(xu, xup):
                return xup
        else:
            def advance(xu, xb):
                # TODO not general; specific to the case where the first nu states are controlled directly
                u = xu[:, config.nx:config.nx + config.nu]
                return cat((xu[:, :config.nu] + u, xb))

    return advance


def linear_model_from_ds(ds):
    XU, Y, _ = ds.training_set()
    XU = XU.numpy()
    Y = Y.numpy()
    # get dynamics
    params, res, rank, _ = np.linalg.lstsq(XU, Y)
    if ds.config.predict_difference:
        # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
        state_offset = 0 if ds.config.predict_all_dims else ds.config.nu
        # A = np.eye(ds.config.nx)
        A = np.zeros((ds.config.nx, ds.config.nx))
        B = np.zeros((ds.config.nx, ds.config.nu))
        A[state_offset:, :] += params[:ds.config.nx, :].T
        if not ds.config.predict_all_dims:
            B[0, 0] = 1
            B[1, 1] = 1
        B[state_offset:, :] += params[ds.config.nx:, :].T
    else:
        if ds.config.predict_all_dims:
            # predict dynamics rather than difference
            A = params[:ds.config.nx, :].T
            B = params[ds.config.nx:, :].T
        else:
            A = np.eye(ds.config.nx)
            B = np.zeros((ds.config.nx, ds.config.nu))
            A[ds.config.nu:, :] = params[:ds.config.nx, :].T
            B[0, 0] = 1
            B[1, 1] = 1
            B[ds.config.nu:, :] = params[ds.config.nx:, :].T
    return A, B


class DynamicsModel(abc.ABC):
    def __init__(self, dataset, use_np=False):
        self.dataset = dataset
        self.advance = advance_state(dataset.config, use_np=use_np)

    def predict(self, xu):
        """
        Predict next state; will return with the same dimensions as xu
        :param xu: B x N x (nx + nu) or N x (nx + nu) full input (if missing B will add it)
        :return: B x N x nx or N x nx next states
        """
        orig_shape = xu.shape
        if len(orig_shape) > 2:
            # reduce all batch dimensions down to the first one
            xu = xu.view(-1, orig_shape[-1])

        if self.dataset.preprocessor:
            dxb = self._apply_model(self.dataset.preprocessor.transform_x(xu))
            dxb = self.dataset.preprocessor.invert_transform(dxb)
        else:
            dxb = self._apply_model(xu)

        x = self.advance(xu, dxb)
        if len(orig_shape) > 2:
            # restore original batch dimensions; keep variable dimension (nx)
            x = x.view(*orig_shape[:-1], -1)
        return x

    @abc.abstractmethod
    def _apply_model(self, xu):
        """
        Apply model to input
        :param xu: N x (nx+nu) State and input
        :return: N x nx Next states or state residuals
        """

    def __call__(self, x, u):
        """
        Wrapper for predict when x and u are given separately
        """
        xu = torch.cat((x, u), dim=1)
        return self.predict(xu)


class NetworkModelWrapper(LearnableParameterizedModel, DynamicsModel):
    def __init__(self, model_user: ModelUser, dataset, lr=1e-3, regularization=1e-5, **kwargs):
        DynamicsModel.__init__(self, dataset)
        LearnableParameterizedModel.__init__(self, cfg.ROOT_DIR, **kwargs)
        self.user = model_user
        self.model = model_user.model

        self.writer = SummaryWriter(flush_secs=20, comment=os.path.basename(self.name))

    def _accumulate_stats(self, loss, vloss):
        self.writer.add_scalar('loss/training', loss, self.step)
        self.writer.add_scalar('loss/validation', vloss, self.step)

    def learn_model(self, max_epoch, batch_N=500):
        self.XU, self.Y, self.labels = self.dataset.training_set()
        self.XUv, self.Yv, self.labelsv = self.dataset.validation_set()
        ds_train = load_data.SimpleDataset(self.XU, self.Y, self.labels)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_N, shuffle=True)

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()

            for i_batch, data in enumerate(train_loader):
                self.step += 1

                XU, Y, contacts = data

                self.optimizer.zero_grad()
                loss = self.user.compute_loss(XU, Y)

                # validation and other analysis
                with torch.no_grad():
                    vloss = self.user.compute_loss(self.XUv, self.Yv)
                    self._accumulate_stats(loss.mean(), vloss.mean())

                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

                logger.debug("Epoch %d loss %f vloss %f", epoch, loss.mean().item(), vloss.mean().item())
        # save after training
        self.save(last=True)

        # compare prediction accuracy against least squares
        XU, Y = self.XU.numpy(), self.Y.numpy()
        params, res, rank, _ = np.linalg.lstsq(XU, Y)
        XU, Y = self.XUv.numpy(), self.Yv.numpy()
        Yhat = XU @ params
        E = np.linalg.norm((Yhat - Y), axis=1)
        Yhatn = self.user.sample(self.XUv).detach().numpy()
        En = np.linalg.norm((Yhatn - Y), axis=1)
        logger.info("Least squares error %f network error %f", E.mean(), En.mean())

    def _model_state_dict(self):
        return self.model.state_dict()

    def _load_model_state_dict(self, saved_state_dict):
        self.model.load_state_dict(saved_state_dict)

    def parameters(self):
        return self.model.parameters()

    def _apply_model(self, xu):
        return self.user.sample(xu)


class LinearModelTorch(DynamicsModel):
    def __init__(self, ds):
        super().__init__(ds)

        self.nu = ds.config.nu
        self.nx = ds.config.nx
        # get dynamics

        self.A, self.B = linear_model_from_ds(ds)

        self.A = torch.from_numpy(self.A)
        self.B = torch.from_numpy(self.B)

    def _apply_model(self, xu):
        dxb = xu[:, :self.nx] @ self.A.transpose(0, 1) + xu[:, self.nx:] @ self.B.transpose(0, 1)
        # dxb = self.A @ xu[:, :self.nx] + self.B @ xu[:, self.nx:]
        # strip x,y of the pusher, which we add directly;
        if not self.dataset.config.predict_all_dims:
            dxb = dxb[:, self.nu:]
        return dxb
