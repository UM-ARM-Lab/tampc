import abc
import logging
import os

import numpy as np
import torch
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from arm_pytorch_utilities.make_data import datasource
from tampc import cfg
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class ModelUser(abc.ABC):
    """Ways of computing loss and sampling from a model; interface to NetworkModelWrapper"""

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def compute_validation_loss(self, XUv, Yv, ds: datasource.FileDataSource):
        """Compute the validation loss in not-preprocessed space (allows for comparison across preprocessors)"""

    @abc.abstractmethod
    def compute_loss(self, XU, Y):
        """Compute the training loss on this batch"""

    @abc.abstractmethod
    def sample(self, xu):
        """Sample y from inputs xu"""


class MDNUser(ModelUser):
    def compute_validation_loss(self, XUv, Yv, ds):
        return self.compute_loss(XUv, Yv)

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
    def compute_validation_loss(self, XUv, Yv, ds):
        if not ds.preprocessor:
            return self.compute_loss(XUv, Yv)
        Yhat = self.sample(XUv)
        XUv_orig, Yv_orig, _ = ds.original_validation_set()
        # compare in original space
        Xv_orig = XUv_orig[:, :ds.original_config().nx]
        Yhat_orig = ds.preprocessor.invert_transform(Yhat, Xv_orig)
        return self._compute_error(Yv_orig, Yhat_orig)

    def _compute_error(self, Y, Yhat):
        E = (Y - Yhat).norm(2, dim=1)
        return E

    def compute_loss(self, XU, Y):
        return self._compute_error(Y, self.model(XU))

    def sample(self, xu):
        y = self.model(xu)
        return y


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


class DynamicsBase(abc.ABC):
    def __init__(self, ds):
        self.ds = ds
        config = ds.original_config()
        self.orig_config = config

        if config.predict_difference:
            if config.predict_all_dims:
                self.advance = self.advance_pred_all_diff
            else:
                raise NotImplementedError(
                    "When not predicting full state the config's parameters are not"
                    " currently sufficiently to figure out which dims to predict")
        else:
            if config.predict_all_dims:
                self.advance = self.advance_pred_all
            else:
                raise NotImplementedError(
                    "When not predicting full state the config's parameters are not"
                    " currently sufficiently to figure out which dims to predict")

    def advance_pred_all_diff(self, xu, dx):
        x = xu[:, :self.orig_config.nx]
        return x + dx

    def advance_pred_all(self, xu, xup):
        return xup


class DynamicsModel(DynamicsBase):
    """
    All public API takes input and returns output in original xu space,
    while internally (functions starting with underscore) they all operate in transformed space.

    All API takes torch tensors as input.
    """

    @tensor_utils.handle_batch_input
    def predict(self, xu, already_transformed=False, get_next_state=True, return_in_orig_space=True):
        """
        Predict next state; will return with the same dimensions as xu
        :param xu: B x N x (nx + nu) or N x (nx + nu) full input (if missing B will add it)
        :param already_transformed: whether the input xu has already been transformed (such as in chained calls)
        :param get_next_state: whether to output the next state, or the output of the model without advancing state
        :param return_in_orig_space: whether the output should be in the original space or the transformed space
        :return: B x N x nx or N x nx next states
        """
        if self.ds.preprocessor and not already_transformed:
            dxb = self._apply_model(self.ds.preprocessor.transform_x(xu))
        else:
            dxb = self._apply_model(xu)

        if self.ds.preprocessor and return_in_orig_space:
            x = xu[:, :self.ds.original_config().nx]
            dxb = self.ds.preprocessor.invert_transform(dxb, x)

        if get_next_state:
            x = self.advance(xu, dxb)
            return x
        else:
            return dxb

    @tensor_utils.handle_batch_input
    def _batch_apply_model(self, xu):
        return self._apply_model(xu)

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

    def reset(self):
        pass


class SimultaneousLearnedLatentDynamics(DynamicsModel):
    def __init__(self, ds):
        super(SimultaneousLearnedLatentDynamics, self).__init__(ds)
        assert(ds.preprocessor is not None)
        # directly use the dynamics learned together with the invariant transform
        self.dynamics = ds.preprocessor.tsf.transforms[-1].tsf.dynamics
        self.dynamics.model.eval()

    def _apply_model(self, xu):
        return self.dynamics.sample(xu)

    def device(self):
        return next(self.dynamics.model.parameters()).device


class NetworkModelWrapper(LearnableParameterizedModel, DynamicsModel):
    def __init__(self, model_user: ModelUser, ds, **kwargs):
        self.user = model_user
        self.model = model_user.model

        DynamicsModel.__init__(self, ds)
        LearnableParameterizedModel.__init__(self, cfg.ROOT_DIR, **kwargs)
        self.name = "{}_{}".format(self.name, ds.config)

        self.writer = None

    def modules(self):
        return {'dynamics': self.model}

    def evaluate_validation(self):
        """Any additional evaluation on the validation set goes here"""

    def evaluate_per_dim_sample_loss(self, XUv, Yv, ds):
        Yhat = self.user.sample(XUv)
        if not ds.preprocessor:
            E = (Yhat - Yv).abs()
            return E.mean(dim=0)
        XUv_orig, Yv_orig, _ = ds.original_validation_set()
        # compare in original space
        Xv_orig = XUv_orig[:, :ds.original_config().nx]
        Yhat_orig = ds.preprocessor.invert_transform(Yhat, Xv_orig)
        E = (Yhat_orig - Yv_orig).abs()
        return E.mean(dim=0)

    def learn_model(self, max_epoch, batch_N=500):
        self.XU, self.Y, self.labels = self.ds.training_set()
        self.XUv, self.Yv, self.labelsv = self.ds.validation_set()
        ds_train = load_data.SimpleDataset(self.XU, self.Y, self.labels)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_N, shuffle=True)

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            if self.writer is None:
                self.writer = SummaryWriter(flush_secs=20, comment=os.path.basename(self.name))
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()

            for i_batch, data in enumerate(train_loader):
                self.step += 1

                XU, Y, contacts = data

                self.optimizer.zero_grad()
                loss = self.user.compute_loss(XU, Y)

                # validation and other analysis
                with torch.no_grad():
                    self.writer.add_scalar('loss/training', loss.mean(), self.step)
                    vloss = self.user.compute_validation_loss(self.XUv, self.Yv, self.ds)
                    self.writer.add_scalar('loss/validation', vloss.mean(), self.step)
                    per_dim_vloss = self.evaluate_per_dim_sample_loss(self.XUv, self.Yv, self.ds)
                    for d in range(per_dim_vloss.shape[0]):
                        self.writer.add_scalar('per_dim_mean_abs_diff/validation{}'.format(d), per_dim_vloss[d].item(),
                                               self.step)
                    self.evaluate_validation()
                    logger.debug("Epoch %d loss %f vloss %f %s", epoch, loss.mean().item(), vloss.mean().item(),
                                 np.round(per_dim_vloss.cpu().numpy(), 3))

                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

        # save after training
        self.save(last=True)
        self._evaluate_against_least_squares()

    def load(self, filename):
        if super(NetworkModelWrapper, self).load(filename):
            self._evaluate_against_least_squares()
            return True
        return False

    def _evaluate_against_least_squares(self):
        # compare prediction accuracy against least squares
        XU, Y, _ = self.ds.training_set()
        params, res, rank, _ = np.linalg.lstsq(XU.cpu().numpy(), Y.cpu().numpy(), rcond=None)
        XU, Y, _ = self.ds.validation_set()
        Y = Y.cpu().numpy()
        Yhat = XU.cpu().numpy() @ params
        E = np.linalg.norm((Yhat - Y), axis=1)
        Yhatn = self.user.sample(XU).cpu().detach().numpy()
        En = np.linalg.norm((Yhatn - Y), axis=1)
        logger.info("Least squares error %f network error %f", E.mean(), En.mean())

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
        if not self.ds.config.predict_all_dims:
            dxb = dxb[:, self.nu:]
        return dxb
