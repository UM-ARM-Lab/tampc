import os
import abc
import numpy as np
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
import torch
from arm_pytorch_utilities.optim import Lookahead
from meta_contact import cfg
from tensorboardX import SummaryWriter
import logging

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
        E = (Y - Yhat).norm(2, dim=1) ** 2
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
            def advance(x, u, dxb):
                return x + dxb
        else:
            # directly move the pusher
            def advance(x, u, dxb):
                return x + cat((u, dxb))
    else:
        if config.predict_all_dims:
            def advance(x, u, dxb):
                return dxb
        else:
            def advance(x, u, dxb):
                return cat((x[:config.nu] + u, dxb))

    return advance


class DynamicsModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x, u):
        """
        :param x: N x nx current state
        :param u: N x nu current action
        :return: N x nx next state
        """


class NetworkModelWrapper(DynamicsModel):
    def __init__(self, model_user: ModelUser, dataset, lr=1e-3, regularization=1e-5, name='', lookahead=True):
        self.dataset = dataset
        self.optimizer = None
        self.step = 0
        self.name = name
        self.advance = advance_state(dataset.config)
        self.XU, self.Y, self.labels = self.dataset.training_set()
        self.XUv, self.Yv, self.labelsv = self.dataset.validation_set()
        self.user = model_user
        self.model = model_user.model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=regularization)
        if lookahead:
            self.optimizer = Lookahead(self.optimizer)

        self.writer = SummaryWriter(flush_secs=20, comment=os.path.basename(name))

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _accumulate_stats(self, loss, vloss):
        self.writer.add_scalar('loss/training', loss, self.step)
        self.writer.add_scalar('loss/validation', vloss, self.step)

    def learn_model(self, max_epoch, batch_N=500):
        ds_train = load_data.SimpleDataset(self.XU, self.Y, self.labels)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_N, shuffle=True)
        self.step = 0

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

                logger.info("Epoch %d loss %f", epoch, loss.mean().item())
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

        dxb = self.user.sample(xu)

        if self.dataset.preprocessor:
            dxb = self.dataset.preprocessor.invert_transform(dxb).reshape(-1)

        # TODO why do I need to convert to numpy again?
        if torch.is_tensor(dxb):
            dxb = dxb.numpy()

        x = self.advance(x, u, dxb)
        return x


class LinearModel(DynamicsModel):
    def __init__(self, ds):
        self.dataset = ds
        XU, Y, _ = ds.training_set()
        self.nu = ds.config.nu
        self.nx = ds.config.nx
        # get dynamics
        params, res, rank, _ = np.linalg.lstsq(XU.numpy(), Y.numpy())
        # our call is setup to handle residual dynamics, so need to make sure that's the case
        if not ds.config.predict_difference:
            raise RuntimeError("Dynamics is set up to only handle residual dynamics")
        if ds.config.predict_all_dims:
            raise RuntimeError("Dynamics is set up to only handle block dynamics")
        # convert dyanmics to x' = Ax + Bu (note that our y is dx, so have to add diag(1))
        self.A = np.zeros((self.nx, self.nx))
        self.A[2:, :] += params[:self.nx, :].T
        self.B = np.zeros((self.nx, self.nu))
        self.B[0, 0] = 1
        self.B[1, 1] = 1
        self.B[2:, :] += params[self.nx:, :].T
        self.advance = advance_state(ds.config, use_np=True)

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
        x = self.advance(x, u, dxb)
        return x


class LinearModelTorch(LinearModel):
    def __init__(self, ds):
        super().__init__(ds)
        self.A = torch.from_numpy(self.A)
        self.B = torch.from_numpy(self.B)
        self.advance = advance_state(ds.config, use_np=False)

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

        x = self.advance(x, u, dxb)
        return x
