import os
import torch
import logging
from tensorboardX import SummaryWriter
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
import numpy as np
from meta_contact import cfg

logger = logging.getLogger(__name__)


class Prior:
    def __init__(self, model, name, dataset, lr, regularization):
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

    def learn_model(self, max_epoch, batch_N=200):
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
        xu = torch.tensor(np.concatenate((x, u))).reshape(1,-1)

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
