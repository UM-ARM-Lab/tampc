import numpy as np
import torch
import logging
import math
import os
import scipy.io
from gym import wrappers, logger as gym_log
from arm_pytorch_utilities import rand, load_data, math_utils
import matplotlib.pyplot as plt

from meta_contact import online_model
from meta_contact import prior
from meta_contact import model
from meta_contact import cfg
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.make_data import datasource

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class PregeneratedDataset(datasource.DataSource):
    def __init__(self, x, u, y, **kwargs):
        self.x = x
        self.u = u
        self.y = y
        super().__init__(**kwargs)

    def make_data(self):
        self.config.load_data_info(self.x, self.u, self.y)
        self.N = self.x.shape[0]

        xu = torch.cat((self.x, self.u), dim=1)
        y = self.y
        if self.preprocessor:
            self.preprocessor.tsf.fit(xu, y)
            self.preprocessor.update_data_config(self.config)
            self._val_unprocessed = xu, y, None
            # apply
            xu, y, _ = self.preprocessor.tsf.transform(xu, y)

        self._train = xu, y, None
        self._val = self._train


rand.seed(0)
# training data for nominal model
N = 200
x = torch.rand(N) * 10
e = torch.randn(N) * 0.1
y = torch.sin(x) + e

plt.scatter(x.numpy(), y.numpy())
plt.show()

mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(ds.config).to(device=d)), ds,
                   name="dynamics_{}".format(tsf_name))

pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                             train_epochs=600)

