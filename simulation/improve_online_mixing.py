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
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


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


relearn_dynamics = False
rand.seed(0)
# training data for nominal model
N = 200
x = torch.rand(N) * 10
x, _ = torch.sort(x)
e = torch.randn(N) * 0.1
y = torch.sin(x) + e

config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, y_in_x_space=False)
ds = PregeneratedDataset(x.view(-1, 1), torch.zeros(x.shape[0], 0), y.view(-1, 1), config=config)
mw = model.NetworkModelWrapper(
    model.DeterministicUser(
        make.make_sequential_network(ds.config, h_units=(16, 16), activation_factory=torch.nn.Tanh).to(
            dtype=x.dtype)), ds,
    name="mix_nominal")

pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                             train_epochs=3000)

# make nominal predictions
yhat = pm.dyn_net.predict(x.view(-1, 1))

plt.scatter(x.numpy(), y.numpy())
plt.plot(x.numpy(), yhat.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.title('nominal data and model')
plt.show()

# local data
N = 30
xx = torch.rand(N) * 3 + 3
xx, _ = torch.sort(xx)
e = torch.randn(N) * 0.1
c = 1
yy = -0.2 * xx + c + e

plt.scatter(x.numpy(), y.numpy(), alpha=0.2)
plt.plot(x.numpy(), yhat.numpy(), alpha=0.2)
plt.scatter(xx.numpy(), yy.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.title('local data')
plt.show()
