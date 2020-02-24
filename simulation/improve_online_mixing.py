import gpytorch
import torch
import logging
from gym import logger as gym_log
from arm_pytorch_utilities import rand, load_data
import matplotlib.pyplot as plt

from meta_contact import online_model
from meta_contact import prior
from meta_contact import model
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


class GPWithNominalModelMean(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nominal_model):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 0.7)
            ),
        )
        self.nominal_model = nominal_model

    def forward(self, x):
        mean_x = self.nominal_model(x)[:, 0]
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


relearn_dynamics = False
rand.seed(0)
# training data for nominal model
N = 200
x = torch.rand(N) * 10
x, _ = torch.sort(x)
u = torch.zeros(x.shape[0], 0)
e = torch.randn(N) * 0.1
y = torch.sin(x) + e

config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, y_in_x_space=False)
ds = PregeneratedDataset(x.view(-1, 1), u, y.view(-1, 1), config=config)
mw = model.NetworkModelWrapper(
    model.DeterministicUser(
        make.make_sequential_network(ds.config, h_units=(16, 16), activation_factory=torch.nn.Tanh).to(
            dtype=x.dtype)), ds,
    name="mix_nominal")

pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                             train_epochs=3000)

# make nominal predictions
yhat = pm.dyn_net.predict(x.view(-1, 1))

# plt.scatter(x.numpy(), y.numpy())
# plt.plot(x.numpy(), yhat.numpy())
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('nominal data and model')
# plt.show()

# local data
N = 30
xx = torch.rand(N) * 3 + 3
xx, _ = torch.sort(xx)
uu = torch.zeros(xx.shape[0], 0)
e = torch.randn(N) * 0.1
c = 1
yy = -0.2 * xx + c + e

config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, y_in_x_space=False)
ds_local = PregeneratedDataset(xx.view(-1, 1), uu, yy.view(-1, 1), config=config)

# linear local model (if we hand assigned mixing weights)
dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds_local, torch.sub, local_mix_weight_scale=1000,
                                            const_local_mix_weight=True)
yhat_linear_fit = dynamics.predict(None, None, x.view(-1, 1), u)

dynamics.local_weight_scale = 10.
yhat_baseline_mix = dynamics.predict(None, None, x.view(-1, 1), u)

dynamics.local_weight_scale = 1.
yhat_baseline_mix2 = dynamics.predict(None, None, x.view(-1, 1), u)

dynamics.const_local_weight = False
dynamics.local_weight_scale = 50.
dynamics.characteristic_length = 0.8
yhat_new1 = dynamics.predict(None, None, x.view(-1, 1), u)

# GP local model
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = xx
train_y = yy
model = GPWithNominalModelMean(train_x, train_y, likelihood, pm.dyn_net.predict)
training_iter = 50

# Find optimal model hyperparameters
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_pred = model(x)
    observed_pred = likelihood(f_pred)

# plt.scatter(x.numpy(), y.numpy(), alpha=0.2)
# plt.plot(x.numpy(), yhat.numpy(), alpha=0.2)
# plt.scatter(xx.numpy(), yy.numpy())
# plt.plot(x.numpy(), yhat_linear_fit.numpy(), label='linear fit')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('local data and models')
# plt.legend()
# plt.show()

plt.scatter(x.numpy(), y.numpy(), alpha=0.2)
plt.plot(x.numpy(), yhat.numpy(), alpha=0.2)
plt.scatter(xx.numpy(), yy.numpy(), alpha=0.2)
plt.plot(x.numpy(), yhat_linear_fit.numpy(), label='linear fit', alpha=0.2)
# plt.plot(x.numpy(), yhat_baseline_mix.numpy(), label='baseline fit high w')
# plt.plot(x.numpy(), yhat_baseline_mix2.numpy(), label='baseline fit low w')
# plt.plot(x.numpy(), yhat_new1.numpy(), label='our fit')

with torch.no_grad():
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # Plot predictive means as blue line
    plt.plot(x.numpy(), observed_pred.mean.numpy(), label='GP mean')
    # Shade between the lower and upper confidence bounds
    plt.fill_between(x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2, label='GP 2 std')

plt.xlabel('x')
plt.ylabel('y')
plt.title('local data and models')
plt.legend()
plt.show()
