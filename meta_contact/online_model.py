import torch
from arm_pytorch_utilities import linalg
from meta_contact import model
from meta_contact import prior
from torch.distributions.multivariate_normal import MultivariateNormal
import logging
import gpytorch
import abc

logger = logging.getLogger(__name__)


class OnlineDynamicsModel(abc.ABC):
    """Different way of mixing local and nominal model; use nominal as mean"""

    def __init__(self, ds, state_difference, device=torch.device("cpu")):
        self.ds = ds
        self.advance = model.advance_state(ds.original_config(), use_np=False)
        self.state_difference = state_difference

        self.nx = ds.config.nx
        self.nu = ds.config.nu

        # device the prior model is on
        self.d = device
        self.dtype = torch.double

    @abc.abstractmethod
    def reset(self):
        """Clear state of model"""

    def update(self, px, pu, cx):
        """Update local model with new (x,u,x') data point in original space"""
        y = self.state_difference(cx, px) if self.ds.original_config().predict_difference else cx
        if self.ds.preprocessor:
            x, u, y = self._make_2d_tensor(px, pu, y)
            xu = torch.cat((x, u), dim=1)
            xu, y, _ = self.ds.preprocessor.tsf.transform(xu, y)
            px = xu[:, :self.nx].view(-1)
            pu = xu[:, self.nx:].view(-1)
            y = y.view(-1)
        self._update(px, pu, y)

    @abc.abstractmethod
    def _update(self, px, pu, y):
        """Do update with (x,u,y) data point in transformed space"""

    def _make_2d_tensor(self, *args):
        if args[0] is None:
            return args
        oned = len(args[0].shape) is 1
        if not torch.is_tensor(args[0]):
            args = (torch.from_numpy(value).to(device=self.d) if value is not None else None for value in args)
        if oned:
            args = (value.view(1, -1) if value is not None else None for value in args)
        return args

    def _apply_transform(self, x, u):
        if x is None:
            return x, u
        xu = torch.cat((x, u), dim=1)
        xu = self.ds.preprocessor.transform_x(xu)
        x = xu[:, :self.nx]
        u = xu[:, self.nx:]
        return x, u

    def predict(self, px, pu, cx, cu, already_transformed=False):
        """
        Predict next state; will return with the same dimensions as cx
        :return: B x N x nx or N x nx next states
        """

        ocx = cx  # original state
        cx, cu, px, pu = self._make_2d_tensor(cx, cu, px, pu)
        # transform if necessary (ensure dynamics is evaluated only in transformed space)
        if self.ds.preprocessor and not already_transformed:
            cx, cu = self._apply_transform(cx, cu)
            px, pu = self._apply_transform(px, pu)

        y = self._dynamics_in_transformed_space(px, pu, cx, cu)

        if self.ds.preprocessor:
            y = self.ds.preprocessor.invert_transform(y, ocx)

        next_state = self.advance(ocx, y)

        return next_state

    @abc.abstractmethod
    def _dynamics_in_transformed_space(self, cx, cu, px, pu):
        """Return y in transformed space"""


class OnlineLinearizeMixing(OnlineDynamicsModel):
    """ Moving average estimate of locally linear dynamics from https://arxiv.org/pdf/1509.06841.pdf

    Note gamma here is (1-gamma) described in the paper, so high gamma forgets quicker.

    All dynamics public API takes input and returns output in original xu space,
    while internally (functions starting with underscore) they all operate in transformed space.

    Currently all the batch API takes torch tensors as input/output while all the single API takes numpy arrays
    """

    def __init__(self, gamma, online_prior: prior.OnlineDynamicsPrior, ds, state_difference,
                 xu_characteristic_length=1., sigreg=1e-5, slice_to_use=None, local_mix_weight_scale=1.,
                 const_local_mix_weight=True, **kwargs):
        """
        :param gamma: How fast to update our empirical local model, with 1 being to completely forget the previous model
        every time we get new data
        :param online_prior: A global prior model that is linearizable (can get Jacobian)
        :param ds: Some data source
        :param state_difference: Function (nx, nx) -> nx getting a - b in state space (order matters!)
        :param xu_characteristic_length: Like for GPs, define what does close in xu space mean; higher value means
        distance drops off faster
        :param local_mix_weight_scale: How much to scale the local weights
        :param const_local_mix_weight: Weight of mixing empirical local model with prior model; relative to n0 and
        m of the prior model, which are typically 1, so use 1 for equal weighting
        :param sigreg: How much to regularize conditioning of dynamics from p(x,u,x') to p(x'|x,u)
        """
        super().__init__(ds, state_difference, **kwargs)
        self.gamma = gamma
        self.prior = online_prior
        self.sigreg = sigreg  # Covariance regularization (adds sigreg*eye(N))

        # mixing parameters
        self.const_local_weight = const_local_mix_weight
        self.local_weight_scale = local_mix_weight_scale
        self.characteristic_length = xu_characteristic_length
        self.local_weights = None

        self.emp_error = None
        self.prior_error = None

        self.prior_trust_coefficient = 0.1  # the lower it is the more we trust the prior; 0 means only ever use prior
        self.sigma, self.mu, self.xxt = None, None, None
        # Initial values
        self.init_sigma, self.init_mu = prior.gaussian_params_from_datasource(ds, slice_to_use)
        self.init_xxt = self.init_sigma + torch.ger(self.init_mu, self.init_mu)
        self.reset()

    def reset(self):
        self.sigma, self.mu, self.xxt = self.init_sigma.clone(), self.init_mu.clone(), self.init_xxt.clone()
        self.local_weights = None
        self.emp_error = None
        self.prior_error = None

    def evaluate_error(self, px, pu, cx, cu):
        """After updating dynamics and using that dynamics to plan an action,
        evaluate the error of empirical and prior dynamics"""
        # can't evaluate if it's the first action
        if px is None:
            self.emp_error = self.prior_error = None
            return

        cx, cu, px, pu = self._make_2d_tensor(cx, cu, px, pu)
        opx, ocx = px, cx
        if self.ds.preprocessor:
            cx, cu = self._apply_transform(cx, cu)
            px, pu = self._apply_transform(px, pu)

        xu, pxu, xux = _cat_xu(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_batch_params(self.nx, self.nu, xu, pxu, xux)

        # mix prior and empirical distribution
        Fe, fe, _ = _batch_conditioned_dynamics(self.nx, self.nu, self.sigma.unsqueeze(0), self.mu.unsqueeze(0),
                                                sigreg=self.sigreg)
        Fp, fp, _ = _batch_conditioned_dynamics(self.nx, self.nu, Phi / n0, mu0, sigreg=self.sigreg)

        emp_y = _batch_evaluate_dynamics(px, pu, Fe, fe)
        prior_y = _batch_evaluate_dynamics(px, pu, Fp, fp)

        if self.ds.preprocessor:
            emp_y = self.ds.preprocessor.invert_transform(emp_y, opx)
            prior_y = self.ds.preprocessor.invert_transform(prior_y, opx)

        emp_x = self.advance(opx, emp_y)
        prior_x = self.advance(opx, prior_y)

        # compare against actual x'
        self.emp_error = self.state_difference(emp_x, ocx).norm()
        self.prior_error = self.state_difference(prior_x, ocx).norm()
        # TODO update gamma based on the relative error of these dynamics
        # rho = self.emp_error / self.prior_error
        # # high gamma means to trust empirical model (paper uses 1-rho, but this makes more sense)
        # self.gamma = self.prior_trust_coefficient / rho

    def _update(self, px, pu, y):
        xux = torch.cat((px, pu, y))

        gamma = self.gamma
        # Do a moving average update (equations 3,4)
        self.mu = self.mu * (1 - gamma) + xux * (gamma)
        self.xxt = self.xxt * (1 - gamma) + torch.ger(xux, xux) * gamma
        self.xxt = 0.5 * (self.xxt + self.xxt.t())
        self.sigma = self.xxt - torch.ger(self.mu, self.mu)

    def _get_batch_dynamics(self, px, pu, cx, cu):
        """
        Compute F, f - the linear dynamics where either dx or next_x = F*[curx, curu] + f
        The semantics depends on the data source the prior was trained on and that this was initialized on
        """
        # prior parameters
        xu, pxu, xux = _cat_xu(px, pu, cx, cu)
        Phi, mu0, m, n0 = self.prior.get_batch_params(self.nx, self.nu, xu, pxu, xux)

        # marginalize joint gaussian of (x,u,x') to get (x,u) then use likelihood of cx, cu as local mix weight
        if not self.const_local_weight:
            xu_slice = slice(self.nx + self.nu)
            mu_xu = self.mu[xu_slice]
            sigma_xu = self.sigma[xu_slice, xu_slice]
            d = MultivariateNormal(mu_xu, sigma_xu)
            self.local_weight = torch.exp(d.log_prob(xu) * self.characteristic_length)
        else:
            self.local_weight = torch.ones(cx.shape[0], device=self.d, dtype=cx.dtype)
        # mix prior and empirical distribution
        sigma, mu = prior.batch_mix_distribution(self.sigma, self.mu,
                                                 self.local_weight.view(-1, 1) * self.local_weight_scale, Phi, mu0, m,
                                                 n0)
        return _batch_conditioned_dynamics(self.nx, self.nu, sigma, mu, self.sigreg)

    def _dynamics_in_transformed_space(self, px, pu, cx, cu):
        params = self._get_batch_dynamics(px, pu, cx, cu)
        y = _batch_evaluate_dynamics(cx, cu, *params)
        return y


class MixingFullGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nominal_model, rank=1):
        super(MixingFullGP, self).__init__(train_x, train_y, likelihood)
        self.nominal_model = nominal_model
        ny = train_y.shape[1]
        # TODO use priors and constraints on the kernels
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=ny
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=ny, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x) + self.nominal_model(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MixingSingleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_index):
        super().__init__(train_x, train_y, likelihood)
        self.i = output_index
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x, nominal_output):
        mean_x = self.mean_module(x) + nominal_output(x, self.i)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NominalModelCache:
    def __init__(self, prior):
        self.prior = prior
        self.res = {}

    def __call__(self, x, i):
        # use size of x as a proxy for whether we've created the result or not
        N = x.shape[0]
        if N not in self.res:
            self.res[N] = self.prior.get_batch_predictions(x)
        return self.res[N][:, i]


class OnlineGPMixing(OnlineDynamicsModel):
    """Different way of mixing local and nominal model; use nominal as mean"""

    def __init__(self, prior: prior.OnlineDynamicsPrior, ds, state_difference, max_data_points=50, training_iter=100,
                 slice_to_use=None, partial_refit=True, use_independent_outputs=False,
                 **kwargs):
        super().__init__(ds, state_difference, **kwargs)
        self.prior = prior

        # actually keep track of data used to fit rather than doing recursive
        self.max_data_points = max_data_points
        self.xu = None
        self.y = None
        self.likelihood = None
        self.gp = None
        self.optimizer = None
        self.training_iter = training_iter
        self.partial_refit = partial_refit
        self.use_independent_outputs = use_independent_outputs

        # mixing parameters
        self.last_prediction = None

        if slice_to_use is None:
            slice_to_use = slice(max_data_points)
        xu, y, _ = self.ds.training_set()
        self.init_xu = xu[slice_to_use]
        self.init_y = y[slice_to_use]
        # Initial values
        self.reset()

    def reset(self):
        self.xu = self.init_xu.clone()
        self.y = self.init_y.clone()
        self._recreate_gp()
        self._fit_params(self.training_iter)

    def _recreate_gp(self):
        if self.use_independent_outputs:
            self.ls = []
            self.gps = []
            for i in range(self.ds.config.ny):
                self.ls.append(gpytorch.likelihoods.GaussianLikelihood().to(device=self.d, dtype=self.dtype))
                self.gps.append(
                    MixingSingleGP(self.xu, self.y[:, i], self.ls[i], i).to(device=self.d, dtype=self.dtype))
            self.likelihood = gpytorch.likelihoods.LikelihoodList(*self.ls)
            self.gp = gpytorch.models.IndependentModelList(*self.gps)
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.ds.config.ny).to(
                device=self.d, dtype=self.dtype)
            self.gp = MixingFullGP(self.xu, self.y, self.likelihood, self.prior.get_batch_predictions,
                                   rank=1).to(device=self.d,
                                              dtype=self.dtype)
        self.optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},
        ], lr=0.1)

    def _update(self, px, pu, y):
        xu = torch.cat((px.view(1, -1), pu.view(1, -1)), dim=1)
        y = y.view(1, -1)
        if self.xu.shape[0] < self.max_data_points:
            self.xu = torch.cat((self.xu, xu), dim=0)
            self.y = torch.cat((self.y, y), dim=0)
        else:
            self.xu = torch.roll(self.xu, -1, dims=0)
            self.xu[-1] = xu
            self.y = torch.roll(self.y, -1, dims=0)
            self.y[-1] = y

        # try refitting from previous parameters (fewer iterations)
        if self.partial_refit:
            if self.use_independent_outputs:
                # for i in range(self.ds.config.ny):
                #     self.gps[0].set_train_data(self.xu, self.y[:, i], strict=False)
                self.gps = []
                for i in range(self.ds.config.ny):
                    self.gps.append(
                        MixingSingleGP(self.xu, self.y[:, i], self.ls[i], i).to(device=self.d, dtype=self.dtype))
                self.gp = gpytorch.models.IndependentModelList(*self.gps)
            else:
                self.gp.set_train_data(self.xu, self.y, strict=False)
            self._fit_params(self.training_iter // 10)
        else:
            # refit from scratch
            self._recreate_gp()
            self._fit_params(self.training_iter)

    def _fit_params(self, training_iter):
        import time
        start = time.time()
        # tune hyperparameters to new data
        self.gp.train()
        self.likelihood.train()
        if self.use_independent_outputs:
            mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.gp)
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        for i in range(training_iter):
            self.optimizer.zero_grad()
            if self.use_independent_outputs:
                nominal_output = NominalModelCache(self.prior)
                output = self.gp(*self.gp.train_inputs, nominal_output=nominal_output)
                loss = -mll(output, self.gp.train_targets)
            else:
                output = self.gp(self.xu)
                loss = -mll(output, self.y)
            loss.backward()
            logger.debug('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            self.optimizer.step()

        self.gp.eval()
        self.likelihood.eval()
        elapsed = time.time() - start
        logger.debug('training took %.4fs', elapsed)

    def _dynamics_in_transformed_space(self, px, pu, cx, cu):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            xu = torch.cat((cx, cu), dim=1)
            if self.use_independent_outputs:
                xs = [xu for _ in range(self.ds.config.ny)]
                nominal_output = NominalModelCache(self.prior)
                b = self.gp(*xs, nominal_output=nominal_output)
                self.last_prediction = self.likelihood(*b)
                y = torch.stack([v.mean for v in self.last_prediction], dim=1)
            else:
                self.last_prediction = self.likelihood(self.gp(xu))
                # not using covariance, but could plot them with .confidence_region()
                y = self.last_prediction.mean
            return y

    def get_confidence_region(self):
        with torch.no_grad():
            if self.use_independent_outputs:
                confidence_regions = [v.confidence_region() for v in self.last_prediction]
                lower = torch.stack([b[0] for b in confidence_regions], dim=1)
                upper = torch.stack([b[1] for b in confidence_regions], dim=1)
            else:
                lower, upper = self.last_prediction.confidence_region()
            return lower, upper


def _cat_xu(px, pu, cx, cu):
    xu = torch.cat((cx, cu), 1)
    pxu = torch.cat((px, pu), 1) if px is not None else None
    xux = torch.cat((px, pu, cx), 1) if px is not None else None
    return xu, pxu, xux


def _batch_conditioned_dynamics(nx, nu, sigma, mu, sigreg=1e-5):
    # TODO assume we want to calculate everything to the right of xu (can't handle pxu)
    it = slice(nx + nu)
    ip = slice(nx + nu, None)
    # guarantee symmetric positive definite with regularization
    sigma[:, it, it] += sigreg * torch.eye(nx + nu, dtype=sigma.dtype, device=sigma.device)

    # sometimes fails for no good reason on the GPU; try a couple of times
    u = None
    for t in range(3):
        try:
            u = torch.cholesky(sigma[:, it, it])
            # if we succeeded just exit for loop
            break
        except RuntimeError as e:
            logger.warning(e)

    if u is None:
        raise RuntimeError("Failed cholesky multiple times; actual problem with data")

    # equivalent to inv * sigma
    # Solve normal equations to get dynamics. (equation 2)
    Fm = torch.cholesky_solve(sigma[:, it, ip], u).transpose(-1, -2)
    fv = mu[:, ip] - linalg.batch_batch_product(mu[:, it], Fm)

    # TODO calculate dyn_covar
    return Fm, fv, None


def _batch_evaluate_dynamics(x, u, F, f, cov=None):
    xu = torch.cat((x, u), 1)
    xp = linalg.batch_batch_product(xu, F) + f
    return xp
