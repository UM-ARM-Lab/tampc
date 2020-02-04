import torch
from arm_pytorch_utilities import linalg
from meta_contact import model
from meta_contact import prior
import logging

logger = logging.getLogger(__name__)


class OnlineDynamicsModel(object):
    """ Moving average estimate of locally linear dynamics from https://arxiv.org/pdf/1509.06841.pdf

    Note gamma here is (1-gamma) described in the paper, so high gamma forgets quicker.

    All dynamics public API takes input and returns output in original xu space,
    while internally (functions starting with underscore) they all operate in transformed space.

    Currently all the batch API takes torch tensors as input/output while all the single API takes numpy arrays
    """

    def __init__(self, gamma, online_prior: prior.OnlineDynamicsPrior, ds, state_difference, local_mix_weight=1.,
                 device='cpu', sigreg=1e-5):
        """
        :param gamma: How fast to update our empirical local model, with 1 being to completely forget the previous model
        every time we get new data
        :param online_prior: A global prior model that is linearizable (can get Jacobian)
        :param ds: Some data source
        :param state_difference: Function (nx, nx) -> nx getting a - b in state space (order matters!)
        :param local_mix_weight: Weight of mixing empirical local model with prior model; relative to n0 and m of the
        prior model, which are typically 1, so use 1 for equal weighting
        :param sigreg: How much to regularize conditioning of dynamics from p(x,u,x') to p(x'|x,u)
        """
        self.gamma = gamma
        self.prior = online_prior
        self.sigreg = sigreg  # Covariance regularization (adds sigreg*eye(N))
        self.ds = ds
        self.advance = model.advance_state(ds.original_config(), use_np=False)
        self.state_difference = state_difference

        self.nx = ds.config.nx
        self.nu = ds.config.nu
        self.empsig_N = local_mix_weight
        self.emp_error = None
        self.prior_error = None
        # device the prior model is on
        self.d = device

        self.prior_trust_coefficient = 0.1  # the lower it is the more we trust the prior; 0 means only ever use prior
        self.sigma, self.mu, self.xxt = None, None, None
        # Initial values
        self.init_sigma, self.init_mu = prior.gaussian_params_from_datasource(ds)
        self.init_xxt = self.init_sigma + torch.ger(self.init_mu, self.init_mu)
        self.reset()

    def reset(self):
        self.sigma, self.mu, self.xxt = self.init_sigma.clone(), self.init_mu.clone(), self.init_xxt.clone()

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

    def update(self, px, pu, cx):
        """ Perform a moving average update on the current dynamics """
        # our internal dynamics could be on dx or x', so convert x' to whatever our model works with
        y = self.state_difference(cx, px) if self.ds.original_config().predict_difference else cx
        # convert xux to transformed coordinates
        if self.ds.preprocessor:
            x, u, y = self._make_2d_tensor(px, pu, y)
            xu = torch.cat((x, u), dim=1)
            xu, y, _ = self.ds.preprocessor.tsf.transform(xu, y)
            px = xu[:, :self.nx].view(-1)
            pu = xu[:, self.nx:].view(-1)
            y = y.view(-1)
        # should be (zi, zo) in transformed space
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

        # mix prior and empirical distribution
        # TODO decrease empsig_N with increasing horizon (trust global model more for more distant points)
        sigma, mu = prior.batch_mix_distribution(self.sigma, self.mu, self.empsig_N, Phi, mu0, m, n0)
        return _batch_conditioned_dynamics(self.nx, self.nu, sigma, mu, self.sigreg)

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

        params = self._get_batch_dynamics(px, pu, cx, cu)
        y = _batch_evaluate_dynamics(cx, cu, *params)

        if self.ds.preprocessor:
            y = self.ds.preprocessor.invert_transform(y, ocx)

        next_state = self.advance(ocx, y)

        return next_state


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
