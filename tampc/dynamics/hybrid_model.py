import torch
from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import serialization
from tampc.controller import gating_function
from tampc.dynamics import online_model
from arm_pytorch_utilities.make_data import datasource
import logging
import abc
import enum
import copy

logger = logging.getLogger(__name__)


class DirectDataSource(datasource.DataSource):
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
            # save old data (if it's for the first time we're using a preprocessor)
            if self._original_val is None:
                self._original_train = xu, y, None
                self._original_val = self._original_train
            # apply on training and validation set
            xu, y, _ = self.preprocessor.tsf.transform(xu, y)

        self._train = xu, y, None
        self._val = self._train


class OnlineAdapt(enum.IntEnum):
    NONE = 0
    LINEARIZE_LIKELIHOOD = 1
    GP_KERNEL = 2
    GP_KERNEL_INDEP_OUT = 3
    GP_KERNEL_TOTALLY_INDEP_OUT = 4


class UseGating:
    MLP = 0
    KDE = 1
    GMM = 2
    TREE = 3
    FORCE = 4
    MLP_SKLEARN = 5
    KNN = 6


def get_gating(dss, tsf_name, use_gating=UseGating.TREE, *args, **kwargs):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors.classification import KNeighborsClassifier

    component_scale = [1, 0.2]
    # TODO this is specific to coordinate transform to slice just the body frame reaction force
    # input_slice = slice(3, None)
    input_slice = None

    if use_gating is UseGating.MLP:
        gating = gating_function.MLPSelector(dss, *args, **kwargs, name=tsf_name, input_slice=input_slice)
    elif use_gating is UseGating.KDE:
        gating = gating_function.KDESelector(dss, component_scale=component_scale, input_slice=input_slice)
    elif use_gating is UseGating.GMM:
        opts = {'n_components': 10, }
        if kwargs is not None:
            opts.update(kwargs)
        gating = gating_function.GMMSelector(dss, gmm_opts=opts, variational=True, component_scale=component_scale,
                                             input_slice=input_slice)
    elif use_gating is UseGating.TREE:
        gating = gating_function.SklearnClassifierSelector(dss, DecisionTreeClassifier(**kwargs),
                                                           input_slice=input_slice)
    elif use_gating is UseGating.FORCE:
        gating = gating_function.ReactionForceHeuristicSelector(12, slice(3, None))
    elif use_gating is UseGating.MLP_SKLEARN:
        gating = gating_function.SklearnClassifierSelector(dss, MLPClassifier(**kwargs), input_slice=input_slice)
    elif use_gating is UseGating.KNN:
        gating = gating_function.SklearnClassifierSelector(dss, KNeighborsClassifier(n_neighbors=1, **kwargs),
                                                           input_slice=input_slice)
    else:
        raise RuntimeError("Unrecognized selector option")
    return gating


def _one_norm_dist(x, x_new):
    return (x_new - x).abs().sum(dim=1)


class HybridDynamicsModel(serialization.Serializable):
    """Different way of mixing local and nominal model; use nominal as mean"""

    def __init__(self, dss, pm, state_diff, state_dist, gating_args, gating_kwargs=None, nominal_model_kwargs=None,
                 local_model_kwargs=None, device='cpu', preprocessor=None, ensemble=(), residual_model_trust_horizon=1,
                 project_by_default=True):
        self.dss = dss
        self.preprocessor = preprocessor
        self.pm = pm
        self.ds_nominal = dss[0]
        self.state_diff = state_diff
        self.state_dist = state_dist
        self.gating_args = gating_args
        self.gating_kwargs = gating_kwargs or {}
        self.local_model_kwargs = local_model_kwargs or {}
        self.residual_model_trust_horizon = residual_model_trust_horizon
        self.d = device
        self.ensemble = ensemble
        self.project_by_default = project_by_default

        # consider expected variation
        # get error per dimension to scale our expectations of accuracy
        XU, Y, _ = dss[0].training_set(original=True)
        predictions = []
        for model in self.ensemble:
            predictions.append(model.predict(XU))
        predictions = torch.stack(predictions)
        var = predictions.std(dim=0)
        self.expected_variance = var.mean(dim=0)

        nominal_model_kwargs = nominal_model_kwargs or {}
        self.nominal_model = self.get_local_model(self.state_diff, self.pm, device, self.ds_nominal, allow_update=True,
                                                  **nominal_model_kwargs)
        self._original_nominal_model = self.nominal_model
        self.using_residual_model = False

        self.local_models = []
        for i, ds_local in enumerate(self.dss):
            if i == 0:
                continue
            local_model = self.get_local_model(self.state_diff, self.pm, self.d, ds_local,
                                               preprocessor=self.preprocessor, **self.local_model_kwargs)
            self.local_models.append(local_model)

    def state_dict(self) -> dict:
        state = {'using_residual_model': self.using_residual_model}
        if self.using_residual_model:
            state['nominal_model'] = self.nominal_model.state_dict()
        return state

    def load_state_dict(self, state: dict) -> bool:
        self.using_residual_model = state['using_residual_model']
        if self.using_residual_model:
            self.use_residual_model()
            if not self.nominal_model.load_state_dict(state['nominal_model']):
                return False
        return True

    def get_local_model(self, state_diff, pm, d, ds_local, preprocessor=None, allow_update=False,
                        online_adapt=OnlineAdapt.GP_KERNEL_TOTALLY_INDEP_OUT,
                        train_slice=None, nom_projection=True):
        nominal_state_projection = self._bound_state_projection if nom_projection else None
        local_dynamics = pm.dyn_net if pm is not None else None
        # if given a preprocessor, we will override the datasource's preprocessor
        if preprocessor:
            ds_local = copy.deepcopy(ds_local)
            ds_local.update_preprocessor(preprocessor)

        if online_adapt is OnlineAdapt.LINEARIZE_LIKELIHOOD:
            local_dynamics = online_model.OnlineLinearizeMixing(0.1 if allow_update else 0.0, pm, ds_local,
                                                                state_diff,
                                                                local_mix_weight_scale=50, xu_characteristic_length=10,
                                                                const_local_mix_weight=False, sigreg=1e-10,
                                                                slice_to_use=train_slice, device=d,
                                                                nominal_state_projection=nominal_state_projection)
        elif online_adapt in [OnlineAdapt.GP_KERNEL, OnlineAdapt.GP_KERNEL_INDEP_OUT,
                              OnlineAdapt.GP_KERNEL_TOTALLY_INDEP_OUT]:
            model_class = online_model.OnlineGPMixing
            if online_adapt is OnlineAdapt.GP_KERNEL_TOTALLY_INDEP_OUT:
                model_class = online_model.OnlineGPMixingModelList
            local_dynamics = model_class(pm, ds_local, state_diff, slice_to_use=train_slice,
                                         allow_update=allow_update, sample=True,
                                         refit_strategy=online_model.RefitGPStrategy.RESET_DATA,
                                         device=d, training_iter=150,
                                         use_independent_outputs=online_adapt is OnlineAdapt.GP_KERNEL_INDEP_OUT,
                                         nominal_state_projection=nominal_state_projection)

        return local_dynamics

    def get_gating(self):
        return get_gating(self.dss, *self.gating_args, **self.gating_kwargs)

    def num_local_models(self):
        return len(self.local_models)

    def reset(self):
        self.nominal_model.reset()
        # don't need to reset local models since those aren't updated anyway

    def update(self, px, pu, cx):
        # we don't touch local models, but we can update our nominal mixed model if applicable
        if self._uses_local_model_api(self.nominal_model):
            return self.nominal_model.update(px, pu, cx)

    @staticmethod
    def _uses_local_model_api(model):
        return isinstance(model, online_model.OnlineDynamicsModel)

    def create_empty_local_model(self, use_prior=True, preprocessor=preprocess.NoTransform(), **kwargs):
        if 'nom_projection' not in kwargs:
            kwargs['nom_projection'] = self.project_by_default
        return self.get_local_model(self.state_diff, self.pm if use_prior else None,
                                    self.d, self.ds_nominal,
                                    preprocessor=preprocessor,
                                    allow_update=True, train_slice=slice(0, 0), **kwargs)

    def use_residual_model(self):
        if self._uses_local_model_api(self.nominal_model):
            # start local model here with no previous data points
            self.nominal_model.init_xu = self.nominal_model.init_xu[slice(0, 0)]
            self.nominal_model.init_y = self.nominal_model.init_y[slice(0, 0)]
            self.nominal_model.reset()
        else:
            self.nominal_model = self.create_empty_local_model(preprocessor=self.preprocessor)
        self.using_residual_model = True

    def use_normal_nominal_model(self):
        self.nominal_model = self._original_nominal_model
        self.using_residual_model = False

    @tensor_utils.handle_batch_input
    def __call__(self, x, u, cls, t=0):
        next_state = torch.zeros_like(x)

        # while we trust the residual model
        if t < self.residual_model_trust_horizon:
            nominal_cls = cls == 0
            if torch.any(nominal_cls):
                if self._uses_local_model_api(self.nominal_model):
                    next_state[nominal_cls] = self.nominal_model.predict(None, None, x[nominal_cls], u[nominal_cls])
                else:
                    next_state[nominal_cls] = self.nominal_model.predict(
                        torch.cat((x[nominal_cls], u[nominal_cls]), dim=1))
            # local models
            for s in range(self.num_local_models()):
                local_cls = cls == (s + 1)
                if torch.any(local_cls):
                    next_state[local_cls] = self.local_models[s].predict(None, None, x[local_cls], u[local_cls])
        else:
            # after stop using residual model we project input once again so the nominal model gets known input
            if t == self.residual_model_trust_horizon and self.using_residual_model and self.project_by_default:
                logger.info("projecting output of residual model for future nominal model rollouts")
                x_known = self.project_input_to_training_distribution(x, u, state_distance=self.state_dist)
                x = x_known
            next_state = self._original_nominal_model.predict(torch.cat((x, u), dim=1))

        return next_state

    @tensor_utils.ensure_2d_input
    def epistemic_uncertainty(self, x, u):
        xu = torch.cat((x, u), dim=1)
        predictions = []
        for model in self.ensemble:
            predictions.append(model.predict(xu))
        predictions = torch.stack(predictions)
        var = predictions.std(dim=0) / self.expected_variance
        return var.mean(dim=1)

    @tensor_utils.ensure_2d_input
    def project_input_to_training_distribution(self, x, u, state_distance=_one_norm_dist, dist_weight=10,
                                               delta_loss_threshold=0.005, lr=0.1, plot=False):
        if x.shape[0] == 0:
            return x
        x_new = x.clone()
        x_new.requires_grad = True
        uncertainties = []
        xs = [x_new.detach().cpu().numpy()]

        optimizer = torch.optim.Adam([x_new], lr=lr)
        steps = 0

        delta_l = 0
        last_l = None
        while True:
            steps += 1
            optimizer.zero_grad()

            e = self.epistemic_uncertainty(x_new, u)

            dist = state_distance(x, x_new)
            loss = e.abs() + dist_weight * dist
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                l = loss.max()
                if last_l is not None:
                    delta_l = delta_l * 0.9 + (last_l - l) * 0.1
                    last_l = l
                else:
                    delta_l = l
                    last_l = l
                if delta_l < delta_loss_threshold and delta_l > 0:
                    break

                if plot:
                    logger.debug('%f %s', delta_l, e.cpu().numpy())
                    xs.append(x_new.cpu().numpy())
                    uncertainties.append(e.cpu().numpy())
        x_new.requires_grad = False

        if plot:
            import matplotlib.pyplot as plt

            iters = range(len(uncertainties))
            xs = xs[:-1]
            f, axes = plt.subplots(4 + 1, 1, sharex='all', figsize=(8, 9))
            axes[0].set_ylabel('epistemic uncertainty')
            axes[1].set_ylabel('diff x value')
            axes[2].set_ylabel('diff y value')
            axes[3].set_ylabel('rx value')
            axes[4].set_ylabel('ry value')
            axes[-1].set_xlabel('iterations')

            axes[0].set_yscale('log')
            for j in range(5):
                axes[j].set_xscale('log')

            for i in range(min(x.shape[0], 5)):
                axes[0].plot(iters, [e[i] for e in uncertainties],
                             label='init {}'.format(x[i].cpu().numpy().round(decimals=2)))
                axes[0].grid('y')
                for j in range(4):
                    if j < 2:
                        axes[j + 1].plot(iters, [v[i, j] - x[i, j] for v in xs])
                    else:
                        axes[j + 1].plot(iters, [v[i, j] for v in xs])
                    axes[j + 1].grid('y')

            axes[0].legend()
            f.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        return x_new

    def _bound_state_projection(self, xu):
        config = self.ds_nominal.original_config()
        x, u = xu[:, :config.nx], xu[:, config.nx:]
        x_known = self.project_input_to_training_distribution(x, u, state_distance=self.state_dist)
        return torch.cat((x_known, u), dim=1)
