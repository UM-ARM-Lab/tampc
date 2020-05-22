import torch
from arm_pytorch_utilities import tensor_utils
from meta_contact.controller import gating_function
from meta_contact.dynamics import online_model
from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities import optim
import logging
import abc
import enum

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


class HybridDynamicsModel(abc.ABC):
    """Different way of mixing local and nominal model; use nominal as mean"""

    def __init__(self, dss, pm,  state_diff, gating_args, gating_kwargs=None, nominal_model_kwargs=None,
                 local_model_kwargs=None, device=optim.get_device()):
        self.dss = dss
        self.pm = pm
        self.ds_nominal = dss[0]
        self.state_diff = state_diff
        self.gating_args = gating_args
        self.gating_kwargs = gating_kwargs or {}
        self.local_model_kwargs = local_model_kwargs or {}

        nominal_model_kwargs = nominal_model_kwargs or {}
        self.nominal_model = HybridDynamicsModel.get_local_model(self.state_diff, self.pm, device, self.ds_nominal,
                                                                 allow_update=True,
                                                                 **nominal_model_kwargs)
        self._original_nominal_model = self.nominal_model

        self.local_models = []
        for i, ds_local in enumerate(self.dss):
            if i == 0:
                continue
            local_model = HybridDynamicsModel.get_local_model(self.state_diff, self.pm, ds_local,
                                                              **self.local_model_kwargs)
            self.local_models.append(local_model)

    @staticmethod
    def get_local_model(state_diff, pm, d, ds_local, allow_update=False, online_adapt=OnlineAdapt.GP_KERNEL,
                        train_slice=None):
        local_dynamics = pm.dyn_net
        if online_adapt is OnlineAdapt.LINEARIZE_LIKELIHOOD:
            local_dynamics = online_model.OnlineLinearizeMixing(0.1 if allow_update else 0.0, pm, ds_local,
                                                                state_diff,
                                                                local_mix_weight_scale=50, xu_characteristic_length=10,
                                                                const_local_mix_weight=False, sigreg=1e-10,
                                                                slice_to_use=train_slice, device=d)
        elif online_adapt is OnlineAdapt.GP_KERNEL:
            local_dynamics = online_model.OnlineGPMixing(pm, ds_local, state_diff, slice_to_use=train_slice,
                                                         allow_update=allow_update, sample=True,
                                                         refit_strategy=online_model.RefitGPStrategy.RESET_DATA,
                                                         device=d, training_iter=150, use_independent_outputs=False)

        return local_dynamics

    def create_local_model(self, x, u):
        logger.info("Saving local model from previous escape")

        config = self.ds_nominal.config
        assert config.predict_difference
        y = self.state_diff(x[1:], x[:-1])
        ds_local = DirectDataSource(x[1:], u[1:], y)
        ds_local.update_preprocessor(self.ds_nominal.preprocessor)

        local_model = HybridDynamicsModel.get_local_model(self.state_diff, self.pm, self.nominal_model.device(),
                                                          ds_local, allow_update=False)

        self.dss.append(ds_local)
        self.local_models.append(local_model)

        return local_model

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

    def use_recovery_nominal_model(self):
        if self._uses_local_model_api(self.nominal_model):
            # start local model here with no previous data points
            self.nominal_model.init_xu = self.nominal_model.init_xu[slice(0, 0)]
            self.nominal_model.init_y = self.nominal_model.init_y[slice(0, 0)]
            self.nominal_model.reset()
        else:
            self.nominal_model = online_model.OnlineGPMixing(self.pm, self.ds_nominal, self.state_diff,
                                                             allow_update=True, sample=True, slice_to_use=slice(0, 0),
                                                             device=self.nominal_model.device())

    def use_normal_nominal_model(self):
        self.nominal_model = self._original_nominal_model

    @tensor_utils.handle_batch_input
    def __call__(self, x, u, cls):
        next_state = torch.zeros_like(x)

        # nominal model
        nominal_cls = cls == 0
        if torch.any(nominal_cls):
            if self._uses_local_model_api(self.nominal_model):
                next_state[nominal_cls] = self.nominal_model.predict(None, None, x[nominal_cls], u[nominal_cls])
            else:
                next_state[nominal_cls] = self.nominal_model.predict(torch.cat((x[nominal_cls], u[nominal_cls]), dim=1))
        # local models
        for s in range(self.num_local_models()):
            local_cls = cls == (s + 1)
            if torch.any(local_cls):
                next_state[local_cls] = self.local_models[s].predict(None, None, x[local_cls], u[local_cls])

        return next_state
