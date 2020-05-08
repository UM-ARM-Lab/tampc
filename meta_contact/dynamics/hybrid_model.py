import torch
from arm_pytorch_utilities import tensor_utils
from meta_contact.dynamics import online_model
import logging
import abc

logger = logging.getLogger(__name__)


class HybridDynamicsModel(abc.ABC):
    """Different way of mixing local and nominal model; use nominal as mean"""

    def __init__(self, nominal_model, local_models):
        self.nominal_model = nominal_model
        self.local_models = local_models

        # nominal model could be mixed with local data (and have a local model API)
        self.local_model_api_for_nom_model = isinstance(self.nominal_model, online_model.OnlineDynamicsModel)

    def num_local_models(self):
        return len(self.local_models)

    def reset(self):
        self.nominal_model.reset()
        # don't need to reset local models since those aren't updated anyway

    def update(self, px, pu, cx):
        # we don't touch local models, but we can update our nominal mixed model if applicable
        if isinstance(self.nominal_model, online_model.OnlineDynamicsModel):
            return self.nominal_model.update(px, pu, cx)

    @tensor_utils.handle_batch_input
    def __call__(self, x, u, cls):
        next_state = torch.zeros_like(x)

        # nominal model
        nominal_cls = cls == 0
        if torch.any(nominal_cls):
            if self.local_model_api_for_nom_model:
                next_state[nominal_cls] = self.nominal_model.predict(None, None, x[nominal_cls], u[nominal_cls])
            else:
                next_state[nominal_cls] = self.nominal_model.predict(torch.cat((x[nominal_cls], u[nominal_cls]), dim=1))
        # local models
        for s in range(self.num_local_models()):
            local_cls = cls == (s + 1)
            if torch.any(local_cls):
                next_state[local_cls] = self.local_models[s].predict(None, None, x[local_cls], u[local_cls])

        return next_state
