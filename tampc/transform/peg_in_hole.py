import torch.nn
import torch
from arm_pytorch_utilities import math_utils, load_data, linalg, tensor_utils
from arm_pytorch_utilities.model import make
from tampc.dynamics import model
from tampc.env import peg_in_hole
from tampc.transform import invariant
from tampc.transform.invariant import TransformToUse
import logging

logger = logging.getLogger(__name__)


def translation_generator():
    for d in [10, 50]:
        for trans in [[1, 1], [-1, 1], [-1, -1]]:
            dd = (trans[0] * d, trans[1] * d)
            yield dd


class PegTransform(invariant.InvariantTransform):
    def _move_data_out_of_distribution(self, data, move_params):
        X, U, Y = data
        translation = move_params
        if translation:
            X = torch.cat((X[:, :2] + torch.tensor(translation, device=X.device, dtype=X.dtype), X[:, 2:]), dim=1)
        return X, U, Y

    def evaluate_validation(self, writer):
        losses = super(PegTransform, self).evaluate_validation(writer)
        if writer is not None:
            for dd in translation_generator():
                ls = self._evaluate_metrics_on_whole_set(True, TransformToUse.LATENT_SPACE, move_params=dd)
                self._record_metrics(writer, ls, suffix="/validation_{}_{}".format(dd[0], dd[1]))

                if self.ds_test is not None:
                    for i, ds_test in enumerate(self.ds_test):
                        ls = self._evaluate_metrics_on_whole_set(False, TransformToUse.LATENT_SPACE, move_params=dd,
                                                                 ds_test=ds_test)
                        if writer is not None:
                            self._record_metrics(writer, ls, suffix="/test{}_{}_{}".format(i, dd[0], dd[1]), log=True)

        return losses


class CoordTransform:
    @staticmethod
    def factory(env, *args, **kwargs):
        tsfs = {peg_in_hole.PegFloatingGripperEnv: CoordTransform.Base, }
        tsf_type = tsfs.get(type(env), None)
        if tsf_type is None:
            raise RuntimeError("No tsf specified for env type {}".format(type(env)))
        return tsf_type(*args, **kwargs)

    class Base(PegTransform):
        def __init__(self, ds, nv=5, **kwargs):
            super().__init__(ds, peg_in_hole.PegFloatingGripperEnv.nx - 2 + peg_in_hole.PegFloatingGripperEnv.nu, nv,
                             name='peg_translation', **kwargs)

        def get_v(self, x, dx, z):
            return self.dx_to_v(x, dx)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            # should be able to ignore xy
            z = torch.cat((state[:, 2:], action), dim=1)
            return z

        @tensor_utils.ensure_2d_input
        def dx_to_v(self, x, dx):
            return dx

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            return v

        def learn_model(self, max_epoch, batch_N=500):
            pass

