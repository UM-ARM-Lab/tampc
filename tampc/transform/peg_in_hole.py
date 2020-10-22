import torch.nn
import torch
from arm_pytorch_utilities import tensor_utils
from tampc.env import peg_in_hole
from tampc.transform.invariant import TranslationEvaluationTransform
import logging

logger = logging.getLogger(__name__)


def translation_generator():
    for d in [10, 50]:
        for trans in [[1, 1], [-1, 1], [-1, -1]]:
            dd = (trans[0] * d, trans[1] * d)
            yield dd


class CoordTransform:
    @staticmethod
    def factory(env, *args, **kwargs):
        tsfs = {peg_in_hole.PegFloatingGripperEnv: CoordTransform.Base, }
        tsf_type = tsfs.get(type(env), None)
        if tsf_type is None:
            raise RuntimeError("No tsf specified for env type {}".format(type(env)))
        return tsf_type(*args, **kwargs)

    class Base(TranslationEvaluationTransform):
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
