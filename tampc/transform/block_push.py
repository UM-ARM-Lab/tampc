import torch.nn
import torch
from arm_pytorch_utilities import math_utils, tensor_utils
from tampc.env import block_push
from tampc.transform import invariant
from tampc.transform.invariant import TranslationEvaluationTransform
import logging

logger = logging.getLogger(__name__)


class PusherNeighboursTransform(TranslationEvaluationTransform, invariant.InvariantNeighboursTransform):
    def _is_in_neighbourhood(self, cur, candidate):
        return abs(cur[3] - candidate[3]) < self.too_far_for_neighbour

    def _calculate_pairwise_dist(self, X, U):
        relevant_xu = torch.cat((X[:, 3].view(-1, 1), U), dim=1)
        return torch.cdist(relevant_xu, relevant_xu)


class CoordTransform:
    @staticmethod
    def factory(env, *args, **kwargs):
        tsfs = {block_push.PushAgainstWallStickyEnv: CoordTransform.StickyEnv,
                block_push.PushWithForceDirectlyEnv: CoordTransform.Direct,
                block_push.PushPhysicallyAnyAlongEnv: CoordTransform.Physical, }
        tsf_type = tsfs.get(type(env), None)
        if tsf_type is None:
            raise RuntimeError("No tsf specified for env type {}".format(type(env)))
        return tsf_type(*args, **kwargs)

    class Base(TranslationEvaluationTransform):
        def __init__(self, ds, nz, nv=4, **kwargs):
            # assume 4 states; anything after we should just passthrough
            # v is dx, dy, dyaw in body frame and d_along
            super().__init__(ds, nz, nv, name='coord', **kwargs)

        def get_v(self, x, dx, z):
            return self.dx_to_v(x, dx)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            # (along, d_along, push magnitude, push direction)
            z = torch.cat((state[:, 3].view(-1, 1), action), dim=1)
            return z

        @tensor_utils.ensure_2d_input
        def dx_to_v(self, x, dx):
            # convert world frame to body frame
            dpos_body = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = dx[:, 2]
            # last element is d_along, which gets passed along directly
            dalong = dx[:, 3]
            v = torch.cat((dpos_body, dyaw.view(-1, 1), dalong.view(-1, 1)), dim=1)
            return v

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            # convert (dx, dy) from body frame back to world frame
            dpos_world = math_utils.batch_rotate_wrt_origin(v[:, :2], x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = v[:, 2]
            # last element is d_along, which gets passed along directly
            dalong = v[:, 3]
            dx = torch.cat((dpos_world, dyaw.view(-1, 1), dalong.view(-1, 1)), dim=1)
            return dx

        def learn_model(self, max_epoch, batch_N=500):
            pass

    class StickyEnv(Base):
        """
        Specific to StickyEnv formulation! (expects the states to be block pose and pusher along)

        Transforms world frame coordinates to input required for body frame dynamics
        (along, d_along, and push magnitude) = z and predicts (dx, dy, dtheta) of block in previous block frame = v
        separate latent space for input and output (z, v)
        this is actually h and h^{-1} combined into 1, h(x,u) = z, learned dynamics hat{f}(z) = v, h^{-1}(v) = dx
        """

        def __init__(self, ds, **kwargs):
            # need along, d_along, and push magnitude; don't need block position or yaw
            super().__init__(ds, 3, **kwargs)

    class Direct(Base):
        def __init__(self, ds, **kwargs):
            # need along, d_along, push magnitude, and push direction; don't need block position or yaw
            super().__init__(ds, 4, **kwargs)

    class Physical(Base):
        def __init__(self, ds, **kwargs):
            # v is dpose and dr in body frame
            super().__init__(ds, ds.config.nu + 2, nv=3 + 2, **kwargs)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            r_body = math_utils.batch_rotate_wrt_origin(state[:, -2:], -state[:, 2])
            z = torch.cat((action, r_body), dim=1)
            return z

        @tensor_utils.ensure_2d_input
        def dx_to_v(self, x, dx):
            # convert world frame to body frame
            dpos_body = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
            dr_body = math_utils.batch_rotate_wrt_origin(dx[:, -2:], -x[:, 2])
            dyaw = dx[:, 2]
            v = torch.cat((dpos_body, dyaw.view(-1, 1), dr_body), dim=1)
            return v

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            # convert (dx, dy) from body frame back to world frame
            dpos_world = math_utils.batch_rotate_wrt_origin(v[:, :2], x[:, 2])
            dr_world = math_utils.batch_rotate_wrt_origin(v[:, -2:], x[:, 2])
            dyaw = v[:, 2]
            dx = torch.cat((dpos_world, dyaw.view(-1, 1), dr_world), dim=1)
            return dx
