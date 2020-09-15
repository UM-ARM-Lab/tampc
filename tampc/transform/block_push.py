import torch.nn
import torch
from arm_pytorch_utilities import math_utils, load_data, linalg, tensor_utils
from arm_pytorch_utilities.model import make
from tampc.dynamics import model
from tampc.env import block_push
from tampc.transform import invariant
from tampc.transform.invariant import TransformToUse
import logging

logger = logging.getLogger(__name__)


def translation_generator():
    for d in [10, 50]:
        for trans in [[1, 1], [-1, 1], [-1, -1]]:
            dd = (trans[0] * d, trans[1] * d)
            yield dd


class PusherTransform(invariant.InvariantTransform):
    def _move_data_out_of_distribution(self, data, move_params):
        X, U, Y = data
        translation = move_params
        if translation:
            X = torch.cat((X[:, :2] + torch.tensor(translation, device=X.device, dtype=X.dtype), X[:, 2:]), dim=1)
        return X, U, Y

    def evaluate_validation(self, writer):
        losses = super(PusherTransform, self).evaluate_validation(writer)
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


class PusherNeighboursTransform(PusherTransform, invariant.InvariantNeighboursTransform):
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

    class Base(PusherTransform):
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


class LearnedTransform:
    class DxToV(PusherTransform):
        def __init__(self, ds, device, nz=5, nv=5, mse_weight=0, reconstruction_weight=1, match_weight=1,
                     encoder_opts=None,
                     decoder_opts=None, dynamics_opts=None, **kwargs):
            self.mse_weight = mse_weight
            self.reconstruction_weight = reconstruction_weight
            self.match_weight = match_weight
            # TODO try penalizing mutual information between xu and z, and v and dx?
            # create encoder xu -> z
            opts = {'h_units': (16, 32)}
            if encoder_opts:
                opts.update(encoder_opts)
            config = load_data.DataConfig()
            config.nx = ds.config.nx + ds.config.nu
            config.ny = nz
            self.encoder = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))

            # TODO try extracting from x
            # create v,x -> dx
            opts = {'h_units': (16, 32)}
            if decoder_opts:
                opts.update(decoder_opts)
            config = load_data.DataConfig()
            config.nx = ds.config.nx
            config.ny = nv * ds.config.nx  # matrix output (original nx, ignore sincos)
            # outputs a linear transformation from v to dx (linear in v), that is dependent on state
            # v C(x) = dx --> v = C(x)^{-1} dx allows both ways
            self.linear_decoder_producer = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))

            # create dynamics (shouldn't have high capacity since we should have simple dynamics in trasnformed space)
            # z -> v
            opts = {'h_units': (16, 16)}
            if dynamics_opts:
                opts.update(dynamics_opts)
            config = load_data.DataConfig()
            config.nx = nz
            config.ny = nv
            self.dynamics = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))
            name = kwargs.pop('name', '')
            super().__init__(ds, nz=nz, nv=nv, name='{}_{}'.format(self._name_prefix(), name), **kwargs)

        def modules(self):
            return {'encoder': self.encoder.model, 'linear decoder': self.linear_decoder_producer.model,
                    'dynamics': self.dynamics.model}

        def _name_prefix(self):
            return 'two_routes_{}_{}_{}'.format(self.mse_weight, self.reconstruction_weight, self.match_weight)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            z = self.encoder.sample(torch.cat((state, action), dim=1))
            return z

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            # make state-dependent linear transforms (matrices) that multiply v to get dx
            B, nx = x.shape
            linear_tsf = self.linear_decoder_producer.sample(x).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

        @tensor_utils.ensure_2d_input
        def get_v(self, x, dx, z):
            B, nx = x.shape
            v_to_dx = self.linear_decoder_producer.sample(x).view(B, nx, self.nv)
            dx_to_v = torch.pinverse(v_to_dx)
            v = linalg.batch_batch_product(dx, dx_to_v)
            return v

        def get_yhat(self, X, U, Y):
            z = self.xu_to_z(X, U)
            # forward dynamics route
            vhat = self.dynamics.sample(z)
            yhat = self.get_dx(X, vhat)
            return yhat

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z = self.xu_to_z(X, U)
            # forward dynamics route
            vhat = self.dynamics.sample(z)
            yhat = self.get_dx(X, vhat)

            # backward decoder route
            v = self.get_v(X, Y, z)
            y_reconstruct = self.get_dx(X, v)

            # matching of the dynamics v and the decoder v
            match_decoder = torch.norm(vhat - v, dim=1)
            # reconstruction of decoder
            reconstruction = torch.norm(y_reconstruct - Y, dim=1)
            # mse loss
            mse_loss = torch.norm(yhat - Y, dim=1)

            return mse_loss, reconstruction, match_decoder, reconstruction / Y.norm(
                dim=1).mean(), match_decoder / v.norm(dim=1).mean()

        @staticmethod
        def loss_names():
            return "mse_loss", "reconstruction", "match_decoder", "percent_reconstruction", "percent_match"

        def loss_weights(self):
            return [self.mse_weight, 0, 0, self.reconstruction_weight, self.match_weight]

    class SeparateDecoder(DxToV):
        """Use a separate network for x,dx -> v instead of taking the inverse"""

        def __init__(self, ds, device, nv=5, inverse_decoder_opts=None, **kwargs):
            # create v,x -> dx
            opts = {'h_units': (16, 32)}
            if inverse_decoder_opts:
                opts.update(inverse_decoder_opts)
            config = load_data.DataConfig()
            config.nx = ds.config.nx
            config.ny = nv * ds.config.nx
            # outputs a linear transformation from v to dx (linear in v), that is dependent on state
            # v C(x) = dx --> v = C(x)^{-1} dx allows both ways
            self.inverse_linear_decoder_producer = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))
            super().__init__(ds, device, nv=nv, **kwargs)

        def modules(self):
            return {'encoder': self.encoder.model, 'linear decoder': self.linear_decoder_producer.model,
                    'inverse linear decoder': self.inverse_linear_decoder_producer.model,
                    'dynamics': self.dynamics.model}

        def _name_prefix(self):
            return 'sep_dec_{}_{}_{}'.format(self.mse_weight, self.reconstruction_weight, self.match_weight)

        @tensor_utils.ensure_2d_input
        def get_v(self, x, dx, z):
            B, nx = x.shape
            dx_to_v = self.inverse_linear_decoder_producer.sample(x).view(B, self.nv, nx)
            v = linalg.batch_batch_product(dx, dx_to_v)
            return v

    class ExtractState(SeparateDecoder):
        """Don't pass through all of x to g"""

        def __init__(self, ds, device, *args, nz=5, nv=5, reduced_decoder_input_dim=2, decoder_opts=None,
                     inverse_decoder_opts=None, **kwargs):
            self.reduced_decoder_input_dim = reduced_decoder_input_dim
            self.x_extractor = torch.nn.Linear(ds.config.nx, self.reduced_decoder_input_dim).to(device=device,
                                                                                                dtype=torch.double)
            opts = {'h_units': (16, 32)}
            if decoder_opts:
                opts.update(decoder_opts)
            config = load_data.DataConfig()
            config.nx = self.reduced_decoder_input_dim
            config.ny = nv * ds.config.nx
            # h_\rho
            self.extracted_linear_decoder = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))

            opts = {'h_units': (16, 32)}
            if inverse_decoder_opts:
                opts.update(inverse_decoder_opts)
            config = load_data.DataConfig()
            config.nx = self.reduced_decoder_input_dim
            config.ny = nv * ds.config.nx
            # outputs a linear transformation from v to dx (linear in v), that is dependent on state
            # v C(x) = dx --> v = C(x)^{-1} dx allows both ways
            # h_\eta
            self.extracted_inverse_linear_decoder_producer = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))

            super().__init__(ds, device, *args, nz=nz, nv=nv, **kwargs)

        def modules(self):
            return {'encoder': self.encoder.model, 'extracted linear decoder': self.extracted_linear_decoder.model,
                    'extracted inverse linear decoder': self.extracted_inverse_linear_decoder_producer.model,
                    'extractor': self.x_extractor, 'dynamics': self.dynamics.model}

        def _name_prefix(self):
            return 'extract_{}'.format(self.reduced_decoder_input_dim)

        @tensor_utils.ensure_2d_input
        def get_v(self, x, dx, z):
            B, nx = x.shape
            dx_to_v = self.extracted_inverse_linear_decoder_producer.sample(self.x_extractor(x)).view(B, self.nv, nx)
            v = linalg.batch_batch_product(dx, dx_to_v)
            return v

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            B, nx = x.shape
            linear_tsf = self.extracted_linear_decoder.sample(self.x_extractor(x)).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

    class RexExtract(ExtractState, invariant.RexTraining):
        def _name_prefix(self):
            return 'rex_extract_{}'.format(self.reduced_decoder_input_dim)

        def learn_model(self, *args, **kwargs):
            return invariant.RexTraining.learn_model(self, *args, **kwargs)

    class ParameterizeYawSelect(invariant.LearnLinearDynamicsTransform, PusherTransform):
        """DEPRECATED Parameterize the coordinate transform such that it has to learn something"""

        def __init__(self, ds, device, model_opts=None, nz=5, nv=5, **kwargs):
            if model_opts is None:
                model_opts = {}
            # default values for the input model_opts to replace
            opts = {'h_units': (16, 32)}
            opts.update(model_opts)

            # v is dx, dy, dyaw in body frame and d_along
            # input is x, output is yaw
            self.yaw_selector = torch.nn.Linear(ds.config.nx, 1, bias=False).to(device=device, dtype=torch.double)
            self.true_yaw_param = torch.zeros(ds.config.nx, device=device, dtype=torch.double)
            self.true_yaw_param[2] = 1
            self.true_yaw_param = self.true_yaw_param.view(1, -1)  # to be consistent with weights
            # try starting at the true parameters
            # self.yaw_selector.weight.data = self.true_yaw_param + torch.randn_like(self.true_yaw_param)
            # self.yaw_selector.weight.requires_grad = False

            # input to local model is z, output is v
            config = load_data.DataConfig()
            config.nx = nz
            config.ny = nv * nz  # matrix output
            self.linear_model_producer = model.DeterministicUser(
                make.make_sequential_network(config, **opts).to(device=device))
            name = kwargs.pop('name', '')
            invariant.LearnLinearDynamicsTransform.__init__(self, ds, nz, nv,
                                                            name='{}_{}'.format(self._name_prefix(), name),
                                                            **kwargs)

        def modules(self):
            return {'yaw': self.yaw_selector, 'linear dynamics': self.linear_model_producer.model}

        def _name_prefix(self):
            return 'param_coord'

        def linear_dynamics(self, z):
            B = z.shape[0]
            return self.linear_model_producer.sample(z).view(B, self.nv, self.nz)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            # (along, d_along, push magnitude)
            z = torch.cat((state[:, -1].view(-1, 1), action), dim=1)
            return z

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            # choose which component of x to take as rotation (should select just theta)
            yaw = self.yaw_selector(x)

            N = v.shape[0]
            dx = torch.zeros((N, 4), dtype=v.dtype, device=v.device)
            # convert (dx, dy) from body frame back to world frame
            dx[:, :2] = math_utils.batch_rotate_wrt_origin(v[:, :2], yaw)
            # second last element is dyaw, which also gets passed along directly
            dx[:, 2] = v[:, 2]
            # last element is d_along, which gets passed along directly
            dx[:, 3] = v[:, 3]
            return dx

        def evaluate_validation(self, writer):
            losses = super().evaluate_validation(writer)
            if writer is not None:
                with torch.no_grad():
                    yaw_param = self.yaw_selector.weight.data
                    cs = torch.nn.functional.cosine_similarity(yaw_param, self.true_yaw_param).item()
                    dist = torch.norm(yaw_param - self.true_yaw_param).item()

                    writer.add_scalar('cosine_similarity', cs, self.step)
                    writer.add_scalar('param_diff', dist, self.step)
                    writer.add_scalar('param_norm', yaw_param.norm().item(), self.step)
            return losses

    class LinearComboLatentInput(ParameterizeYawSelect):
        """DEPRECATED Relax parameterization structure to allow (each dimension of) z to be some linear combination of x,u"""

        def __init__(self, ds, device, nz=4, **kwargs):
            # input is x, output is z
            # constrain output to 0 and 1
            self.z_selector = torch.nn.Linear(ds.config.nx + ds.config.nu, nz, bias=False).to(device=device,
                                                                                              dtype=torch.double)
            self.true_z_param = torch.tensor(
                [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]],
                device=device,
                dtype=torch.double)
            # try starting at the true parameters
            # self.z_selector.weight.data = self.true_z_param + torch.randn_like(self.true_z_param) * 0.1
            # self.z_selector.weight.requires_grad = True
            super().__init__(ds, device, nz=nz, **kwargs)

        def modules(self):
            m = super().modules()
            m['z selector'] = self.z_selector
            return m

        def _name_prefix(self):
            return "z_select"

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            # more general parameterized versions where we select which components to take
            xu = torch.cat((state, action), dim=1)
            z = self.z_selector(xu)
            return z

    class ParameterizeDecoder(LinearComboLatentInput):
        """DEPRECATED Relax parameterization structure to allow decoder to be some state dependent transformation of v"""

        def __init__(self, ds, device, use_sincos_angle=False, nv=5, **kwargs):
            # replace angle with their sin and cos
            self.use_sincos_angle = use_sincos_angle
            # input to producer is x, output is matrix to multiply v to get dx by
            config = load_data.DataConfig()
            config.nx = ds.config.nx + (1 if use_sincos_angle else 0)
            config.ny = nv * ds.config.nx  # matrix output (original nx, ignore sincos)
            # outputs a linear transformation from v to dx (linear in v), that is dependent on state
            self.linear_decoder_producer = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(16, 32)).to(device=device))
            super().__init__(ds, device, nv=nv, **kwargs)

        def modules(self):
            return {'linear decoder': self.linear_decoder_producer.model, 'z selector': self.z_selector,
                    'linear dynamics': self.linear_model_producer.model}

        def _name_prefix(self):
            return 'state_dep_linear_tsf_{}'.format(int(self.use_sincos_angle))

        def linear_dynamics(self, z):
            B = z.shape[0]
            return self.linear_model_producer.sample(z).view(B, self.nv, self.nz)

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            # make state-dependent linear transforms (matrices) that multiply v to get dx
            B, nx = x.shape
            if self.use_sincos_angle:
                angle_index = 2
                s = torch.sin(x[:, angle_index]).view(-1, 1)
                c = torch.cos(x[:, angle_index]).view(-1, 1)
                x = torch.cat((x[:, :angle_index], s, c, x[:, angle_index + 1:]), dim=1)
            linear_tsf = self.linear_decoder_producer.sample(x).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

    class LearnedPartialPassthrough(ParameterizeDecoder):
        """DEPRECATED Don't pass through all of x to g; learn which parts to pass to g and which to h"""

        def __init__(self, ds, device, *args, nz=5, nv=5, reduced_decoder_input_dim=2, **kwargs):
            self.reduced_decoder_input_dim = reduced_decoder_input_dim
            self.x_extractor = torch.nn.Linear(ds.config.nx, self.reduced_decoder_input_dim).to(device=device,
                                                                                                dtype=torch.double)

            config = load_data.DataConfig()
            config.nx = self.reduced_decoder_input_dim
            config.ny = nv * ds.config.nx
            self.partial_decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(16, 32)).to(device=device))

            super().__init__(ds, device, *args, nz=nz, nv=nv, **kwargs)

        def modules(self):
            return {'partial decoder': self.partial_decoder.model, 'z selector': self.z_selector,
                    'linear dynamics': self.linear_model_producer.model, 'x extractor': self.x_extractor}

        def _name_prefix(self):
            return 'feedforward_passthrough_{}'.format(self.reduced_decoder_input_dim)

        def _evaluate_batch_apply_tsf(self, X, U, tsf):
            assert tsf is TransformToUse.LATENT_SPACE
            z = self.xu_to_z(X, U)
            A = self.linear_dynamics(z)
            v = linalg.batch_batch_product(z, A.transpose(-1, -2))
            yhat = self.get_dx(X, v)
            return z, A, v, yhat

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z, A, v, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)
            # reconstruction of decoder
            reconstruction = torch.norm(yhat - Y, dim=1)
            # mse loss
            mse_loss = torch.norm(yhat - Y, dim=1)

            return mse_loss, reconstruction, reconstruction / Y.norm(dim=1).mean(),

        @staticmethod
        def loss_names():
            return "mse_loss", "reconstruction", "percent_reconstruction"

        def loss_weights(self):
            return [0, 0, 1]

        @tensor_utils.ensure_2d_input
        def get_dx(self, x, v):
            B, nx = x.shape
            extracted_from_x = self.x_extractor(x)
            linear_tsf = self.partial_decoder.sample(extracted_from_x).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

    class SkipLatentInput(ExtractState):
        """Use a transform combining encoder to z and dynamics from z to v; equivalently, z=[x,u]"""

        def __init__(self, ds, device, *args, dynamics_opts=None, **kwargs):
            new_dyn_opts = {'h_units': (32, 32, 16)}
            if dynamics_opts:
                new_dyn_opts.update(dynamics_opts)
            super().__init__(ds, device, *args, nz=ds.config.nx + ds.config.nu, dynamics_opts=new_dyn_opts,
                             **kwargs)

        def modules(self):
            return {'extracted linear decoder': self.extracted_linear_decoder.model,
                    'extracted inverse linear decoder': self.extracted_inverse_linear_decoder_producer.model,
                    'extractor': self.x_extractor, 'encoder and dynamics': self.dynamics.model}

        def _name_prefix(self):
            return 'skipz_{}'.format(self.reduced_decoder_input_dim)

        @tensor_utils.ensure_2d_input
        def xu_to_z(self, state, action):
            return torch.cat((state, action), dim=1)

    class RexSkip(ExtractState, invariant.RexTraining):
        def _name_prefix(self):
            return 'rex_skip_{}'.format(self.reduced_decoder_input_dim)

        def learn_model(self, *args, **kwargs):
            return invariant.RexTraining.learn_model(self, *args, **kwargs)
