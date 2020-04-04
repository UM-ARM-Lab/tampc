import torch.nn
import torch
from arm_pytorch_utilities import math_utils, load_data, linalg
from arm_pytorch_utilities.model import make
from meta_contact.dynamics import model
from meta_contact.env import block_push
from meta_contact.transform import invariant
from meta_contact.transform.invariant import TransformToUse
import logging

logger = logging.getLogger(__name__)


def translation_generator():
    for d in [4, 10, 50]:
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
                ls = self._evaluate_metrics_on_whole_set(self.neighbourhood_validation, TransformToUse.LATENT_SPACE,
                                                         move_params=dd)
                self._record_metrics(writer, ls, suffix="/validation_{}_{}".format(dd[0], dd[1]))
        return losses

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
                block_push.PushWithForceDirectlyReactionInStateEnv: CoordTransform.ReactionInState,
                block_push.PushPhysicallyAnyAlongEnv: CoordTransform.Physical, }
        tsf_type = tsfs.get(type(env), None)
        if tsf_type is None:
            raise RuntimeError("No tsf specified for env type {}".format(type(env)))
        return tsf_type(*args, **kwargs)

    class Base(PusherTransform):
        def __init__(self, ds, nz, nv=4, reaction_force_start_index=4, **kwargs):
            # assume 4 states; anything after we should just passthrough
            self.passthrough_states = ds.config.nx - reaction_force_start_index
            # v is dx, dy, dyaw in body frame and d_along
            super().__init__(ds, nz + self.passthrough_states, nv + self.passthrough_states, name='coord', **kwargs)

        def _add_passthrough_states(self, z, x):
            # we always use the last few dims for passthrough; so use the same for v
            if self.passthrough_states:
                z = torch.cat((z, x[:, -self.passthrough_states:].view(-1, self.passthrough_states)), dim=1)
            return z

        def get_v(self, x, dx, z):
            return self.dx_to_v(x, dx)

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            # (along, d_along, push magnitude, push direction)
            z = torch.cat((state[:, 3].view(-1, 1), action), dim=1)
            z = self._add_passthrough_states(z, state)
            return z

        def dx_to_v(self, x, dx):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                dx = dx.view(1, -1)
            # convert world frame to body frame
            dpos_body = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = dx[:, 2]
            # last element is d_along, which gets passed along directly
            dalong = dx[:, 3]
            v = torch.cat((dpos_body, dyaw.view(-1, 1), dalong.view(-1, 1)), dim=1)
            v = self._add_passthrough_states(v, dx)
            return v

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)
            # convert (dx, dy) from body frame back to world frame
            dpos_world = math_utils.batch_rotate_wrt_origin(v[:, :2], x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = v[:, 2]
            # last element is d_along, which gets passed along directly
            dalong = v[:, 3]
            dx = torch.cat((dpos_world, dyaw.view(-1, 1), dalong.view(-1, 1)), dim=1)
            dx = self._add_passthrough_states(dx, v)
            return dx

        def parameters(self):
            return [torch.zeros(1)]

        def _model_state_dict(self):
            return None

        def _load_model_state_dict(self, saved_state_dict):
            pass

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

    class ReactionInState(Base):
        def __init__(self, ds, **kwargs):
            super().__init__(ds, 5, **kwargs)

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            # NOTE we could just as well make the model work for free-space if we rotate reaction force to be in body frame
            # but then it wouldn't work for pushing against a wall since it's no longer rotationally invariant
            # additionally need theta in z to estimate reaction force direction
            # (theta, along, d_along, push magnitude, push direction)
            z = torch.cat((state[:, 2:4], action), dim=1)
            z = self._add_passthrough_states(z, state)
            return z

    class Physical(Base):
        def __init__(self, ds, **kwargs):
            super().__init__(ds, ds.config.nu + 1, nv=3, reaction_force_start_index=3, **kwargs)

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            z = torch.cat((state[:, 2].view(-1, 1), action), dim=1)
            z = self._add_passthrough_states(z, state)
            return z

        def dx_to_v(self, x, dx):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                dx = dx.view(1, -1)
            # convert world frame to body frame
            dpos_body = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = dx[:, 2]
            v = torch.cat((dpos_body, dyaw.view(-1, 1)), dim=1)
            v = self._add_passthrough_states(v, dx)
            return v

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)
            # convert (dx, dy) from body frame back to world frame
            dpos_world = math_utils.batch_rotate_wrt_origin(v[:, :2], x[:, 2])
            # second last element is dyaw, which also gets passed along directly
            dyaw = v[:, 2]
            dx = torch.cat((dpos_world, dyaw.view(-1, 1)), dim=1)
            dx = self._add_passthrough_states(dx, v)
            return dx


# TODO handle passthrough states! (for now none of these should work with reaction force)
# TODO try mutual information penalty (want low mutual information of input and output - throw out noise)
def compression_reward(v, xu, dist_regularization=1e-8, top_percent=None):
    # normalize here so loss wouldn't change if all v were scaled by a fixed amount (tested)
    mn = v.norm(dim=1).mean()
    vv = v / mn
    dv = torch.nn.functional.pdist(vv)
    # TODO generalize special treatment of angular distance
    angle_index = 2
    s = torch.sin(xu[:, angle_index]).view(-1, 1)
    c = torch.cos(xu[:, angle_index]).view(-1, 1)
    xu = torch.cat((xu[:, :angle_index], s, c, xu[:, angle_index + 1:]), dim=1)
    # invariance captured by small distance in v despite large difference in XU
    dxu = torch.nn.functional.pdist(xu)
    # only consider the reward for the closest dv
    if top_percent is not None:
        k = int(top_percent * dv.shape[0])
        dv, indices = torch.topk(dv, k=k, largest=False, sorted=False)
        dxu = dxu[indices]
    # use log to prevent numerical issues so dxu/dv -> log(dxu) - log(dv)
    return torch.log(dxu) - torch.log(dv + dist_regularization)


class LearnedTransform:
    class ParameterizedYawSelect(invariant.LearnLinearDynamicsTransform, PusherTransform):
        """Parameterize the coordinate transform such that it has to learn something"""

        def __init__(self, ds, device, model_opts=None, nz=4, nv=4, **kwargs):
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

        def _name_prefix(self):
            return 'param_coord'

        def linear_dynamics(self, z):
            B = z.shape[0]
            return self.linear_model_producer.sample(z).view(B, self.nv, self.nz)

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            # (along, d_along, push magnitude)
            z = torch.cat((state[:, -1].view(-1, 1), action), dim=1)
            return z

        def dx_to_v(self, x, dx):
            raise RuntimeError("Shouldn't have to convert from dx to v")

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

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

        def parameters(self):
            return list(self.yaw_selector.parameters()) + list(self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = {'yaw': self.yaw_selector.state_dict(), 'linear': self.linear_model_producer.model.state_dict()}
            return d

        def _load_model_state_dict(self, saved_state_dict):
            self.yaw_selector.load_state_dict(saved_state_dict['yaw'])
            self.linear_model_producer.model.load_state_dict(saved_state_dict['linear'])

        def evaluate_validation(self, writer):
            losses = super().evaluate_validation(writer)
            if writer is not None:
                with torch.no_grad():
                    yaw_param = self.yaw_selector.weight.data
                    cs = torch.nn.functional.cosine_similarity(yaw_param, self.true_yaw_param).item()
                    dist = torch.norm(yaw_param - self.true_yaw_param).item()

                    logger.debug("step %d yaw cos sim %f dist %f", self.step, cs, dist)

                    writer.add_scalar('cosine_similarity', cs, self.step)
                    writer.add_scalar('param_diff', dist, self.step)
                    writer.add_scalar('param_norm', yaw_param.norm().item(), self.step)
            return losses

    class LinearComboLatentInput(ParameterizedYawSelect):
        """Relax parameterization structure to allow (each dimension of) z to be some linear combination of x,u"""

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

        def _name_prefix(self):
            return "z_select"

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            # more general parameterized versions where we select which components to take
            xu = torch.cat((state, action), dim=1)
            z = self.z_selector(xu)
            return z

        def parameters(self):
            return super().parameters() + list(self.z_selector.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['z'] = self.z_selector.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.z_selector.load_state_dict(saved_state_dict['z'])

    class ParameterizeDecoder(LinearComboLatentInput):
        """Relax parameterization structure to allow decoder to be some state dependent transformation of v"""

        def __init__(self, ds, device, use_sincos_angle=False, nv=4, **kwargs):
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

        def _name_prefix(self):
            return 'state_dep_linear_tsf_{}'.format(int(self.use_sincos_angle))

        def parameters(self):
            return list(self.linear_decoder_producer.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['decoder'] = self.linear_decoder_producer.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.linear_decoder_producer.model.load_state_dict(saved_state_dict['decoder'])

        def linear_dynamics(self, z):
            B = z.shape[0]
            return self.linear_model_producer.sample(z).view(B, self.nv, self.nz)

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

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

    class ParameterizeDecoderBatch(ParameterizeDecoder, invariant.LearnFromBatchTransform):
        """Train using randomized neighbourhoods instead of fixed given ones"""

        def learn_model(self, max_epoch, batch_N=500):
            return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)

    class LinearRelaxEncoder(ParameterizeDecoderBatch):
        """Still enforce that dynamics and decoder are linear, but relax parameterization of encoder"""

        def __init__(self, ds, device, nz=4, **kwargs):
            config = load_data.DataConfig()
            config.nx = ds.config.nx + ds.config.nu
            config.ny = nz
            self.encoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))

            super().__init__(ds, device, nz=nz, **kwargs)

        def _name_prefix(self):
            return 'linear_relax_encoder'

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            xu = torch.cat((state, action), dim=1)
            z = self.encoder.sample(xu)
            return z

        def parameters(self):
            return list(self.linear_decoder_producer.model.parameters()) + list(self.encoder.model.parameters()) + list(
                self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['encoder'] = self.encoder.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.encoder.model.load_state_dict(saved_state_dict['encoder'])

    class DistRegularization(ParameterizeDecoderBatch):
        """Try requiring pairwise distances are proportional"""

        def __init__(self, *args, dist_loss_weight=1., **kwargs):
            self.dist_loss_weight = dist_loss_weight
            super().__init__(*args, **kwargs)

        def _name_prefix(self):
            return 'dist_reg'

        def _loss_weight_name(self):
            return "dist_{}".format(self.dist_loss_weight)

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z, A, v, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)

            # hypothesize that close in z iff close in A, v space
            # TODO maybe only require that things close in z should be close in A, v space (not bidirectional)
            # TODO try distance instead of cosine/directionality
            di = torch.nn.functional.pdist(z)
            # dA = torch.nn.functional.pdist(A.view(A.shape[0], -1))
            do = torch.nn.functional.pdist(v)
            dist_loss = 1 - torch.nn.functional.cosine_similarity(di, do, dim=0)
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss, dist_loss

        @staticmethod
        def loss_names():
            return "mse_loss", "dist_loss"

        def _reduce_losses(self, losses):
            return torch.mean(losses[0]) + self.dist_loss_weight * torch.mean(losses[1])

    class CompressionReward(ParameterizeDecoderBatch):
        """Reward mapping large differences in x,u space to small differences in v space"""

        def __init__(self, *args, compression_loss_weight=1e-3, dist_regularization=1e-8, **kwargs):
            self.compression_loss_weight = compression_loss_weight
            # avoid log(0)
            self.dist_regularization = dist_regularization
            super().__init__(*args, **kwargs)

        def _name_prefix(self):
            return 'reward_compression'

        def _loss_weight_name(self):
            return "compress_{}".format(self.compression_loss_weight)

        def _evaluate_batch_apply_tsf(self, X, U, tsf):
            assert tsf is TransformToUse.LATENT_SPACE
            z = self.xu_to_z(X, U)
            A = self.linear_dynamics(z)
            v = linalg.batch_batch_product(z, A.transpose(-1, -2))
            yhat = self.get_dx(X, v)
            return z, A, v, yhat

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z, A, v, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)
            compression_r = compression_reward(v, torch.cat((X, U), dim=1),
                                               dist_regularization=self.dist_regularization,
                                               top_percent=0.05)
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss, -compression_r

        @staticmethod
        def loss_names():
            return "mse_loss", "compression_loss"

        def _reduce_losses(self, losses):
            return torch.mean(losses[0]) + self.compression_loss_weight * torch.mean(losses[1])

    class PartialPassthrough(CompressionReward):
        """Don't pass through all of x to g, since it could just learn to map v to u"""

        def __init__(self, ds, device, *args, nv=4, **kwargs):
            config = load_data.DataConfig()
            config.nx = 2
            config.ny = nv * ds.config.nx  # matrix output (original nx, ignore sincos)
            self.partial_decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(16, 32)).to(device=device))
            super().__init__(ds, device, *args, nv=nv, **kwargs)

        def _name_prefix(self):
            return 'partial_passthrough'

        def parameters(self):
            return list(self.partial_decoder.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['pd'] = self.partial_decoder.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.partial_decoder.model.load_state_dict(saved_state_dict['pd'])

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

            B, nx = x.shape
            angle_index = 2
            s = torch.sin(x[:, angle_index]).view(-1, 1)
            c = torch.cos(x[:, angle_index]).view(-1, 1)
            x = torch.cat((s, c), dim=1)
            linear_tsf = self.partial_decoder.sample(x).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

    class PartitionBlock(torch.nn.Module):
        def __init__(self, input_dim, output_dim, split_at):
            super().__init__()
            self.split_at = split_at
            self.linear = torch.nn.Parameter(torch.randn((input_dim, output_dim)), requires_grad=True)
            # self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

        def forward(self, x, split):
            # require that a single dimension is split up (can't have multiple copies of same element)
            # each column should sum to 1
            weights = torch.nn.functional.softmax(self.linear, dim=0)
            y = x @ weights
            if split is 0:
                y = y[:, :self.split_at]
            else:
                y = y[:, self.split_at:]
            return y

    class LearnedPartialPassthrough(CompressionReward):
        """Don't pass through all of x to g; learn which parts to pass to g and which to h"""

        def __init__(self, ds, device, *args, nz=4, nv=4, reduced_decoder_input_dim=2, **kwargs):
            self.reduced_decoder_input_dim = reduced_decoder_input_dim
            self.x_extractor = torch.nn.Linear(ds.config.nx, self.reduced_decoder_input_dim).to(device=device,
                                                                                                dtype=torch.double)

            config = load_data.DataConfig()
            config.nx = self.reduced_decoder_input_dim
            config.ny = nv * ds.config.nx
            self.partial_decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(16, 32)).to(device=device))

            super().__init__(ds, device, *args, nz=nz, nv=nv, **kwargs)

        def _name_prefix(self):
            return 'extract_passthrough_{}'.format(self.reduced_decoder_input_dim)

        def parameters(self):
            return list(self.partial_decoder.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.linear_model_producer.model.parameters()) + list(self.x_extractor.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['pd'] = self.partial_decoder.model.state_dict()
            d['extract'] = self.x_extractor.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.partial_decoder.model.load_state_dict(saved_state_dict['pd'])
            self.x_extractor.load_state_dict(saved_state_dict['extract'])

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

            B, nx = x.shape
            extracted_from_x = self.x_extractor(x)
            linear_tsf = self.partial_decoder.sample(extracted_from_x).view(B, nx, self.nv)
            dx = linalg.batch_batch_product(v, linear_tsf)
            return dx

    class GroundTruthWithCompression(CompressionReward):
        """Ground truth coordinate transform with only f learned"""

        def _name_prefix(self):
            return 'ground_truth'

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            # (along, d_along, push magnitude)
            z = torch.cat((state[:, 3].view(-1, 1), action), dim=1)
            return z

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

            yaw = x[:, 2]
            N = v.shape[0]
            dx = torch.zeros((N, 4), dtype=v.dtype, device=v.device)
            # convert (dx, dy) from body frame back to world frame
            dx[:, :2] = math_utils.batch_rotate_wrt_origin(v[:, :2], yaw)
            dx[:, 2:] = v[:, 2:]
            return dx

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z, A, v, yhat = self._evaluate_batch_apply_tsf(X, U, tsf)
            # we know what v actually is based on y; train the network to output the actual v
            v = self.get_v(X, Y, z)

            # compression reward should be a constant
            compression_r = compression_reward(v, torch.cat((X, U), dim=1),
                                               dist_regularization=self.dist_regularization,
                                               top_percent=0.05)
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss, -compression_r

        def parameters(self):
            return list(self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = {'linear': self.linear_model_producer.model.state_dict()}
            return d

        def _load_model_state_dict(self, saved_state_dict):
            self.linear_model_producer.model.load_state_dict(saved_state_dict['linear'])


class AblationOnTransform:
    class RelaxDecoderLinearity(LearnedTransform.LinearComboLatentInput, invariant.LearnFromBatchTransform):
        """Don't require that g_rho output a linear transformation of v"""

        def __init__(self, ds, device, nv=4, **kwargs):
            config = load_data.DataConfig()
            config.nx = ds.config.nx + nv
            config.ny = ds.config.nx
            # directly decode v and x to dx
            self.decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
            super().__init__(ds, device, nv=nv, **kwargs)

        def _name_prefix(self):
            return 'ablation_remove_decoder_linear'

        def parameters(self):
            return list(self.decoder.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.linear_model_producer.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['decoder'] = self.decoder.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.decoder.model.load_state_dict(saved_state_dict['decoder'])

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

            decoder_input = torch.cat((x, v), dim=1)
            dx = self.decoder.sample(decoder_input)
            return dx

        def learn_model(self, max_epoch, batch_N=500):
            return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)

    class RelaxLinearLocalDynamics(LearnedTransform.ParameterizeDecoder, invariant.LearnFromBatchTransform):
        """Don't require that f_psi output a linear transformation of z; instead allow it to output v directly"""

        def __init__(self, ds, device, nz=4, nv=4, **kwargs):
            config = load_data.DataConfig()
            # directly learn dynamics from z to v
            config.nx = nz
            config.ny = nv
            self.dynamics = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
            super().__init__(ds, device, nz=nz, nv=nv, **kwargs)

        def _name_prefix(self):
            return 'ablation_remove_dynamics_linear'

        def get_v(self, x, dx, z):
            v = self.dynamics.sample(z)
            return v

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            assert tsf is TransformToUse.LATENT_SPACE
            z = self.xu_to_z(X, U)
            v = self.get_v(X, Y, z)
            yhat = self.get_dx(X, v)

            # mse loss
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss,

        @staticmethod
        def loss_names():
            return "mse_loss",

        def _reduce_losses(self, losses):
            return torch.mean(losses[0])

        def parameters(self):
            return list(self.linear_decoder_producer.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.dynamics.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['dynamics'] = self.dynamics.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.dynamics.model.load_state_dict(saved_state_dict['dynamics'])

        def linear_dynamics(self, z):
            raise RuntimeError("Shouldn't be calling this; instead should transform z to v directly")

        def learn_model(self, max_epoch, batch_N=500):
            return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)

    class RelaxLinearDynamicsAndLinearDecoder(LearnedTransform.LinearComboLatentInput,
                                              invariant.LearnFromBatchTransform):
        """Combine the previous 2 ablations"""

        def __init__(self, ds, device, nz=4, nv=4, **kwargs):
            config = load_data.DataConfig()
            # directly learn dynamics from z to v
            config.nx = nz
            config.ny = nv
            self.dynamics = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))

            config = load_data.DataConfig()
            config.nx = ds.config.nx + nv
            config.ny = ds.config.nx
            # directly decode v and x to dx
            self.decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
            super().__init__(ds, device, nz=nz, nv=nv, **kwargs)

        def _name_prefix(self):
            return 'ablation_remove_all_linear'

        def parameters(self):
            return list(self.decoder.model.parameters()) + list(self.z_selector.parameters()) + list(
                self.dynamics.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['decoder'] = self.decoder.model.state_dict()
            d['dynamics'] = self.dynamics.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.decoder.model.load_state_dict(saved_state_dict['decoder'])
            self.dynamics.model.load_state_dict(saved_state_dict['dynamics'])

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                x = x.view(1, -1)
                v = v.view(1, -1)

            decoder_input = torch.cat((x, v), dim=1)
            dx = self.decoder.sample(decoder_input)
            return dx

        def get_v(self, x, dx, z):
            v = self.dynamics.sample(z)
            return v

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            assert tsf is TransformToUse.LATENT_SPACE
            z = self.xu_to_z(X, U)
            v = self.get_v(X, Y, z)
            yhat = self.get_dx(X, v)

            # mse loss
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss,

        @staticmethod
        def loss_names():
            return "mse_loss",

        def _reduce_losses(self, losses):
            return torch.mean(losses[0])

        def linear_dynamics(self, z):
            raise RuntimeError("Shouldn't be calling this; instead should transform z to v directly")

        def learn_model(self, max_epoch, batch_N=500):
            return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)

    class RelaxEncoder(RelaxLinearDynamicsAndLinearDecoder):
        """Relax the parameterization of z encoding"""

        def __init__(self, ds, device, nz=4, **kwargs):
            config = load_data.DataConfig()
            # directly learn dynamics from z to v
            config.nx = ds.config.nx + ds.config.nu
            config.ny = nz
            self.encoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))

            super().__init__(ds, device, nz=nz, **kwargs)

        def _name_prefix(self):
            return 'ablation_relax_encoder'

        def xu_to_z(self, state, action):
            if len(state.shape) < 2:
                state = state.reshape(1, -1)
                action = action.reshape(1, -1)

            xu = torch.cat((state, action), dim=1)
            z = self.encoder.sample(xu)
            return z

        def parameters(self):
            return list(self.decoder.model.parameters()) + list(self.encoder.model.parameters()) + list(
                self.dynamics.model.parameters())

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['encoder'] = self.encoder.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.encoder.model.load_state_dict(saved_state_dict['encoder'])

    class UseDecoderForDynamics(RelaxEncoder):
        """Remove v and pass z directly to the decoder"""

        def __init__(self, ds, device, nz=4, **kwargs):
            super().__init__(ds, device, nz=nz, nv=ds.config.ny, **kwargs)

        def _name_prefix(self):
            return 'ablation_direct_no_v'

        def get_v(self, x, dx, z):
            return self.get_dx(x, z)

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            assert tsf is TransformToUse.LATENT_SPACE
            z = self.xu_to_z(X, U)
            # use z instead of v
            yhat = self.get_dx(X, z)

            # mse loss
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss,

        def parameters(self):
            return list(self.decoder.model.parameters()) + list(self.encoder.model.parameters())

    class NoPassthrough(UseDecoderForDynamics):
        """Remove x from decoder input (only pass x to decoder)"""

        def __init__(self, ds, device, nz=4, **kwargs):
            config = load_data.DataConfig()
            config.nx = nz
            config.ny = ds.config.nx
            # directly decode v and x to dx
            self.direct_decoder = model.DeterministicUser(
                make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
            super().__init__(ds, device, nz=nz, **kwargs)

        def _name_prefix(self):
            return 'ablation_no passthrough'

        def _model_state_dict(self):
            d = super()._model_state_dict()
            d['d2'] = self.direct_decoder.model.state_dict()
            return d

        def _load_model_state_dict(self, saved_state_dict):
            super()._load_model_state_dict(saved_state_dict)
            self.direct_decoder.model.load_state_dict(saved_state_dict['d2'])

        def get_dx(self, x, v):
            if len(x.shape) == 1:
                v = v.view(1, -1)

            dx = self.direct_decoder.sample(v)
            return dx

        def parameters(self):
            return list(self.decoder.model.parameters()) + list(self.encoder.model.parameters())

    class ReguarlizePairwiseDistances(RelaxLinearDynamicsAndLinearDecoder):
        """Try requiring pairwise distances are proportional"""

        def __init__(self, *args, compression_loss_weight=1e-3, **kwargs):
            self.compression_loss_weight = compression_loss_weight
            super().__init__(*args, **kwargs)

        def _name_prefix(self):
            return 'ablation_reg'

        def _loss_weight_name(self):
            return "compress_{}".format(self.compression_loss_weight)

        def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
            z = self.xu_to_z(X, U)
            v = self.get_v(X, Y, z)
            yhat = self.get_dx(X, v)
            compression_r = compression_reward(v, torch.cat((X, U), dim=1), dist_regularization=1e-8,
                                               top_percent=0.05)
            mse_loss = torch.norm(yhat - Y, dim=1)
            return mse_loss, -compression_r

        @staticmethod
        def loss_names():
            return "mse_loss", "compression_loss"

        def _reduce_losses(self, losses):
            return torch.mean(losses[0]) + self.compression_loss_weight * torch.mean(losses[1])
