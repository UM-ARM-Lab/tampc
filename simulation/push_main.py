import logging
import math
import typing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from tensorboardX import SummaryWriter

from meta_contact import cfg, invariant
from meta_contact import model
from meta_contact import online_model
from meta_contact import prior
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.controller import global_controller
from meta_contact.env import block_push

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def random_touching_start(env, w=block_push.DIST_FOR_JUST_TOUCHING):
    init_block_pos = (np.random.random((2,)) - 0.5)
    init_block_yaw = (np.random.random() - 0.5) * 2 * math.pi
    # randomly initialize pusher adjacent to block
    # choose which face we will be next to
    env_type = type(env)
    if env_type == block_push.PushAgainstWallEnv:
        along_face = (np.random.random() - 0.5) * 2
        face = np.random.randint(0, 4)
        init_pusher = block_push.pusher_pos_for_touching(init_block_pos, init_block_yaw, from_center=w,
                                                         face=face,
                                                         along_face=along_face)
    elif env_type == block_push.PushAgainstWallStickyEnv or env_type == block_push.PushWithForceDirectlyEnv:
        init_pusher = np.random.uniform(-1, 1)
    else:
        raise RuntimeError("Unrecognized env type")
    return init_block_pos, init_block_yaw, init_pusher


# have to be set after selecting an environment
env_dir = None


def collect_touching_freespace_data(trials=20, trial_length=40, level=0):
    env = get_easy_env(p.DIRECT, level)
    u_min, u_max = env.get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    # use mode p.GUI to see what the trials look like
    save_dir = '{}{}'.format(env_dir, level)
    sim = block_push.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                     stop_when_done=False, save_dir=save_dir)
    rand.seed(4)
    # randomly distribute data
    for _ in range(trials):
        seed = rand.seed()
        # start at fixed location
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(env)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_easy_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_block_pos = [0, 0]
    init_block_yaw = -1
    init_pusher = 0
    goal_pos = [-0.3, 0.3]
    # env = interactive_block_pushing.PushAgainstWallEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher,
    #                                                    init_block=init_block_pos, init_yaw=init_block_yaw,
    #                                                    environment_level=level)
    # env_dir = 'pushing/no_restriction'
    # env = block_push.PushAgainstWallStickyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher, log_video=log_video,
    #                                           init_block=init_block_pos, init_yaw=init_block_yaw,
    #                                           environment_level=level)
    # env_dir = 'pushing/sticky'
    env = block_push.PushWithForceDirectlyEnv(mode=mode, goal=goal_pos, init_pusher=init_pusher, log_video=log_video,
                                              init_block=init_block_pos, init_yaw=init_block_yaw,
                                              environment_level=level)
    env_dir = 'pushing/direct_force'
    return env


def get_free_space_env_init(seed=1, **kwargs):
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(kwargs.pop('mode', p.DIRECT), **kwargs)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))
    return d, env, config, ds


def get_preprocessor_for_invariant_tsf(invariant_tsf):
    # evaluate tsf on validation set
    losses = invariant_tsf.evaluate_validation(None)
    logger.info("tsf on validation %s",
                "  ".join(
                    ["{} {:.5f}".format(name, loss.mean().cpu().item()) if loss is not None else "" for name, loss in
                     zip(invariant_tsf.loss_names(), losses)]))

    # wrap the transform as a data preprocessor
    preprocessor = preprocess.Compose(
        [invariant.InvariantTransformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    return preprocessor


class PusherNetwork(model.NetworkModelWrapper):
    """Network wrapper with some special validation evaluation"""

    def evaluate_validation(self):
        with torch.no_grad():
            XUv, _, _ = self.ds.original_validation_set()
            # try validation loss outside of our training region (by translating input)
            for d in [4, 10]:
                for trans in [[1, 1], [-1, 1], [-1, -1]]:
                    dd = (trans[0] * d, trans[1] * d)
                    XU = torch.cat(
                        (XUv[:, :2] + torch.tensor(dd, device=XUv.device, dtype=XUv.dtype),
                         XUv[:, 2:]),
                        dim=1)
                    if self.ds.preprocessor is not None:
                        XU = self.ds.preprocessor.transform_x(XU)
                    vloss = self.user.compute_validation_loss(XU, self.Yv, self.ds)
                    self.writer.add_scalar("loss/validation_{}_{}".format(dd[0], dd[1]), vloss.mean(),
                                           self.step)


def constrain_state(state):
    # yaw gets normalized
    state[:, 2] = math_utils.angle_normalize(state[:, 2])
    # along gets constrained
    state[:, 3] = math_utils.clip(state[:, 3], torch.tensor(-1, dtype=torch.double, device=state.device),
                                  torch.tensor(1, dtype=torch.double, device=state.device))
    return state


# ------- Hand designed coordinate transform classes using architecture 2
class HandDesignedCoordTransform(invariant.InvariantTransform):
    def __init__(self, ds, nz, **kwargs):
        # v is dx, dy, dyaw in body frame and d_along
        super().__init__(ds, nz, 4, name='coord', **kwargs)

    def get_v(self, x, dx, z):
        return self.dx_to_v(x, dx)

    def dx_to_v(self, x, dx):
        if len(x.shape) == 1:
            x = x.view(1, -1)
            dx = dx.view(1, -1)
        N = dx.shape[0]
        v = torch.zeros((N, self.nv), dtype=dx.dtype, device=dx.device)
        # convert world frame to body frame
        v[:, :2] = math_utils.batch_rotate_wrt_origin(dx[:, :2], -x[:, 2])
        # second last element is dyaw, which also gets passed along directly
        v[:, 2] = dx[:, 2]
        # last element is d_along, which gets passed along directly
        v[:, 3] = dx[:, 3]
        return v

    def get_dx(self, x, v):
        if len(x.shape) == 1:
            x = x.view(1, -1)
            v = v.view(1, -1)
        N = v.shape[0]
        dx = torch.zeros((N, 4), dtype=v.dtype, device=v.device)
        # convert (dx, dy) from body frame back to world frame
        dx[:, :2] = math_utils.batch_rotate_wrt_origin(v[:, :2], x[:, 2])
        # second last element is dyaw, which also gets passed along directly
        dx[:, 2] = v[:, 2]
        # last element is d_along, which gets passed along directly
        dx[:, 3] = v[:, 3]
        return dx

    def parameters(self):
        return [torch.zeros(1)]

    def _model_state_dict(self):
        return None

    def _load_model_state_dict(self, saved_state_dict):
        pass

    def learn_model(self, max_epoch, batch_N=500):
        pass


class WorldBodyFrameTransformForStickyEnv(HandDesignedCoordTransform):
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

    def xu_to_z(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # (along, d_along, push magnitude)
        z = torch.from_numpy(np.column_stack((state[:, -1], action)))
        return z


class WorldBodyFrameTransformForDirectPush(HandDesignedCoordTransform):
    def __init__(self, ds, **kwargs):
        # need along, d_along, push magnitude, and push direction; don't need block position or yaw
        super().__init__(ds, 4, **kwargs)

    def xu_to_z(self, state, action):
        if len(state.shape) < 2:
            state = state.reshape(1, -1)
            action = action.reshape(1, -1)

        # (along, d_along, push magnitude, push direction)
        # z = torch.from_numpy(np.column_stack((state[:, -1], action)))
        z = torch.cat((state[:, -1].view(-1, 1), action), dim=1)
        return z


# ------- Learned transform classes using architecture 3
class ParameterizedCoordTransform(invariant.LearnLinearDynamicsTransform):
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
        super().__init__(ds, nz, nv, name='{}_{}'.format(self._name_prefix(), name), **kwargs)

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
        losses = super(ParameterizedCoordTransform, self).evaluate_validation(writer)
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


class Parameterized2Transform(ParameterizedCoordTransform):
    """Relax parameterization structure to allow (each dimension of) z to be some linear combination of x,u"""

    def __init__(self, ds, device, nz=4, **kwargs):
        # input is x, output is z
        # constrain output to 0 and 1
        self.z_selector = torch.nn.Linear(ds.config.nx + ds.config.nu, nz, bias=False).to(device=device,
                                                                                          dtype=torch.double)
        self.true_z_param = torch.tensor(
            [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], device=device,
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


class Parameterized3Transform(Parameterized2Transform):
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


class Parameterized3BatchTransform(Parameterized3Transform, invariant.LearnFromBatchTransform):
    """Train using randomized neighbourhoods instead of fixed given ones"""

    def learn_model(self, max_epoch, batch_N=500):
        return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)


class LinearRelaxEncoderTransform(Parameterized3BatchTransform):
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


from meta_contact.invariant import TransformToUse


class DistRegularizedTransform(Parameterized3BatchTransform):
    """Try requiring pairwise distances are proportional"""

    def __init__(self, *args, dist_loss_weight=1., **kwargs):
        self.dist_loss_weight = dist_loss_weight
        super(DistRegularizedTransform, self).__init__(*args, **kwargs)

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
        return torch.sum(losses[0]) + self.dist_loss_weight * torch.sum(losses[1])


# ------- Ablations on architecture 3
class AblationRemoveLinearDecoderTransform(Parameterized2Transform, invariant.LearnFromBatchTransform):
    """Don't require that g_rho output a linear transformation of v"""

    def __init__(self, ds, device, nv=4, **kwargs):
        config = load_data.DataConfig()
        config.nx = ds.config.nx + nv
        config.ny = ds.config.nx
        # directly decode v and x to dx
        self.decoder = model.DeterministicUser(make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
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


class AblationRemoveLinearDynamicsTransform(Parameterized3Transform, invariant.LearnFromBatchTransform):
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
        return torch.sum(losses[0])

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


class AblationRemoveAllLinearityTransform(Parameterized2Transform, invariant.LearnFromBatchTransform):
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
        self.decoder = model.DeterministicUser(make.make_sequential_network(config, h_units=(32, 32)).to(device=device))
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
        return torch.sum(losses[0])

    def linear_dynamics(self, z):
        raise RuntimeError("Shouldn't be calling this; instead should transform z to v directly")

    def learn_model(self, max_epoch, batch_N=500):
        return invariant.LearnFromBatchTransform.learn_model(self, max_epoch, batch_N)


class AblationRelaxEncoderTransform(AblationRemoveAllLinearityTransform):
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


class AblationDirectDynamics(AblationRelaxEncoderTransform):
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


class AblationNoPassthrough(AblationDirectDynamics):
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


class AblationAllRegularizedTransform(AblationRemoveAllLinearityTransform):
    """Try requiring pairwise distances are proportional"""

    def __init__(self, *args, dist_loss_weight=1., **kwargs):
        self.dist_loss_weight = dist_loss_weight
        super(AblationAllRegularizedTransform, self).__init__(*args, **kwargs)

    def _name_prefix(self):
        return 'ablation_dist_reg'

    def _loss_weight_name(self):
        return "dist_{}".format(self.dist_loss_weight)

    def _evaluate_batch(self, X, U, Y, weights=None, tsf=TransformToUse.LATENT_SPACE):
        z = self.xu_to_z(X, U)
        v = self.get_v(X, Y, z)
        yhat = self.get_dx(X, v)

        dz = torch.nn.functional.pdist(z)
        dv = torch.nn.functional.pdist(v)
        dist_loss = 1 - torch.nn.functional.cosine_similarity(dz, dv, dim=0)
        mse_loss = torch.norm(yhat - Y, dim=1)
        return mse_loss, dist_loss

    @staticmethod
    def loss_names():
        return "mse_loss", "dist_loss"

    def _reduce_losses(self, losses):
        return torch.sum(losses[0]) + self.dist_loss_weight * torch.sum(losses[1])


def coord_tsf_factory(env, *args, **kwargs):
    tsfs = {block_push.PushAgainstWallStickyEnv: WorldBodyFrameTransformForStickyEnv,
            block_push.PushWithForceDirectlyEnv: WorldBodyFrameTransformForDirectPush}
    tsf_type = tsfs.get(type(env), None)
    if tsf_type is None:
        raise RuntimeError("No tsf specified for env type {}".format(type(env)))
    return tsf_type(*args, **kwargs)


def verify_coordinate_transform():
    def get_dx(px, cx):
        dpos = cx[:2] - px[:2]
        dyaw = math_utils.angular_diff(cx[2], px[2])
        dalong = cx[3] - px[3]
        dx = torch.from_numpy(np.r_[dpos, dyaw, dalong])
        return dx

    # comparison tolerance
    tol = 2e-4
    env = get_easy_env(p.GUI)
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config)

    tsf = coord_tsf_factory(env, ds)

    along = 0.7
    init_block_pos = [0, 0]
    init_block_yaw = 0
    env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=along)
    action = np.array([0, 0.4, 0])
    # push with original yaw (try this N times to confirm that all pushes are consistent)
    N = 10
    px, dx = None, None
    dxes = torch.zeros((N, env.ny))
    for i in range(N):
        px = env.reset()
        cx, _, _, _ = env.step(action)
        # get actual difference dx
        dx = get_dx(px, cx)
        dxes[i] = dx
    assert torch.allclose(dxes.std(0), torch.zeros(env.ny))
    assert px is not None
    # get input in latent space
    px = torch.from_numpy(px)
    z = tsf.xu_to_z(px, action)
    # try inverting the transforms
    v_1 = tsf.dx_to_v(px, dx)
    dx_inverted = tsf.get_dx(px, v_1)
    assert torch.allclose(dx, dx_inverted)
    # same push with yaw, should result in the same z and the dx should give the same v but different dx

    N = 16
    dxes = torch.zeros((N, env.ny))
    vs = torch.zeros((N, env.ny))
    # for i, yaw_shift in enumerate(np.linspace(0, math.pi*2, 4)):
    for i, yaw_shift in enumerate(np.linspace(0, math.pi * 2, N)):
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw + yaw_shift, init_pusher=along)
        px = env.reset()
        cx, _, _, _ = env.step(action)
        # get actual difference dx
        dx = get_dx(px, cx)
        px = torch.from_numpy(px)
        z_2 = tsf.xu_to_z(px, action)
        assert torch.allclose(z, z_2, atol=tol / 10)
        v_2 = tsf.dx_to_v(px, dx)
        vs[i] = v_2
        dxes[i] = dx
        dx_inverted_2 = tsf.get_dx(px, v_2)
        assert torch.allclose(dx, dx_inverted_2)
    # change in body frame should be exactly the same
    logger.info(vs)
    # relative standard deviation
    logger.info(vs.std(0) / torch.abs(vs.mean(0)))
    assert torch.allclose(vs.std(0), torch.zeros(4), atol=tol)
    # actual dx should be different since we have yaw
    assert not torch.allclose(dxes.std(0), torch.zeros(4), atol=tol)


def test_online_model():
    seed = 1
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(p.DIRECT, level=0)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(0), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))

    invariant_tsf = coord_tsf_factory(env, ds)
    transformer = invariant.InvariantTransformer
    preprocessor = preprocess.Compose(
        [transformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    ds.update_preprocessor(preprocessor)

    prior_name = 'coord_prior'

    mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds, name=prior_name)

    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=600)

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, env.state_difference, local_mix_weight=0, sigreg=1e-10)

    # evaluate linearization by comparing error from applying model directly vs applying linearized model
    xuv, yv, _ = ds.original_validation_set()
    xv = xuv[:, :ds.original_config().nx]
    uv = xuv[:, ds.original_config().nx:]
    if ds.original_config().predict_difference:
        yv = yv + xv
    # full model prediction
    yhat1 = pm.dyn_net.predict(xuv)
    # linearized prediction
    yhat2 = dynamics.predict(None, None, xv, uv)

    e1 = torch.norm((yhat1 - yv), dim=1)
    e2 = torch.norm((yhat2 - yv), dim=1)
    assert torch.allclose(yhat1, yhat2)
    logger.info("Full model MSE %f linearized model MSE %f", e1.mean(), e2.mean())

    mix_weights = [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
    errors = torch.zeros(len(mix_weights))
    divergence = torch.zeros_like(errors)
    # debug updating linear model and using non-0 weight (error should increase with distance from update)
    for i, weight in enumerate(mix_weights):
        dynamics.empsig_N = weight
        yhat2 = dynamics.predict(None, None, xv, uv)
        errors[i] = torch.mean(torch.norm((yhat2 - yv), dim=1))
        divergence[i] = torch.mean(torch.norm((yhat2 - yhat1), dim=1))
    logger.info("error with increasing weight %s", errors)
    logger.info("divergence increasing weight %s", divergence)

    # use just a single trajectory
    N = 49  # xv.shape[0]-1
    xv = xv[:N]
    uv = uv[:N]
    yv = yv[:N]

    horizon = 3
    dynamics.gamma = 0.1
    dynamics.empsig_N = 1.0
    compare_against_last_updated = False
    errors = torch.zeros((N, 3))
    GLOBAL = 0
    BEFORE = 1
    AFTER = 2

    yhat2 = dynamics.predict(None, None, xv, uv)
    e2 = torch.norm((yhat2 - yv), dim=1)
    for i in range(N - 1):
        if compare_against_last_updated:
            yhat2 = dynamics.predict(None, None, xv, uv)
            e2 = torch.norm((yhat2 - yv), dim=1)
        dynamics.update(xv[i], uv[i], xv[i + 1])
        # after
        yhat3 = dynamics.predict(None, None, xv, uv)
        e3 = torch.norm((yhat3 - yv), dim=1)

        errors[i, GLOBAL] = e1[i + 1:i + 1 + horizon].mean()
        errors[i, BEFORE] = e2[i + 1:i + 1 + horizon].mean()
        errors[i, AFTER] = e3[i + 1:i + 1 + horizon].mean()
        # when updated with xux' from ground truth, should do better at current location
        if errors[i, AFTER] > errors[i, BEFORE]:
            logger.warning("error increased after update at %d", i)
        # also look at error with global model
        logger.info("global before after %s", errors[i])

    errors = errors[:N - 1]
    # plot these two errors relative to the global model error
    plt.figure()
    plt.plot(errors[:, BEFORE] / errors[:, GLOBAL])
    plt.plot(errors[:, AFTER] / errors[:, GLOBAL])
    plt.title(
        'local error after update for horizon {} gamma {} weight {}'.format(horizon, dynamics.gamma, dynamics.empsig_N))
    plt.xlabel('step')
    plt.ylabel('relative error to global model')
    plt.yscale('log')
    plt.legend(['before update', 'after update'])
    plt.grid()
    plt.show()


class UseTransform:
    NO_TRANSFORM = 0
    COORDINATE_TRANSFORM = 1
    PARAMETERIZED_1 = 2
    PARAMETERIZED_2 = 3
    PARAMETERIZED_3 = 4
    PARAMETERIZED_4 = 5
    PARAMETERIZED_3_BATCH = 6
    PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER = 7
    PARAMETERIZED_ABLATE_NO_V = 8


def test_dynamics(level=0, use_tsf=UseTransform.COORDINATE_TRANSFORM, relearn_dynamics=False, online_adapt=True):
    seed = 1
    plot_model_error = False
    enforce_model_rollout = False
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = get_easy_env(p.GUI, level=level, log_video=True)

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=get_data_dir(level), validation_ratio=0.1, config=config, device=d)

    logger.info("initial random seed %d", rand.seed(seed))

    # add in invariant transform here
    transforms = {UseTransform.NO_TRANSFORM: None,
                  UseTransform.COORDINATE_TRANSFORM: coord_tsf_factory(env, ds),
                  UseTransform.PARAMETERIZED_1: ParameterizedCoordTransform(ds, d, too_far_for_neighbour=0.3,
                                                                            name="_s2"),
                  UseTransform.PARAMETERIZED_2: Parameterized2Transform(ds, d, too_far_for_neighbour=0.3,
                                                                        name="rand_start_s9"),
                  UseTransform.PARAMETERIZED_3: Parameterized3Transform(ds, d, too_far_for_neighbour=0.3, name="_s9"),
                  UseTransform.PARAMETERIZED_4: Parameterized3Transform(ds, d, too_far_for_neighbour=0.3,
                                                                        name="sincos_s2", use_sincos_angle=True),
                  UseTransform.PARAMETERIZED_3_BATCH: Parameterized3BatchTransform(ds, d, name="_s1"),
                  UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER: AblationRelaxEncoderTransform(ds, d,
                                                                                                                name="_s0"),
                  UseTransform.PARAMETERIZED_ABLATE_NO_V: AblationDirectDynamics(ds, d, name="_s3"),
                  }
    transform_names = {UseTransform.NO_TRANSFORM: 'none',
                       UseTransform.COORDINATE_TRANSFORM: 'coord',
                       UseTransform.PARAMETERIZED_1: 'param1',
                       UseTransform.PARAMETERIZED_2: 'param2',
                       UseTransform.PARAMETERIZED_3: 'param3',
                       UseTransform.PARAMETERIZED_4: 'param4',
                       UseTransform.PARAMETERIZED_3_BATCH: 'param3_batch',
                       UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER: 'ablate_linear_relax_encoder',
                       UseTransform.PARAMETERIZED_ABLATE_NO_V: 'ablate_no_v',
                       }
    invariant_tsf = transforms[use_tsf]

    if invariant_tsf:
        # load transform (only 1 function for learning transform reduces potential for different learning params)
        if use_tsf is not UseTransform.COORDINATE_TRANSFORM and not invariant_tsf.load(
                invariant_tsf.get_last_checkpoint()):
            raise RuntimeError("Transform {} should be learned before using".format(invariant_tsf.name))

        preprocessor = get_preprocessor_for_invariant_tsf(invariant_tsf)
    else:
        # use minmax scaling if we're not using an invariant transform (baseline)
        preprocessor = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
        # preprocessor = preprocess.Compose([preprocess.PytorchTransformer(preprocess.AngleToCosSinRepresentation(2),
        #                                                                  preprocess.NullSingleTransformer()),
        #                                    preprocess.PytorchTransformer(preprocess.MinMaxScaler())])
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)

    mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                       name="dynamics_{}".format(transform_names[use_tsf]))

    pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                                 train_epochs=600)
    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    # pm = prior.NoPrior()

    # test that the model predictions are relatively symmetric for positive and negative along
    if isinstance(env, block_push.PushAgainstWallStickyEnv) and isinstance(pm, prior.NNPrior):
        N = 5
        x_top = torch.tensor([0, 0, 0, 1], dtype=torch.double, device=d).repeat(N, 1)
        x_bot = torch.tensor([0, 0, 0, -1], dtype=torch.double, device=d).repeat(N, 1)
        # push straight
        u = torch.tensor([0, 1, 0], dtype=torch.double, device=d)
        # do rollouts
        for i in range(N - 1):
            x_top[i + 1] = mw.predict(torch.cat((x_top[i], u)).view(1, -1))
            x_bot[i + 1] = mw.predict(torch.cat((x_bot[i], u)).view(1, -1))
        try:
            # check sign of the last states
            x = x_top[N - 1]
            assert x[0] > 0
            assert x[2] < 0  # yaw decreased (rotated ccw)
            assert abs(x[3] - x_top[0, 3]) < 0.1  # along hasn't changed much
            x = x_bot[N - 1]
            assert x[0] > 0
            assert x[2] > 0  # yaw increased (rotated cw)
            assert abs(x[3] - x_bot[0, 3]) < 0.1  # along hasn't changed much
        except AssertionError as e:
            # either fail or just warn that it's an error
            if enforce_model_rollout:
                raise e
            else:
                logger.error(e)
                logger.info(x_top)

    # plot model prediction
    if plot_model_error:
        XU, Y, _ = ds.validation_set()
        Y = Y.cpu().numpy()
        Yhatn = mw.user.sample(XU).cpu().detach().numpy()
        E = Yhatn - Y
        # relative error (compared to the mean magnitude)
        Er = E / np.mean(np.abs(Y), axis=0)
        for i in range(config.ny):
            plt.subplot(4, 2, 2 * i + 1)
            plt.plot(Y[:, i])
            plt.ylabel("$y_{}$".format(i))
            plt.subplot(4, 2, 2 * i + 2)
            plt.plot(Er[:, i])
            # plt.plot(E[:, i])
            plt.ylabel("$e_{}$".format(i))
        plt.show()

    u_min, u_max = env.get_control_bounds()
    Q = torch.diag(torch.tensor([10, 10, 0, 0.01], dtype=torch.double))
    R = 0.01
    # tune this so that we figure out to make u-turns
    sigma = torch.ones(env.nu, dtype=torch.double, device=d) * 0.2
    sigma[1] = 0.5
    mpc_opts = {
        'num_samples': 10000,
        'noise_sigma': torch.diag(sigma),
        'noise_mu': torch.tensor([0, 0.5, 0], dtype=torch.double, device=d),
        'lambda_': 1,
        'horizon': 35,
        'u_init': torch.tensor([0, 0.5, 0], dtype=torch.double, device=d),
        'sample_null_action': False,
    }
    if online_adapt:
        dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, env.state_difference, local_mix_weight=1.0,
                                                    sigreg=1e-10)
        ctrl = online_controller.OnlineMPPI(dynamics, untransformed_config, Q=Q.numpy(), R=R, u_min=u_min, u_max=u_max,
                                            compare_to_goal=env.state_difference,
                                            constrain_state=constrain_state,
                                            device=d, mpc_opts=mpc_opts)
    else:
        ctrl = global_controller.GlobalMPPI(mw, untransformed_config, Q=Q, R=R, u_min=u_min, u_max=u_max,
                                            compare_to_goal=env.state_difference,
                                            device=d,
                                            mpc_opts=mpc_opts)

    name = pm.dyn_net.name if isinstance(pm, prior.NNPrior) else "{}_{}".format(transform_names[use_tsf],
                                                                                pm.__class__.__name__)
    # expensive evaluation
    evaluate_controller(env, ctrl, name, translation=(4, 4), tasks=[885440, 214219, 305012, 102921])

    # name = "{}_{}".format(ctrl.__class__.__name__, name)
    # env.draw_user_text(name, 14, left_offset=-1.5)
    # sim = block_push.InteractivePush(env, ctrl, num_frames=150, plot=False, save=False, stop_when_done=False)
    # seed = rand.seed()
    # sim.run(seed)
    # logger.info("last run cost %f", np.sum(sim.last_run_cost))
    # plt.ioff()
    # plt.show()

    env.close()


def evaluate_controller(env: block_push.PushAgainstWallStickyEnv, ctrl: controller.Controller, name,
                        tasks: typing.Union[list, int] = 10, tries=10,
                        start_seed=0,
                        translation=(0, 0)):
    """Fixed set of benchmark tasks to do control over, with the total reward for each task collected and reported"""
    num_frames = 150
    env.set_camera_position(translation)
    env.draw_user_text('center {}'.format(translation), 1)
    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=False, save=False)

    name = "{}_{}".format(ctrl.__class__.__name__, name)
    env.draw_user_text(name, 14, left_offset=-1.5)
    writer = SummaryWriter(flush_secs=20, comment=name)

    seed = rand.seed(start_seed)

    if type(tasks) is int:
        tasks = [rand.seed() for _ in range(tasks)]

    try_seeds = []
    for _ in tasks:
        try_seeds.append([rand.seed() for _ in range(tries)])

    logger.info("evaluation seed %d tasks %s tries %d", seed, tasks, tries)

    # load previous runs to avoid doing duplicates
    fullname = os.path.join(cfg.DATA_DIR, 'ctrl_eval.pkl')
    if os.path.exists(fullname):
        with open(fullname, 'rb') as f:
            runs = pickle.load(f)
            logger.info("loaded runs from %s", fullname)
    else:
        runs = {}

    total_costs = torch.zeros((len(tasks), tries))
    lowest_costs = torch.zeros_like(total_costs)
    successes = torch.zeros_like(total_costs)
    for t in range(len(tasks)):
        task_name = '{}{}'.format(tasks[t], translation)
        if task_name not in runs:
            runs[task_name] = {}

        saved = runs[task_name].get(name, None)
        if saved and len(saved) is 4:
            tc, ss, lc, ts = saved
        # new controller for this task or legacy saved results
        else:
            ts = []
        # try only non-duplicated task seeds
        new_tries = [i for i in range(len(try_seeds[t])) if try_seeds[t] not in ts]
        if not new_tries:
            continue
        try_seeds[t] = [try_seeds[t][i] for i in new_tries]

        task_seed = tasks[t]
        rand.seed(task_seed)
        # configure init and goal for task
        init_block_pos, init_block_yaw, init_pusher = random_touching_start(env)
        init_block_pos = np.add(init_block_pos, translation)
        goal_pos = np.add(np.random.uniform(-0.6, 0.6, 2), translation)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher, goal=goal_pos)
        env.draw_user_text('task {}'.format(task_seed), 2)
        logger.info("task %d init block %s goal %s", task_seed, init_block_pos, goal_pos)

        task_costs = np.zeros((num_frames, tries))

        for i, try_seed in enumerate(try_seeds[t]):
            rand.seed(try_seed)
            env.draw_user_text('try {}'.format(try_seed), 3)
            env.draw_user_text('success {}/{}'.format(int(torch.sum(successes[t])), tries), 4)
            sim.run(try_seed)
            logger.info("task %d try %d run cost %f", task_seed, try_seed, sum(sim.last_run_cost))
            total_costs[t, i] = sum(sim.last_run_cost)
            lowest_costs[t, i] = min(sim.last_run_cost)
            task_costs[:len(sim.last_run_cost), i] = sim.last_run_cost
            if task_costs[-1, i] == 0:
                successes[t, i] = 1
            ctrl.reset()

        for step, costs in enumerate(task_costs):
            writer.add_histogram('ctrl_eval/task_{}'.format(task_seed), costs, step)

        task_mean_cost = torch.mean(total_costs[t])
        writer.add_scalar('ctrl_eval/task_{}'.format(task_seed), task_mean_cost, 0)
        logger.info("task %d cost: %f std %f", task_seed, task_mean_cost, torch.std(total_costs[t]))
        # clear trajectories of this task
        env.clear_debug_trajectories()

        if saved is None or len(saved) is 3:
            runs[task_name][name] = total_costs[t], successes[t], lowest_costs[t], try_seeds[t]
        else:
            tc, ss, lc, ts = saved
            ts = ts + try_seeds[t]
            tc = torch.cat((tc, total_costs[t][new_tries]), dim=0)
            ss = torch.cat((ss, successes[t][new_tries]), dim=0)
            lc = torch.cat((lc, lowest_costs[t][new_tries]), dim=0)
            runs[task_name][name] = tc, ss, lc, ts

    # summarize stats
    logger.info("accumulated cost")
    logger.info(total_costs)
    logger.info("successes")
    logger.info(successes)
    logger.info("lowest costs per task and try")
    logger.info(lowest_costs)

    for t in range(len(tasks)):
        logger.info("task %d success %d/%d t cost %.2f (%.2f) l cost %.2f (%.2f)", tasks[t], torch.sum(successes[t]),
                    tries, torch.mean(total_costs[t]), torch.std(total_costs[t]), torch.mean(lowest_costs),
                    torch.std(lowest_costs))
    logger.info("total cost: %f (%f)", torch.mean(total_costs), torch.std(total_costs))
    logger.info("lowest cost: %f (%f)", torch.mean(lowest_costs), torch.std(lowest_costs))
    logger.info("total success: %d/%d", torch.sum(successes), torch.numel(successes))

    # save to file
    with open(fullname, 'wb') as f:
        pickle.dump(runs, f)
        logger.info("saved runs to %s", fullname)
    return total_costs


def learn_invariant(seed=1, name="", MAX_EPOCH=10, BATCH_SIZE=10, **kwargs):
    d, env, config, ds = get_free_space_env_init(seed)

    common_opts = {'too_far_for_neighbour': 0.3, 'name': "{}_s{}".format(name, seed)}
    # add in invariant transform here
    # invariant_tsf = ParameterizedCoordTransform(ds, d, **common_opts)
    # invariant_tsf = Parameterized2Transform(ds, d, **common_opts)
    # invariant_tsf = Parameterized3Transform(ds, d, **common_opts)
    # parameterization 4
    # invariant_tsf = Parameterized3Transform(ds, d, **common_opts, use_sincos_angle=True)
    # invariant_tsf = Parameterized3BatchTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = DistRegularizedTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationRemoveLinearDecoderTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationRemoveLinearDynamicsTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationRemoveAllLinearityTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationAllRegularizedTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationRelaxEncoderTransform(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationDirectDynamics(ds, d, **common_opts, **kwargs)
    # invariant_tsf = AblationNoPassthrough(ds, d, **common_opts, **kwargs)
    invariant_tsf = LinearRelaxEncoderTransform(ds, d, **common_opts, **kwargs)
    invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)


def learn_model(seed=1, name="", transform_name="", train_epochs=600, batch_N=500):
    d, env, config, ds = get_free_space_env_init(seed)

    invariant_tsf = Parameterized3BatchTransform(ds, d, too_far_for_neighbour=0.3, name=transform_name)
    # we expect this transform to have been learned already
    if not invariant_tsf.load(invariant_tsf.get_last_checkpoint()):
        raise RuntimeError("transform {} need to be learned first - use learn_invariant")

    preprocessor = get_preprocessor_for_invariant_tsf(invariant_tsf)
    ds.update_preprocessor(preprocessor)

    mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                       name="dynamics_{}{}".format(transform_name, name))
    mw.learn_model(train_epochs, batch_N=batch_N)


if __name__ == "__main__":
    # collect_touching_freespace_data(trials=200, trial_length=50, level=0)
    # test_dynamics(0, use_tsf=UseTransform.NO_TRANSFORM, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.NO_TRANSFORM, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_1, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_1, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_2, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_2, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_3, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_3, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_4, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_4, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_3_BATCH, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_3_BATCH, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER, online_adapt=False)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER, online_adapt=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_ABLATE_NO_V, online_adapt=False, relearn_dynamics=True)
    # test_dynamics(0, use_tsf=UseTransform.PARAMETERIZED_ABLATE_NO_V, online_adapt=True)
    # verify_coordinate_transform()
    # test_online_model()
    for seed in range(10):
        learn_invariant(seed=seed, name="", MAX_EPOCH=1000, BATCH_SIZE=500)
    # for seed in range(5):
    #     learn_model(seed=seed, transform_name="knn_regularization_s{}".format(seed), name="cov_reg")
