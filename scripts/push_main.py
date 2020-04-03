import logging
import math
import typing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
import torch.nn
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from tensorboardX import SummaryWriter

from meta_contact import cfg, invariant
from meta_contact.dynamics import online_model, model, prior
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.controller import mode_selector
from meta_contact.env import block_push
from meta_contact.invariant import TransformToUse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

REACTION_IN_STATE = True

# have to be set after selecting an environment
env_dir = None


# --- SHARED GETTERS
def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_block_pos = [-0.8, 0.12]
    init_block_yaw = 0
    init_pusher = 0
    goal_pos = [0.85, -0.35]
    # goal_pos = [-0.5, 0.12]
    env_opts = {
        'mode': mode,
        'goal': goal_pos,
        'init_pusher': init_pusher,
        'log_video': log_video,
        'init_block': init_block_pos,
        'init_yaw': init_block_yaw,
        'environment_level': level,
    }
    # env = interactive_block_pushing.PushAgainstWallEnv(**env_opts)
    # env_dir = 'pushing/no_restriction'
    # env = block_push.PushAgainstWallStickyEnv(**env_opts)
    # env_dir = 'pushing/sticky'
    # if REACTION_IN_STATE:
    #     env = block_push.PushWithForceDirectlyReactionInStateEnv(**env_opts)
    # else:
    #     env = block_push.PushWithForceDirectlyEnv(**env_opts)
    # env_dir = 'pushing/direct_force_mini_step'
    env = block_push.PushPhysicallyAnyAlongEnv(**env_opts)
    env_dir = 'pushing/physical'
    # env = block_push.FixedPushDistPhysicalEnv(**env_opts)
    # env_dir = 'pushing/fixed_mag_physical'
    return env


def get_controller_options(env):
    d = get_device()
    u_min, u_max = env.get_control_bounds()
    Q = torch.tensor(env.state_cost(), dtype=torch.double)
    R = 0.01
    if env.nu is 3:
        sigma = [0.3, 0.4, 0.8]
        noise_mu = [0, 0.1, 0]
        u_init = [0, 0.5, 0]
    elif env.nu is 2:
        sigma = [0.3, 0.2]
        noise_mu = [0, 0]
        u_init = [0, 0]
    else:
        raise RuntimeError("Unrecognized environment")
    # tune this so that we figure out to make u-turns
    sigma = torch.tensor(sigma, dtype=torch.double, device=d)
    common_wrapper_opts = {
        'Q': Q,
        'R': R,
        'u_min': u_min,
        'u_max': u_max,
        'compare_to_goal': env.state_difference,
        'device': d,
        'terminal_cost_multiplier': 50,
        'adjust_model_pred_with_prev_error': False,
        'use_orientation_terminal_cost': False,
    }
    mpc_opts = {
        'num_samples': 1000,
        'noise_sigma': torch.diag(sigma),
        'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
        'lambda_': 1e-2,
        'horizon': 40,
        'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
        'sample_null_action': False,
        'step_dependent_dynamics': True,
    }
    return common_wrapper_opts, mpc_opts


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_ds(env, data_dir, **kwargs):
    d = get_device()
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = block_push.PushDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
    return ds, config


def get_free_space_env_init(seed=1, **kwargs):
    d = get_device()
    env = get_env(kwargs.pop('mode', p.DIRECT), **kwargs)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))
    return d, env, config, ds


def get_preprocessor_for_invariant_tsf(invariant_tsf, evaluate_transform=True):
    if evaluate_transform:
        losses = invariant_tsf.evaluate_validation(None)
        logger.info("tsf on validation %s",
                    "  ".join(
                        ["{} {:.5f}".format(name, loss.mean().cpu().item()) if loss is not None else "" for name, loss
                         in zip(invariant_tsf.loss_names(), losses)]))

    # wrap the transform as a data preprocessor
    preprocessor = preprocess.Compose(
        [invariant.InvariantTransformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    return preprocessor


def get_full_controller_name(pm, ctrl, tsf_name):
    name = pm.dyn_net.name if isinstance(pm, prior.NNPrior) else "{}_{}".format(tsf_name, pm.__class__.__name__)
    class_names = "{}".format(ctrl.__class__.__name__)
    if isinstance(ctrl, controller.MPC):
        class_names += "_{}".format(ctrl.dynamics.__class__.__name__)
        if isinstance(ctrl.dynamics, online_model.OnlineGPMixing) and not ctrl.dynamics.use_independent_outputs:
            class_names += "_full"
    name = "{}_{}".format(class_names, name)
    return name


def get_loaded_prior(prior_class, ds, tsf_name, relearn_dynamics):
    d = get_device()
    if prior_class is prior.NNPrior:
        mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(ds.config).to(device=d)), ds,
                           name="dynamics_{}".format(tsf_name))

        pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                                     train_epochs=600)
    elif prior_class is prior.NoPrior:
        pm = prior.NoPrior()
    else:
        pm = prior_class.from_data(ds)
    return pm


class OnlineAdapt:
    NONE = 0
    LINEARIZE_LIKELIHOOD = 1
    GP_KERNEL = 2


def get_controller(env, pm, ds, untransformed_config, online_adapt=OnlineAdapt.GP_KERNEL):
    common_wrapper_opts, mpc_opts = get_controller_options(env)
    d = common_wrapper_opts['device']
    if online_adapt is not OnlineAdapt.NONE:
        if online_adapt is OnlineAdapt.LINEARIZE_LIKELIHOOD:
            dynamics = online_model.OnlineLinearizeMixing(0.1, pm, ds, env.state_difference,
                                                          local_mix_weight_scale=50.0, xu_characteristic_length=10,
                                                          const_local_mix_weight=False, sigreg=1e-10, device=d)
        elif online_adapt is OnlineAdapt.GP_KERNEL:
            dynamics = online_model.OnlineGPMixing(pm, ds, env.state_difference, device=d, max_data_points=50,
                                                   use_independent_outputs=False, sample=False, training_iter=100,
                                                   refit_strategy=online_model.RefitGPStrategy.RESET_DATA)
        else:
            raise RuntimeError("Unrecognized online adaption value {}".format(online_adapt))
        ctrl = online_controller.OnlineMPPI(dynamics, untransformed_config, **common_wrapper_opts,
                                            mpc_opts=mpc_opts)
    else:
        ctrl = controller.MPPI(pm.dyn_net, untransformed_config, **common_wrapper_opts, mpc_opts=mpc_opts)

    return ctrl


def translation_generator():
    for d in [4, 10, 50]:
        for trans in [[1, 1], [-1, 1], [-1, -1]]:
            dd = (trans[0] * d, trans[1] * d)
            yield dd


# --- pushing specific data structures
class PusherNetwork(model.NetworkModelWrapper):
    """Network wrapper with some special validation evaluation"""

    def evaluate_validation(self):
        with torch.no_grad():
            XUv, _, _ = self.ds.original_validation_set()
            # try validation loss outside of our training region (by translating input)
            for dd in translation_generator():
                XU = torch.cat(
                    (XUv[:, :2] + torch.tensor(dd, device=XUv.device, dtype=XUv.dtype),
                     XUv[:, 2:]),
                    dim=1)
                if self.ds.preprocessor is not None:
                    XU = self.ds.preprocessor.transform_x(XU)
                vloss = self.user.compute_validation_loss(XU, self.Yv, self.ds)
                self.writer.add_scalar("loss/validation_{}_{}".format(dd[0], dd[1]), vloss.mean(),
                                       self.step)


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


def constrain_state(state):
    # yaw gets normalized
    state[:, 2] = math_utils.angle_normalize(state[:, 2])
    # along gets constrained
    state[:, 3] = math_utils.clip(state[:, 3], torch.tensor(-1, dtype=torch.double, device=state.device),
                                  torch.tensor(1, dtype=torch.double, device=state.device))
    return state


# --- offline data collection through predetermined or random controllers
def rn(scale):
    return np.random.randn() * scale


class OfflineDataCollection:
    @staticmethod
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
        elif env_type in (block_push.PushAgainstWallStickyEnv, block_push.PushWithForceDirectlyEnv,
                          block_push.PushWithForceDirectlyReactionInStateEnv):
            init_pusher = np.random.uniform(-1, 1)
        elif env_type in (block_push.PushPhysicallyAnyAlongEnv, block_push.FixedPushDistPhysicalEnv):
            init_pusher = 0
        else:
            raise RuntimeError("Unrecognized env type")
        return init_block_pos, init_block_yaw, init_pusher

    @staticmethod
    def freespace(trials=20, trial_length=40, level=0):
        env = get_env(p.DIRECT, level)
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
            init_block_pos, init_block_yaw, init_pusher = OfflineDataCollection.random_touching_start(env)
            env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()

    @staticmethod
    def push_against_wall_recovery():
        # get data in and around the bug trap we want to avoid in the future
        env = get_env(p.GUI, 1, log_video=True)
        u = []
        if isinstance(env, block_push.PushWithForceDirectlyEnv):
            seed = rand.seed(124512)
            for _ in range(10):
                u.append([0.7 + rn(0.2), 0.7 + rn(0.3), 0.6 + rn(0.4)])
            for _ in range(8):
                u.append([0.1 + rn(0.2), 0.7 + rn(0.3), 0.7 + rn(0.3)])
            for _ in range(10):
                u.append([-0.3 + rn(0.4), 0.6 + rn(0.4), 0.0 + rn(0.8)])
            for _ in range(10):
                u.append([-0.8 + rn(0.2), 0.2 + rn(0.1), -0.2 + rn(0.4)])
            for _ in range(90):
                u.append([-0.3 + rn(0.2), 0.8 + rn(0.1), -0.9 + rn(0.2)])
            for _ in range(40):
                u.append([0.1 + rn(0.2), 0.7 + rn(0.1), 0.1 + rn(0.3)])
        elif isinstance(env, block_push.PushPhysicallyAnyAlongEnv):
            seed = rand.seed(3)
            # different friction between wall and block leads to very different behaviour
            high_friction = True
            if high_friction:
                for _ in range(25):
                    u.append([0.8 + rn(0.2), 0.7 + rn(0.3), 0.7 + rn(0.4)])
                for _ in range(15):
                    u.append([-0.8 + rn(0.2), 0.8 + rn(0.1), -0.9 + rn(0.2)])
                for _ in range(10):
                    u.append([0.1 + rn(0.2), 0.7 + rn(0.1), 0.1 + rn(0.3)])
            else:
                for _ in range(20):
                    u.append([0.8 + rn(0.2), 0.7 + rn(0.3), 0.7 + rn(0.4)])
                for _ in range(25):
                    u.append([-1.0 + rn(0.1), 0.8 + rn(0.1), -0.9 + rn(0.1)])
                for _ in range(10):
                    u.append([0.1 + rn(0.2), 0.7 + rn(0.1), 0.1 + rn(0.3)])
        else:
            raise RuntimeError("Unrecognized environment")

        ctrl = controller.PreDeterminedController(np.array(u), *env.get_control_bounds())
        sim = block_push.InteractivePush(env, ctrl, num_frames=len(u), plot=False, save=True, stop_when_done=False)
        sim.run(seed, 'predetermined_bug_trap')

    @staticmethod
    def model_selector_evaluation(seed=5, level=1, relearn_dynamics=False,
                                  prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior):
        # load a reasonable model
        env = get_env(p.GUI, level=level, log_video=True)
        ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

        logger.info("initial random seed %d", rand.seed(seed))

        untransformed_config, tsf_name, _ = update_ds_with_transform(env, ds, UseTransform.COORDINATE_TRANSFORM,
                                                                     evaluate_transform=False)
        pm = get_loaded_prior(prior_class, ds, tsf_name, relearn_dynamics)

        u = []
        for _ in range(10):
            u.append([0.8 + rn(0.2), 0.7 + rn(0.3), 0.7 + rn(0.4)])
        for _ in range(10):
            u.append([-0.8 + rn(0.2), 0.8 + rn(0.1), -0.9 + rn(0.2)])
        for _ in range(5):
            u.append([0.5 + rn(0.5), 0.7 + rn(0.4), -0.2 + rn(0.5)])
        for _ in range(100):
            u.append([0.1 + rn(0.8), 0.7 + rn(0.3), -0.2 + rn(0.7)])

        ctrl = controller.PreDeterminedControllerWithPrediction(np.array(u), pm.dyn_net, *env.get_control_bounds())
        sim = block_push.InteractivePush(env, ctrl, num_frames=len(u), plot=False, save=True, stop_when_done=False)
        sim.run(seed, 'model_selector_evaluation')


# ------- Hand designed coordinate transform classes using architecture 2
class CoordTransform:
    @staticmethod
    def factory(env, *args, **kwargs):
        tsfs = {block_push.PushAgainstWallStickyEnv: CoordTransform.StickyEnv,
                block_push.PushWithForceDirectlyEnv: CoordTransform.Direct,
                block_push.PushWithForceDirectlyReactionInStateEnv: CoordTransform.ReactionInState,
                block_push.PushPhysicallyAnyAlongEnv: CoordTransform.Physical,
                block_push.FixedPushDistPhysicalEnv: CoordTransform.Physical}
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


# ------- Learned transform classes using architecture 3
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


# ------- Ablations on architecture 3
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


def verify_coordinate_transform():
    def get_dx(px, cx):
        dpos = cx[:2] - px[:2]
        dyaw = math_utils.angular_diff(cx[2], px[2])
        dalong = cx[3] - px[3]
        dx = torch.from_numpy(np.r_[dpos, dyaw, dalong])
        return dx

    # comparison tolerance
    tol = 2e-4
    env = get_env(p.GUI)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    tsf = CoordTransform.factory(env, ds)

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
    env = get_env(p.DIRECT, level=0)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))

    invariant_tsf = CoordTransform.factory(env, ds)
    transformer = invariant.InvariantTransformer
    preprocessor = preprocess.Compose(
        [transformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    ds.update_preprocessor(preprocessor)

    prior_name = 'coord_prior'

    mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds, name=prior_name)

    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=600)

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    dynamics = online_model.OnlineLinearizeMixing(0.1, pm, ds, env.state_difference, local_mix_weight_scale=0,
                                                  sigreg=1e-10)

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
        dynamics.const_local_weight = weight
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
    dynamics.const_local_weight = 1.0
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
        'local error after update for horizon {} gamma {} weight {}'.format(horizon, dynamics.gamma,
                                                                            dynamics.const_local_weight))
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
    COORDINATE_LEARN_DYNAMICS_TRANSFORM = 9
    WITH_COMPRESSION_AND_PARTITION = 10


def get_transform(env, ds, use_tsf):
    # add in invariant transform here
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transforms = {
        UseTransform.NO_TRANSFORM: None,
        UseTransform.COORDINATE_TRANSFORM: CoordTransform.factory(env, ds),
        UseTransform.PARAMETERIZED_1: LearnedTransform.ParameterizedYawSelect(ds, d, too_far_for_neighbour=0.3,
                                                                              name="_s2"),
        UseTransform.PARAMETERIZED_2: LearnedTransform.LinearComboLatentInput(ds, d, too_far_for_neighbour=0.3,
                                                                              name="rand_start_s9"),
        UseTransform.PARAMETERIZED_3: LearnedTransform.ParameterizeDecoder(ds, d, too_far_for_neighbour=0.3,
                                                                           name="_s9"),
        UseTransform.PARAMETERIZED_4: LearnedTransform.ParameterizeDecoder(ds, d, too_far_for_neighbour=0.3,
                                                                           name="sincos_s2", use_sincos_angle=True),
        UseTransform.PARAMETERIZED_3_BATCH: LearnedTransform.ParameterizeDecoderBatch(ds, d, name="_s1"),
        UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER: AblationOnTransform.RelaxEncoder(ds, d,
                                                                                                         name="_s0"),
        UseTransform.PARAMETERIZED_ABLATE_NO_V: AblationOnTransform.UseDecoderForDynamics(ds, d, name="_s3"),
        UseTransform.COORDINATE_LEARN_DYNAMICS_TRANSFORM: LearnedTransform.GroundTruthWithCompression(ds, d,
                                                                                                      name="_s0"),
        UseTransform.WITH_COMPRESSION_AND_PARTITION: LearnedTransform.LearnedPartialPassthrough(ds, d, name="_s0"),
    }
    transform_names = {
        UseTransform.NO_TRANSFORM: 'none',
        UseTransform.COORDINATE_TRANSFORM: 'coord',
        UseTransform.PARAMETERIZED_1: 'param1',
        UseTransform.PARAMETERIZED_2: 'param2',
        UseTransform.PARAMETERIZED_3: 'param3',
        UseTransform.PARAMETERIZED_4: 'param4',
        UseTransform.PARAMETERIZED_3_BATCH: 'param3_batch',
        UseTransform.PARAMETERIZED_ABLATE_ALL_LINEAR_AND_RELAX_ENCODER: 'ablate_linear_relax_encoder',
        UseTransform.PARAMETERIZED_ABLATE_NO_V: 'ablate_no_v',
        UseTransform.COORDINATE_LEARN_DYNAMICS_TRANSFORM: 'coord_learn',
        UseTransform.WITH_COMPRESSION_AND_PARTITION: 'compression + partition',
    }
    return transforms[use_tsf], transform_names[use_tsf]


def update_ds_with_transform(env, ds, use_tsf, **kwargs):
    invariant_tsf, tsf_name = get_transform(env, ds, use_tsf)

    if invariant_tsf:
        # load transform (only 1 function for learning transform reduces potential for different learning params)
        if use_tsf is not UseTransform.COORDINATE_TRANSFORM and not invariant_tsf.load(
                invariant_tsf.get_last_checkpoint()):
            raise RuntimeError("Transform {} should be learned before using".format(invariant_tsf.name))

        preprocessor = get_preprocessor_for_invariant_tsf(invariant_tsf, **kwargs)
    else:
        # use minmax scaling if we're not using an invariant transform (baseline)
        preprocessor = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
        # preprocessor = preprocess.Compose([preprocess.PytorchTransformer(preprocess.AngleToCosSinRepresentation(2),
        #                                                                  preprocess.NullSingleTransformer()),
        #                                    preprocess.PytorchTransformer(preprocess.MinMaxScaler())])
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)
    return untransformed_config, tsf_name, preprocessor


def test_dynamics(level=0, use_tsf=UseTransform.COORDINATE_TRANSFORM, relearn_dynamics=False, override=False,
                  prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior, **kwargs):
    seed = 1
    plot_model_error = False
    enforce_model_rollout = False
    full_evaluation = True
    d = get_device()
    if plot_model_error:
        env = get_env(p.DIRECT, level=level)
    else:
        env = get_env(p.GUI, level=level, log_video=True)

    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))

    untransformed_config, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)

    pm = get_loaded_prior(prior_class, ds, tsf_name, relearn_dynamics)

    # test that the model predictions are relatively symmetric for positive and negative along
    if isinstance(env, block_push.PushAgainstWallStickyEnv) and isinstance(pm,
                                                                           prior.NNPrior) and untransformed_config.nx is 4:
        N = 5
        x_top = torch.tensor([0, 0, 0, 1], dtype=torch.double, device=d).repeat(N, 1)
        x_bot = torch.tensor([0, 0, 0, -1], dtype=torch.double, device=d).repeat(N, 1)
        # push straight
        u = torch.tensor([0, 1, 0], dtype=torch.double, device=d)
        # do rollouts
        for i in range(N - 1):
            x_top[i + 1] = pm.dyn_net.predict(torch.cat((x_top[i], u)).view(1, -1))
            x_bot[i + 1] = pm.dyn_net.predict(torch.cat((x_bot[i], u)).view(1, -1))
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
            logger.error(e)
            logger.error(x_top)
            # either fail or just warn that it's an error
            if enforce_model_rollout:
                raise e
            else:
                pass

    # plot model prediction
    if plot_model_error:
        XU, Y, _ = ds.validation_set()
        Y = Y.cpu().numpy()
        Yhatn = pm.dyn_net.user.sample(XU).cpu().detach().numpy()
        E = Yhatn - Y
        # relative error (compared to the mean magnitude)
        Er = E / np.mean(np.abs(Y), axis=0)
        ylabels = ['d' + label for label in env.state_names()]
        for i in range(config.ny):
            plt.subplot(config.ny, 2, 2 * i + 1)
            plt.plot(Y[:, i])
            plt.ylabel(ylabels[i])
            plt.subplot(config.ny, 2, 2 * i + 2)
            # plt.plot(Er[:, i])
            plt.plot(E[:, i])
            plt.ylabel("$e_{}$".format(i))
        plt.show()
        return

    ctrl = get_controller(env, pm, ds, untransformed_config, **kwargs)
    name = get_full_controller_name(pm, ctrl, tsf_name)

    if full_evaluation:
        evaluate_controller(env, ctrl, name, translation=(10, 10), tasks=[102921],
                            override=override)
    else:
        env.draw_user_text(name, 14, left_offset=-1.5)
        # env.sim_step_wait = 0.01
        sim = block_push.InteractivePush(env, ctrl, num_frames=200, plot=True, save=True, stop_when_done=False)
        seed = rand.seed(2)
        env.draw_user_text("try {}".format(seed), 2)
        sim.run(seed, 'evaluation{}'.format(seed))
        logger.info("last run cost %f", np.sum(sim.last_run_cost))
        plt.ioff()
        plt.show()

    env.close()


def test_local_model_sufficiency_for_escaping_wall(use_tsf=UseTransform.COORDINATE_TRANSFORM,
                                                   prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior):
    seed = 1
    plot_model_eval = True
    plot_online_update = False
    use_gp = False
    d = get_device()
    if plot_model_eval:
        env = get_env(p.DIRECT)
    else:
        env = get_env(p.GUI, level=1, log_video=True)

    logger.info("initial random seed %d", rand.seed(seed))

    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)
    untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)
    pm = get_loaded_prior(prior_class, ds, tsf_name, False)

    # use data of transition out of bug trap rather than just inside the bug trap
    # get some data when we're next to a wall and use it fit a local model
    # data from previous policy that got out of bug trap
    # ds_wall, config = get_ds(env, "pushing/push_recovery_global.mat", validation_ratio=0.)
    # train_slice = slice(15, 55)
    # data from predetermined policy for getting into and out of bug trap
    ds_wall, config = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    train_slice = slice(90, 135)

    # use same preprocessor
    ds_wall.update_preprocessor(preprocessor)

    # don't use all of the data; use just the part that's stuck against a wall?
    dynamics = online_model.OnlineLinearizeMixing(0.0, pm, ds_wall, env.state_difference,
                                                  local_mix_weight_scale=50, xu_characteristic_length=10,
                                                  const_local_mix_weight=False, sigreg=1e-10,
                                                  slice_to_use=train_slice, device=d)
    dynamics_gp = online_model.OnlineGPMixing(pm, ds_wall, env.state_difference, slice_to_use=train_slice,
                                              allow_update=plot_online_update, sample=False,
                                              refit_strategy=online_model.RefitGPStrategy.RESET_DATA,
                                              device=d, training_iter=150, use_independent_outputs=False)

    if plot_model_eval:
        ds_test, config = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
        ds_test.update_preprocessor(preprocessor)
        test_slice = slice(0, 170)

        # visualize data and linear fit onto it
        t = np.arange(test_slice.start, test_slice.stop)
        xu, y, info = (v[test_slice] for v in ds_test.training_set())
        reaction_forces = info[:, :2]
        model_errors = info[:, 2:]

        yhat_freespace = pm.dyn_net.user.sample(xu)
        cx, cu = xu[:, :config.nx], xu[:, config.nx:]
        # an actual linear fit on data
        dynamics.const_local_weight = True
        dynamics.local_weight_scale = 1000
        yhat_linear = dynamics._dynamics_in_transformed_space(None, None, cx, cu)

        # our mixed model
        dynamics.const_local_weight = False
        dynamics.local_weight_scale = 50
        dynamics.characteristic_length = 10
        yhat_linear_mix = dynamics._dynamics_in_transformed_space(None, None, cx, cu)
        weight_linear_mix = dynamics.get_local_weight(xu)
        # scale max for display
        weight_linear_mix /= torch.max(weight_linear_mix) * 2

        yhat_gp = dynamics_gp._dynamics_in_transformed_space(None, None, cx, cu)
        yhat_gp_mean = dynamics_gp.mean()
        samples = 20
        gp_samples = dynamics_gp.sample(torch.Size([samples]))
        # TODO gyptorch bug forces us to calculate var for only small batches at a time
        max_sample_each_time = 50
        lower, upper = torch.zeros_like(yhat_gp), torch.zeros_like(yhat_gp)
        for i in range(0, cx.shape[0], max_sample_each_time):
            b_slice = slice(i, i + max_sample_each_time)
            dynamics_gp._make_prediction(cx[b_slice], cu[b_slice])
            lower[b_slice], upper[b_slice], _ = dynamics_gp.get_last_prediction_statistics()

        yhat_linear_online = torch.zeros_like(yhat_linear)
        yhat_gp_online = torch.zeros_like(yhat_gp)
        lower_online = torch.zeros_like(lower)
        upper_online = torch.zeros_like(upper)
        yhat_gp_online_mean = torch.zeros_like(yhat_gp_online)
        px, pu = None, None
        gp_online_fit_loss = torch.zeros(yhat_gp.shape[0])
        gp_online_fit_last_loss_diff = torch.zeros_like(gp_online_fit_loss)
        for i in range(xu.shape[0] - 1):
            cx = xu[i, :config.nx]
            cu = xu[i, config.nx:]
            if px is not None:
                px, pu = px.view(1, -1), pu.view(1, -1)
            yhat_linear_online[i] = dynamics._dynamics_in_transformed_space(px, pu, cx.view(1, -1), cu.view(1, -1))
            yhat_gp_online[i] = dynamics_gp._dynamics_in_transformed_space(px, pu, cx.view(1, -1), cu.view(1, -1))
            with torch.no_grad():
                lower_online[i], upper_online[i], yhat_gp_online_mean[i] = dynamics_gp.get_last_prediction_statistics()
            dynamics._update(cx, cu, y[i])
            dynamics_gp._update(cx, cu, y[i])
            px, pu = cx, cu
            gp_online_fit_loss[i], gp_online_fit_last_loss_diff[i] = dynamics_gp.last_loss, dynamics_gp.last_loss_diff

        XU, Y, Yhat_freespace, Yhat_linear, Yhat_linear_online, reaction_forces = (v.cpu().numpy() for v in (
            xu, y, yhat_freespace, yhat_linear, yhat_linear_online, reaction_forces))

        axis_name = ['d{}'.format(name) for name in env.state_names()]
        to_plot_y_dims = [0, 2, 4, 5]
        num_plots = len(to_plot_y_dims) + 1  # additional reaction force magnitude
        if not use_gp:
            num_plots += 1  # weights for local model (marginalized likelihood of data)
        f, axes = plt.subplots(num_plots, 1, sharex=True)
        for i, dim in enumerate(to_plot_y_dims):
            axes[i].scatter(t, Y[:, dim], label='truth', alpha=0.4)
            axes[i].plot(t, Yhat_freespace[:, dim], label='nominal', alpha=0.4, linewidth=3)

            if use_gp:
                axes[i].plot(t, yhat_gp_mean[:, dim].cpu().numpy(), label='gp')
                axes[i].fill_between(t, lower[:, dim].cpu().numpy(), upper[:, dim].cpu().numpy(), alpha=0.3)
                # axes[i].scatter(np.tile(t, samples), gp_samples[:, :, dim].view(-1).cpu().numpy(), label='gp sample',
                #                 marker='*', color='k', alpha=0.3)
                if plot_online_update:
                    axes[i].plot(t, yhat_gp_online_mean[:, dim].cpu().numpy(), label='online gp')
                    axes[i].fill_between(t, lower_online[:, dim].cpu().numpy(), upper_online[:, dim].cpu().numpy(),
                                         alpha=0.3)
            else:
                # axes[i].plot(t, Yhat_linear[:, dim], label='linear fit', alpha=0.5)
                m = yhat_linear_mix[:, dim].cpu().numpy()
                axes[i].plot(t, m, label='mix', color='g')

            # axes[i].plot(t, Yhat_linear_online[:, dim], label='online mix')
            if dim in [4, 5]:
                # axes[i].set_ybound(-10, 10)
                pass
            else:
                axes[i].set_ybound(-0.2, 1.2)
            axes[i].set_ylabel(axis_name[dim])

        if not use_gp:
            w = weight_linear_mix.cpu().numpy()
            axes[-2].plot(t, w, color='g')
            axes[-2].set_ylabel('local model weight')

        axes[0].legend()
        axes[-1].plot(t, np.linalg.norm(reaction_forces, axis=1))
        axes[-1].set_ylabel('|r|')
        axes[-1].set_ybound(0., 50)
        axes[-1].axvspan(train_slice.start, train_slice.stop, alpha=0.3)
        axes[-1].text((train_slice.start + train_slice.stop) * 0.5, 20, 'local train interval')

        if plot_online_update:
            f, axes = plt.subplots(2, 1, sharex=True)
            axes[0].plot(gp_online_fit_loss)
            axes[0].set_ylabel('online fit loss')
            axes[1].plot(gp_online_fit_last_loss_diff)
            axes[1].set_ylabel('loss last gradient')

        e = (yhat_linear_mix - y).norm(dim=1)
        logger.info('linear mix scale %f length %f mse %.4f', dynamics.local_weight_scale,
                    dynamics.characteristic_length,
                    e.median())
        e = (yhat_linear_online - y).norm(dim=1)
        logger.info('linear online %.4f', e.median())
        e = (yhat_linear - y).norm(dim=1)
        logger.info('linear mse %.4f', e.median())
        e = (yhat_gp - y).norm(dim=1)
        em = (yhat_gp_mean - y).norm(dim=1)
        logger.info('gp mse (sample) %.4f (mean) %.4f', e.median(), em.median())
        e = (yhat_gp_online - y).norm(dim=1)
        em = (yhat_gp_online_mean - y).norm(dim=1)
        logger.info('gp online mse (sample) %.4f (mean) %.4f', e.median(), em.median())
        e = (yhat_freespace - y).norm(dim=1)
        logger.info('nominal mse %.4f', e.median())

        plt.show()
        env.close()
        return

    common_wrapper_opts, mpc_opts = get_controller_options(env)
    common_wrapper_opts['adjust_model_pred_with_prev_error'] = False
    ctrl = online_controller.OnlineMPPI(dynamics_gp if use_gp else dynamics, untransformed_config,
                                        **common_wrapper_opts, constrain_state=constrain_state, mpc_opts=mpc_opts)

    name = get_full_controller_name(pm, ctrl, tsf_name)

    env.draw_user_text(name, 14, left_offset=-1.5)
    sim = block_push.InteractivePush(env, ctrl, num_frames=200, plot=False, save=True, stop_when_done=False)
    seed = rand.seed()
    sim.run(seed, 'test_sufficiency{}'.format(seed))
    logger.info("last run cost %f", np.sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


def evaluate_controller(env: block_push.PushAgainstWallStickyEnv, ctrl: controller.Controller, name,
                        tasks: typing.Union[list, int] = 10, tries=10,
                        start_seed=0,
                        translation=(0, 0),
                        override=False):
    """Fixed set of benchmark tasks to do control over, with the total reward for each task collected and reported"""
    # for evaluations in freespace we always use local model
    if isinstance(ctrl, online_controller.OnlineMPC):
        ctrl.mode_select = mode_selector.AlwaysSelectLocal()
    num_frames = 150
    env.set_camera_position(translation)
    env.draw_user_text('center {}'.format(translation), 2)
    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=False, save=False)

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
        # throw out old data
        if override:
            saved = None

        if saved and len(saved) is 4:
            tc, ss, lc, ts = saved
        # new controller for this task or legacy saved results
        else:
            ts = []
        # try only non-duplicated task seeds
        new_tries = [i for i, s in enumerate(try_seeds[t]) if s not in ts]
        if not new_tries:
            continue
        try_seeds[t] = [try_seeds[t][i] for i in new_tries]

        task_seed = tasks[t]
        rand.seed(task_seed)
        # configure init and goal for task
        init_block_pos, init_block_yaw, init_pusher = OfflineDataCollection.random_touching_start(env)
        init_block_pos = np.add(init_block_pos, translation)
        goal_pos = np.add(np.random.uniform(-0.6, 0.6, 2), translation)
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher, goal=goal_pos)
        env.draw_user_text('task {}'.format(task_seed), 3)
        logger.info("task %d init block %s goal %s", task_seed, init_block_pos, goal_pos)

        task_costs = np.zeros((num_frames, tries))

        for i, try_seed in enumerate(try_seeds[t]):
            rand.seed(try_seed)
            env.draw_user_text('try {}'.format(try_seed), 4)
            env.draw_user_text('success {}/{}'.format(int(torch.sum(successes[t])), tries), 5)
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


def evaluate_model_selector(use_tsf=UseTransform.COORDINATE_TRANSFORM):
    plot_definite_negatives = False
    num_pos_samples = 5000  # start with balanced data

    _, env, _, ds = get_free_space_env_init()
    _, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)
    ds_neg, _ = get_ds(env, "pushing/model_selector_evaluation.mat", validation_ratio=0.)
    ds_neg.update_preprocessor(preprocessor)
    ds_recovery, _ = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    ds_recovery.update_preprocessor(preprocessor)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    dss = [ds, ds_recovery]
    # selector = mode_selector.ReactionForceHeuristicSelector(16, slice(env.nx - 2, None))
    # selector = mode_selector.KDESelector(dss, [1, 0.2])
    # selector = mode_selector.SklearnClassifierSelector(dss, KNeighborsClassifier(n_neighbors=3))
    # selector = mode_selector.SklearnClassifierSelector(dss, GaussianProcessClassifier( n_restarts_optimizer=5, random_state=0))
    # selector = mode_selector.SklearnClassifierSelector(dss, SVC(probability=True, gamma='auto'))
    selector = mode_selector.SklearnClassifierSelector(dss, MLPClassifier())
    # selector = mode_selector.GMMSelector(dss, gmm_opts={'n_components': 3})
    # selector = mode_selector.GMMSelector(dss, gmm_opts={'n_components': 10}, variational=True)

    # get evaluation data by getting definite positive samples from the freespace dataset
    pm = get_loaded_prior(prior.NNPrior, ds, tsf_name, False)
    # get freespace model error as well
    XUf, Yf, infof = ds.training_set(original=True)
    Yhatf = pm.dyn_net.predict(XUf, get_next_state=False)
    mef = Yhatf - Yf
    memf = mef.norm(dim=1)
    freespace_threshold = mef.abs().kthvalue(int(mef.shape[0] * 0.99), dim=0)[0]
    logger.info("freespace error median %f max %f near max each dim %s", memf.median(), memf.max(),
                freespace_threshold)
    # look at evaluation dataset's model errors
    XU, Y, info = ds_neg.training_set(original=True)
    me = info[:, ds_neg.loader.info_desc['model error']]
    contacts = info[:, ds_neg.loader.info_desc['wall contact']]

    # get definitely positive data from the freespace set
    above_thresh = mef.abs() < freespace_threshold
    def_pos = above_thresh.all(dim=1)

    # we label it as bad if any dim's error is above freespace threshold
    above_thresh = me.abs() > freespace_threshold
    def_neg = above_thresh.any(dim=1)

    num_pos = def_pos.sum()
    pos_ind = torch.randperm(num_pos)[:num_pos_samples]

    # combine to form data to test on
    xu = torch.cat((XUf[def_pos][pos_ind], XU[def_neg]), dim=0)
    # label corresponds to the component the mode selector should pick
    target = torch.zeros(xu.shape[0], dtype=torch.long)
    target[num_pos_samples:] = 1

    x, u = torch.split(xu, env.nx, dim=1)
    output = selector.sample_mode(x, u)

    # metrics for binary classification (since it's binary, we can reduce it to prob of not nominal model)
    scores = torch.from_numpy(selector.relative_weights).transpose(0, 1)
    loss = torch.nn.CrossEntropyLoss()
    m = {'cross entropy': loss(scores, target).item()}

    y_score = selector.relative_weights[1]

    from sklearn import metrics
    precision, recall, pr_thresholds = metrics.precision_recall_curve(target, y_score)
    m['average precision'] = metrics.average_precision_score(target, y_score)
    m['PR AUC'] = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(target, y_score)
    m['ROC AUC'] = metrics.auc(fpr, tpr)
    m['f1 (sample)'] = metrics.f1_score(target, output.cpu())

    print('{} {}'.format(selector.name(), num_pos_samples))
    for key, value in m.items():
        print('{}: {:.3f}'.format(key, value))

    # visualize
    N = x.shape[0]
    f, ax1 = plt.subplots()
    t = list(range(N))
    ax1.scatter(t, output.cpu(), label='target', marker='*', color='k', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.stackplot(t, selector.relative_weights,
                  labels=['prob {}'.format(i) for i in range(selector.relative_weights.shape[0])], alpha=0.3)
    plt.legend()
    plt.title('Sample and relative weights {}'.format(selector.name()))
    plt.xlabel('data point ({} separates target)'.format(num_pos_samples))
    ax1.set_ylabel('component')
    ax2.set_ylabel('prob')

    # lw = 2
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % m['ROC AUC'])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC {} (#neg={})'.format(selector.name(), num_pos_samples))
    # plt.legend(loc="lower right")
    # for x, y, txt in zip(fpr[::5], tpr[::5], thresholds[::5]):
    #     plt.annotate(np.round(txt, 3), (x, y - 0.04))
    # rnd_idx = len(thresholds) // 2
    # plt.annotate('this point refers to the tpr and the fpr\n at a probability threshold of {}'.format(
    #     np.round(thresholds[rnd_idx], 3)),
    #     xy=(fpr[rnd_idx], tpr[rnd_idx]), xytext=(fpr[rnd_idx] + 0.2, tpr[rnd_idx] - 0.25),
    #     arrowprops=dict(facecolor='black', lw=2, arrowstyle='->'), )

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (AP = {:.2f}, area = {:.2f})'.format(m['average precision'], m['PR AUC']))
    no_skill = len(target[target == 1]) / len(target)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR {} (#neg={})'.format(selector.name(), num_pos_samples))
    plt.legend(loc="lower left")
    stride = max(len(pr_thresholds) // 15, 1)
    for x, y, txt in zip(recall[::stride], precision[::stride], pr_thresholds[::stride]):
        plt.annotate(np.round(txt, 2), (x + 0.02, y - 0.02))
    # rnd_idx = len(pr_thresholds) // 2
    # plt.annotate('this point refers to the recall and the precision\n at a probability threshold of {}'.format(
    #     np.round(thresholds[rnd_idx], 3)),
    #     xy=(recall[rnd_idx], precision[rnd_idx]), xytext=(0.5, 0.7),
    #     arrowprops=dict(facecolor='black', lw=2, arrowstyle='->'), )

    if plot_definite_negatives:
        from arm_pytorch_utilities import draw
        names = env.state_names()
        f, ax = plt.subplots(len(names) + 1, 1, sharex=True)
        N = me.shape[0]
        for i, name in enumerate(names):
            ax[i].set_ylabel('error {}'.format(name))
            ax[i].plot(me[:, i].abs().cpu(), label='observed')
            ax[i].plot([1, N], [freespace_threshold[i].cpu()] * 2, ':', label='freespace max')
        ax[-1].set_ylabel('num contacts (oracle)')
        ax[-1].plot(contacts.cpu())
        draw.highlight_value_ranges(def_neg.to(dtype=torch.int), ax=ax[-1])
    plt.show()


class Learn:
    @staticmethod
    def invariant(seed=1, name="", MAX_EPOCH=10, BATCH_SIZE=10, **kwargs):
        d, env, config, ds = get_free_space_env_init(seed)

        common_opts = {'too_far_for_neighbour': 0.3, 'name': "{}_s{}".format(name, seed)}
        # add in invariant transform here
        invariant_tsf = LearnedTransform.LearnedPartialPassthrough(ds, d, **common_opts, **kwargs)
        invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)

    @staticmethod
    def model(use_tsf, seed=1, name="", train_epochs=600, batch_N=500):
        d, env, config, ds = get_free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf)
        # tsf_name = "none_at_all"

        mw = PusherNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                           name="dynamics_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


class Visualize:

    @staticmethod
    def _dataset_training_dist(env, ds, z_names=None, v_names=None, fs=(None, None, None), axes=(None, None, None)):
        import seaborn as sns
        plt.ioff()

        XU, Y, _ = ds.training_set()
        X, U = torch.split(XU, env.nx, dim=1)

        def plot_series(series, dim_names, title, f=None, ax=None):
            if f is None:
                f, ax = plt.subplots(1, len(dim_names), figsize=(12, 6))
            f.tight_layout()
            f.suptitle(title)
            for i, name in enumerate(dim_names):
                sns.distplot(series[:, i].cpu().numpy(), ax=ax[i])
                ax[i].set_xlabel(name)
            return f, ax

        ofs = [None] * 3
        oaxes = [None] * 3
        if ds.preprocessor is None:
            ofs[0], oaxes[0] = plot_series(X, env.state_names(), 'states X', fs[0], axes[0])
            ofs[1], oaxes[1] = plot_series(U, ["u{}".format(i) for i in range(env.nu)], 'control U', fs[1], axes[1])
            ofs[2], oaxes[2] = plot_series(Y, ["d{}".format(n) for n in env.state_names()], 'prediction Y', fs[2],
                                           axes[2])
        else:
            if z_names is None:
                z_names = ["$z_{}$".format(i) for i in range(XU.shape[1])]
                v_names = ["$v_{}$".format(i) for i in range(Y.shape[1])]
            ofs[0], oaxes[0] = plot_series(XU, z_names, 'latent input Z (XU)', fs[0], axes[0])
            ofs[1], oaxes[1] = plot_series(Y, v_names, 'prediction V (Y)', fs[1], axes[1])

        return ofs, oaxes

    @staticmethod
    def dist_diff_nominal_and_bug_trap():
        _, env, _, ds = get_free_space_env_init()
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds,
                                                                                UseTransform.COORDINATE_TRANSFORM,
                                                                                evaluate_transform=False)
        coord_z_names = ['p', '\\theta', 'f', '\\beta', '$r_x$', '$r_y$']
        coord_v_names = ['d{}'.format(n) for n in coord_z_names]

        ds_wall, _ = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
        ds_wall.update_preprocessor(preprocessor)
        train_slice = slice(90, 135)
        ds_wall.restrict_training_set_to_slice(train_slice)

        fs, axes = Visualize._dataset_training_dist(env, ds, coord_z_names, coord_v_names)
        Visualize._dataset_training_dist(env, ds_wall, coord_z_names, coord_v_names, fs=fs, axes=axes)
        plt.show()

    @staticmethod
    def _conditioned_dataset(x_limits, u_limits, env, ds, pm, output_dim_index=2, range_epsilon=0.05):
        assert len(x_limits) is env.nx
        assert len(u_limits) is env.nu
        XU, Y, _ = ds.training_set(original=True)
        X, U = torch.split(XU, env.nx, dim=1)
        # TODO by default marginalize over X, which doesn't work for all environments
        output_name = "d{}".format(env.state_names()[output_dim_index])
        # find index of the input variable (we condition on all other input dimensions)
        input_dim_index = u_limits.index(None)
        input_name = env.control_names()[input_dim_index]

        # condition on the given dimensions (since it's continuous, our limits are ranges)
        indices = torch.arange(0, U.shape[0], dtype=torch.long)
        for i, conditioned_value in enumerate(u_limits):
            if conditioned_value is None:
                continue
            allowed = (U[indices, i] < (conditioned_value + range_epsilon)) & (
                    U[indices, i] > (conditioned_value - range_epsilon))
            indices = indices[allowed]

        Yhat = pm.dyn_net.predict(XU[indices], get_next_state=False)

        plt.figure()
        plt.scatter(U[indices, input_dim_index].cpu(), Y[indices, output_dim_index].cpu(), label='true')
        plt.scatter(U[indices, input_dim_index].cpu(), Yhat[:, output_dim_index].cpu(), label='predicted')
        plt.xlabel(input_name)
        plt.ylabel(output_name)
        plt.title('conditioned on u = {} +- {}'.format(u_limits, range_epsilon))
        plt.legend()

    @staticmethod
    def dynamics_stochasticity(use_tsf=UseTransform.COORDINATE_TRANSFORM):
        _, env, _, ds = get_free_space_env_init()
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf,
                                                                                evaluate_transform=False)

        ds_eval, _ = get_ds(env, "pushing/fixed_p_and_beta.mat", validation_ratio=0.)
        ds_eval.update_preprocessor(preprocessor)

        pm = get_loaded_prior(prior.NNPrior, ds, tsf_name, False)
        Visualize._conditioned_dataset([None] * env.nx, [0.9, None, 0.8], env, ds_eval, pm)
        plt.show()

    @staticmethod
    def model_actions_at_givne_state():
        seed = 1
        d = get_device()
        env = get_env(p.GUI, level=1)

        logger.info("initial random seed %d", rand.seed(seed))

        # TODO encapsulate getting loaded online model in a separate function
        ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds,
                                                                                UseTransform.COORDINATE_TRANSFORM,
                                                                                evaluate_transform=False)
        pm = get_loaded_prior(prior.NNPrior, ds, tsf_name, False)

        ds_wall, config = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
        train_slice = slice(90, 135)

        # use same preprocessor
        ds_wall.update_preprocessor(preprocessor)
        dynamics_gp = online_model.OnlineGPMixing(pm, ds_wall, env.state_difference, slice_to_use=train_slice,
                                                  allow_update=False, sample=True,
                                                  refit_strategy=online_model.RefitGPStrategy.RESET_DATA,
                                                  device=d, training_iter=150, use_independent_outputs=False)

        XU, Y, info = ds_wall.training_set(original=True)
        X, U = torch.split(XU, env.nx, dim=1)

        i = 95
        x = X[i]
        u = U[i]
        env.set_state(x.cpu().numpy())
        N = 1000
        # query over range of u and get variance in each dimension
        u_sample = np.random.uniform(*env.get_control_bounds(), (N, env.nu))
        u_sample[-1] = u.cpu().numpy()
        u_dist = np.linalg.norm(u_sample - u.cpu().numpy(), axis=1)
        next_x = dynamics_gp.predict(None, None, x.repeat(N, 1).cpu().numpy(), u_sample)
        var = dynamics_gp.last_prediction.variance.detach().cpu().numpy()

        f, axes = plt.subplots(env.nx, 1, sharex=True)
        for j, name in enumerate(env.state_names()):
            axes[j].scatter(u_dist, var[:, j], alpha=0.3)
            axes[j].set_ylabel('var d{}'.format(name))
        axes[-1].set_xlabel('eucliden dist to trajectory u')
        plt.show()
        input('do visualization')


if __name__ == "__main__":
    level = 0
    # OfflineDataCollection.freespace(trials=200, trial_length=50, level=0)
    # OfflineDataCollection.push_against_wall_recovery()
    # OfflineDataCollection.model_selector_evaluation()
    # Visualize.dist_diff_nominal_and_bug_trap()
    # Visualize.model_actions_at_givne_state()
    # Visualize.dynamics_stochasticity(use_tsf=UseTransform.NO_TRANSFORM)

    evaluate_model_selector()
    # test_local_model_sufficiency_for_escaping_wall(use_tsf=UseTransform.COORDINATE_TRANSFORM)

    # test_dynamics(level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.LINEARIZE_LIKELIHOOD,
    #               override=True)
    # test_dynamics(level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.NONE, override=True)
    # test_dynamics(level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.GP_KERNEL, override=True)

    # test_online_model()
    # for seed in range(1):
    #     Learn.invariant(seed=seed, name="", MAX_EPOCH=1000, BATCH_SIZE=500)
    # for seed in range(1):
    #     Learn.model(UseTransform.NO_TRANSFORM, seed=seed, name="physical first")
