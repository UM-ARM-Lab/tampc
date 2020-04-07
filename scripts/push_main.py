import logging
import math
import typing
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pybullet as p
import torch
import torch.nn
from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from meta_contact.transform.block_push import CoordTransform, LearnedTransform, AblationOnTransform, \
    translation_generator
from tensorboardX import SummaryWriter

from meta_contact import cfg
from meta_contact.transform import invariant
from meta_contact.dynamics import online_model, model, prior
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.controller import mode_selector
from meta_contact.env import block_push

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
    init_block_pos = [-0.8, 0.12 - 0.025]
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
    d = get_device()
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


class OnlineAdapt:
    NONE = 0
    LINEARIZE_LIKELIHOOD = 1
    GP_KERNEL = 2


def get_mixed_model(env, use_tsf=UseTransform.COORDINATE_TRANSFORM,
                    prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior, allow_update=False,
                    d=get_device(), online_adapt=OnlineAdapt.GP_KERNEL):
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)
    untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)
    pm = get_loaded_prior(prior_class, ds, tsf_name, False)

    # data from predetermined policy for getting into and out of bug trap
    ds_wall, config = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    train_slice = slice(0, 26)

    # use same preprocessor
    ds_wall.update_preprocessor(preprocessor)

    dynamics = pm.dyn_net
    if online_adapt is OnlineAdapt.LINEARIZE_LIKELIHOOD:
        dynamics = online_model.OnlineLinearizeMixing(0.0, pm, ds_wall, env.state_difference,
                                                      local_mix_weight_scale=50, xu_characteristic_length=10,
                                                      const_local_mix_weight=False, sigreg=1e-10,
                                                      slice_to_use=train_slice, device=d)
    elif online_adapt is OnlineAdapt.GP_KERNEL:
        dynamics = online_model.OnlineGPMixing(pm, ds_wall, env.state_difference, slice_to_use=train_slice,
                                               allow_update=allow_update, sample=False,
                                               refit_strategy=online_model.RefitGPStrategy.RESET_DATA,
                                               device=d, training_iter=150, use_independent_outputs=False)
    return dynamics, ds, ds_wall, train_slice


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


class UseSelector:
    MLP = 0
    KDE = 1
    GMM = 2
    TREE = 3


def get_selector(dss, use_selector=UseSelector.TREE, *args, **kwargs):
    from sklearn.tree import DecisionTreeClassifier

    component_scale = [1, 0.2]

    if use_selector is UseSelector.MLP:
        selector = mode_selector.MLPSelector(dss, *args, **kwargs)
    elif use_selector is UseSelector.KDE:
        selector = mode_selector.KDESelector(dss, component_scale=component_scale)
    elif use_selector is UseSelector.GMM:
        opts = {'n_components': 10, }
        if kwargs is not None:
            opts.update(kwargs)
        selector = mode_selector.GMMSelector(dss, gmm_opts=opts, variational=True, component_scale=component_scale)
    elif use_selector is UseSelector.TREE:
        selector = mode_selector.SklearnClassifierSelector(dss, DecisionTreeClassifier(**kwargs))
    else:
        raise RuntimeError("Unrecognized selector option")
    return selector


def get_controller(env, pm, ds, untransformed_config, online_adapt=OnlineAdapt.GP_KERNEL):
    common_wrapper_opts, mpc_opts = get_controller_options(env)
    d = common_wrapper_opts['device']
    if online_adapt is not OnlineAdapt.NONE:
        # TODO use get_mixed_model for getting mixed dynamics
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
        elif env_type in (block_push.PushPhysicallyAnyAlongEnv,):
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
                for _ in range(12):
                    u.append([0.8 + rn(0.2), 0.8 + rn(0.2), 0.7 + rn(0.4)])
                for _ in range(10):
                    u.append([-0.9 + rn(0.1), 0.8 + rn(0.1), -0.9 + rn(0.2)])
                for _ in range(6):
                    u.append([0.0 + rn(0.2), 0.7 + rn(0.1), 0.1 + rn(0.1)])
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


def verify_coordinate_transform(seed=6, use_tsf=UseTransform.COORDINATE_TRANSFORM):
    # comparison tolerance
    tol = 2e-4
    name = 'same_action_repeated'
    env = get_env(p.DIRECT, level=0)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    tsf, _ = get_transform(env, ds, use_tsf=use_tsf)
    # tsf = invariant.InvariantTransformer(tsf)
    tsf = preprocess.Compose(
        [invariant.InvariantTransformer(tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    # confirm that inversion is correct
    XU, Y, _ = ds.training_set(True)
    tsf.fit(XU, Y)
    Z, V, _ = tsf.transform(XU, Y)
    X, U = torch.split(XU, env.nx, dim=1)
    Yhat = tsf.invert_transform(V, X)

    torch.allclose(Y, Yhat)

    # assuming we have translational and rotational invariance, doing the same action should result in same delta
    while True:
        try:
            ds_test, config = get_ds(env, 'pushing/{}.mat'.format(name), validation_ratio=0.)
        except IOError:
            # collect data if we don't have any yet
            seed = rand.seed(seed)
            u = [rn(0.8), 0.7 + rn(0.2), rn(0.8)]
            u = [u for _ in range(50)]
            ctrl = controller.PreDeterminedController(np.array(u), *env.get_control_bounds())
            sim = block_push.InteractivePush(env, ctrl, num_frames=len(u), plot=False, save=True, stop_when_done=False)
            sim.run(seed, name)
            continue
        break

    def abs_std(X):
        return (X.std(dim=0) / X.abs().mean(dim=0)).cpu().numpy()

    XU, Y, _ = ds_test.training_set(True)
    logger.info('before tsf x std/mean %s', abs_std(XU[:, :env.nx]))
    logger.info('before tsf y std/mean %s', abs_std(Y))
    # after transform the outputs Y should be the same
    Z, V, _ = tsf.transform(XU, Y)
    logger.info('after tsf z std/mean %s', abs_std(Z))
    logger.info('after tsf v std/mean %s', abs_std(V))


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


def test_dynamics(seed=1, level=0, use_tsf=UseTransform.COORDINATE_TRANSFORM, relearn_dynamics=False, override=False,
                  plot_model_error=False, enforce_model_rollout=False, full_evaluation=True,
                  prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior, **kwargs):
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


def test_local_model_sufficiency_for_escaping_wall(plot_model_eval=True, plot_online_update=False, use_gp=True,
                                                   use_tsf=UseTransform.COORDINATE_TRANSFORM,
                                                   prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior):
    seed = 1

    if plot_model_eval:
        env = get_env(p.DIRECT)
    else:
        env = get_env(p.GUI, level=1, log_video=True)

    logger.info("initial random seed %d", rand.seed(seed))

    dynamics, _, _, _ = get_mixed_model(env, use_tsf=use_tsf, prior_class=prior_class,
                                        online_adapt=OnlineAdapt.LINEARIZE_LIKELIHOOD, allow_update=plot_online_update)
    dynamics_gp, ds, ds_wall, train_slice = get_mixed_model(env, use_tsf=use_tsf, prior_class=prior_class,
                                                            allow_update=plot_online_update)

    pm = dynamics_gp.prior
    if plot_model_eval:
        # TODO use selector and plot the model selector's choices
        ds_test, config = get_ds(env, ("pushing/predetermined_bug_trap.mat", "pushing/physical0.mat"),
                                 validation_ratio=0.)
        ds_test.update_preprocessor(ds.preprocessor)
        test_slice = slice(0, 150)

        # visualize data and linear fit onto it
        xu, y, info = (v[test_slice] for v in ds_test.training_set())
        t = np.arange(test_slice.start, test_slice.stop)
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
        to_plot_y_dims = [0, 2, 3, 4]
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
    ctrl = online_controller.OnlineMPPI(dynamics_gp if use_gp else dynamics, ds.original_config(),
                                        **common_wrapper_opts, constrain_state=constrain_state, mpc_opts=mpc_opts)

    _, tsf_name = get_transform(env, ds, use_tsf)
    name = get_full_controller_name(pm, ctrl, tsf_name)

    env.draw_user_text(name, 14, left_offset=-1.5)
    sim = block_push.InteractivePush(env, ctrl, num_frames=100, plot=False, save=True, stop_when_done=False)
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
    rand.seed(10)

    _, env, _, ds = get_free_space_env_init()
    _, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)
    ds_neg, _ = get_ds(env, "pushing/model_selector_evaluation.mat", validation_ratio=0.)
    ds_neg.update_preprocessor(preprocessor)
    ds_recovery, _ = get_ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    ds_recovery.update_preprocessor(preprocessor)

    selector = get_selector([ds, ds_recovery], use_selector=UseSelector.TREE)

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
    if torch.is_tensor(selector.relative_weights):
        selector.relative_weights = selector.relative_weights.cpu().numpy()
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

    print('{} {}'.format(selector.name, num_pos_samples))
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
    plt.title('Sample and relative weights {}'.format(selector.name))
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
    plt.title('PR {} (#neg={})'.format(selector.name, num_pos_samples))
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


class RolloutMethod:
    NONE = 0
    STATES = 1
    ACTIONS = 2
    ORIG_ACTIONS = 3


def evaluate_ctrl_sampler():
    seed = 1
    rollout_method = RolloutMethod.ACTIONS
    disp_dim = [4]
    eval_i = 59
    train_i = 29

    env = get_env(p.GUI, level=1)
    logger.info("initial random seed %d", rand.seed(seed))

    dynamics, ds, ds_wall, _ = get_mixed_model(env)
    ds_eval, _ = get_ds(env, 'pushing/test_sufficiency140891.mat', validation_ratio=0.)
    ds_eval.update_preprocessor(ds.preprocessor)

    XU, Y, info = ds_wall.training_set(original=True)
    X_train, U_train = torch.split(XU, env.nx, dim=1)

    # evaluate on a non-recovery dataset to see if rolling out the actions from the recovery set is helpful
    XU, Y, info = ds_eval.training_set(original=True)
    X, U = torch.split(XU, env.nx, dim=1)

    x = X[eval_i].cpu().numpy()
    u = U[eval_i].cpu().numpy()
    env.set_state(x, u)

    from sklearn.svm import SVC

    dss = [ds, ds_wall]
    # selector = mode_selector.ReactionForceHeuristicSelector(16, slice(env.nx - 2, None))
    # selector = mode_selector.KDESelector(dss, [1, 0.2])
    selector = mode_selector.SklearnClassifierSelector(dss, SVC(probability=True, gamma='auto'))

    common_wrapper_opts, mpc_opts = get_controller_options(env)
    N = 1000
    mpc_opts['num_samples'] = N
    mpc_opts['lambda_'] = 1
    ctrl = online_controller.OnlineMPPI(dynamics, ds.original_config(), **common_wrapper_opts, mpc_opts=mpc_opts,
                                        mode_select=selector)
    ctrl.set_goal(env.goal)

    # have to rollout previous states up to now beacuse controller is stateful
    if rollout_method is RolloutMethod.ACTIONS:
        ctrl_rollout_steps = min(train_i, ctrl.mpc.T)
        # need to reverse because the stored U is reverse chronological
        ctrl.mpc.U[:ctrl_rollout_steps] = U_train[train_i - ctrl_rollout_steps:train_i].flip(0)
    elif rollout_method is RolloutMethod.STATES:
        for ii in range(train_i):
            ctrl.command(X_train[ii].cpu().numpy())
    elif rollout_method is RolloutMethod.ORIG_ACTIONS:
        ctrl_rollout_steps = min(eval_i, ctrl.mpc.T)
        ctrl.mpc.U[:ctrl_rollout_steps] = U[eval_i - ctrl_rollout_steps:eval_i].flip(0)

    # use controller to sample next action
    u_best = ctrl.command(x)
    u_sample = ctrl.mpc.actions[:, 0, :].cpu().numpy()  # only consider the first step
    next_x = dynamics.predict(None, None, np.tile(x, (N, 1)), u_sample)
    var = dynamics.last_prediction.variance.detach().cpu().numpy()

    # visualize best actions in simulator
    path_cost = ctrl.mpc.cost_total.cpu()
    v, ind = path_cost.sort()
    num_best_actions_to_show = 6
    for top_k in range(num_best_actions_to_show):
        env._draw_action(u_sample[ind[top_k]], debug=top_k + 1)

    names = env.state_names()

    # f, axes = plt.subplots(len(disp_dim), 1)
    # if len(disp_dim) is 1:
    #     axes = [axes]
    # for j, dim in enumerate(disp_dim):
    #     sns.distplot(var[:, dim], ax=axes[j])
    #     axes[j].set_ylabel('d{}'.format(names[dim]))
    # axes[-1].set_xlabel('variance')
    # f.suptitle('distribution variance in sampled actions')
    # f.tight_layout()

    f, axes = plt.subplots(len(disp_dim), 1)
    if len(disp_dim) is 1:
        axes = [axes]
    for j, dim in enumerate(disp_dim):
        axes[j].scatter(var[:, dim], path_cost, alpha=0.3, label='d{}'.format(names[dim]))
        axes[j].set_ylabel('cost')
        axes[j].legend()
    axes[-1].set_xlabel('variance')
    f.suptitle('trajectory cost vs variance in sampled actions')
    f.tight_layout()
    plt.show()
    input('do visualization')


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
    def model_actions_at_given_state():
        seed = 1
        env = get_env(p.GUI, level=1)

        logger.info("initial random seed %d", rand.seed(seed))

        dynamics_gp, ds, ds_wall, _ = get_mixed_model(env)

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
    # Visualize.model_actions_at_given_state()
    # Visualize.dynamics_stochasticity(use_tsf=UseTransform.NO_TRANSFORM)

    # verify_coordinate_transform(UseTransform.COORDINATE_TRANSFORM)
    evaluate_model_selector()
    # evaluate_ctrl_sampler()
    # test_local_model_sufficiency_for_escaping_wall(plot_model_eval=False, use_tsf=UseTransform.COORDINATE_TRANSFORM)

    # test_dynamics(level=level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.LINEARIZE_LIKELIHOOD,
    #               override=True)
    # test_dynamics(level=level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.NONE, override=True,
    #               full_evaluation=False)
    # test_dynamics(level=level, use_tsf=UseTransform.COORDINATE_TRANSFORM, online_adapt=OnlineAdapt.GP_KERNEL, override=True)

    # test_online_model()
    # for seed in range(1):
    #     Learn.invariant(seed=seed, name="", MAX_EPOCH=1000, BATCH_SIZE=500)
    # for seed in range(1):
    #     Learn.model(UseTransform.COORDINATE_TRANSFORM, seed=seed, name="physical r in body")
