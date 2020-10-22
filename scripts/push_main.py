import logging
import math
import typing
import os
import time
import pickle
from datetime import datetime
import argparse
import pprint

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
from arm_pytorch_utilities.optim import get_device
from tampc.transform.block_push import CoordTransform
from tampc.util import update_ds_with_transform, no_tsf_preprocessor, UseTsf, get_transform, TranslationNetworkWrapper, \
    EnvGetter
from tensorboardX import SummaryWriter

from tampc import cfg
from tampc.transform import invariant
from tampc.dynamics import online_model, model, prior, hybrid_model
from tampc.dynamics.hybrid_model import OnlineAdapt, get_gating
from tampc.controller import controller
from tampc.controller import online_controller
from tampc.controller import ilqr
from tampc.controller.gating_function import AlwaysSelectNominal
from tampc.env import block_push
from tampc import util

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

REACTION_IN_STATE = True


# --- SHARED GETTERS
class BlockPushGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "dynamics"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = block_push.PushDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

    @staticmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        if use_tsf is UseTsf.COORD:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer())
        else:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer(),
                                                 preprocess.RobustMinMaxScaler(feature_range=[[0, 0, 0], [3, 3, 1.5]]))

    @staticmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        d = get_device()
        u_min, u_max = env.get_control_bounds()
        Q = torch.tensor(env.state_cost(), dtype=torch.double)
        R = 0.01
        sigma = [0.2, 0.4, 0.7]
        noise_mu = [0, 0.1, 0]
        u_init = [0, 0.5, 0]
        # tune this so that we figure out to make u-turns
        sigma = torch.tensor(sigma, dtype=torch.double, device=d)
        common_wrapper_opts = {
            'Q': Q,
            'R': R,
            'R_env': env.control_cost(),
            'recovery_scale': 2000,
            'u_min': u_min,
            'u_max': u_max,
            'compare_to_goal': env.state_difference,
            'state_dist': env.state_distance,
            'u_similarity': env.control_similarity,
            'device': d,
            'terminal_cost_multiplier': 50,
            'abs_unrecognized_threshold': 10,
            # 'nominal_max_velocity':  0.02,
        }
        mpc_opts = {
            'num_samples': 500,
            'noise_sigma': torch.diag(sigma),
            'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
            'lambda_': 1e-2,
            'horizon': 40,
            'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
            'sample_null_action': False,
            'step_dependent_dynamics': True,
            'rollout_samples': 10,
            'rollout_var_cost': 0,  # penalize variance of trajectory cost across samples
        }
        return common_wrapper_opts, mpc_opts

    @classmethod
    def env(cls, mode=p.GUI, level=0, log_video=False):
        init_block_pos = [-0.8, 0.12 - 0.025]
        init_block_yaw = 0
        init_pusher = 0
        goal_pos = [0.1, 0.5]
        if level is 1:
            init_block_pos = [-0.8, 0.23]
            init_block_yaw = -math.pi / 5
            goal_pos = [0.85, -0.35]
        if level is 2:
            translation = 10
            init_block_pos = [-0.8 + translation, 0.23 + translation]
            init_block_yaw = -math.pi / 5
            goal_pos = [0.85 + translation, -0.35 + translation]
        elif level is 3:
            init_block_pos = [0., 0.6]
            init_block_yaw = -math.pi / 4
            goal_pos = [-0.2, -0.45]
        elif level is 4:
            init_block_pos = [-0.4, 0.23]
            init_block_yaw = -math.pi / 5
            goal_pos = [0.25, -0.65]
        elif level is 5:
            init_block_pos = [-0.4, 0.23]
            init_block_yaw = -math.pi / 5.2
            goal_pos = [0.55, -0.65]
        elif level is 6:
            init_block_pos = [0., 0.6]
            init_block_yaw = -math.pi / 2.2
            goal_pos = [-0.2, -0.45]
        env_opts = {
            'mode': mode,
            'goal': goal_pos,
            'init_pusher': init_pusher,
            'log_video': log_video,
            'init_block': init_block_pos,
            'init_yaw': init_block_yaw,
            'environment_level': level,
        }
        env = block_push.PushPhysicallyAnyAlongEnv(**env_opts)
        cls.env_dir = 'pushing/physical'
        return env


def get_local_model(env, pm, ds_local, d=get_device(), **kwargs):
    return hybrid_model.HybridDynamicsModel.get_local_model(env.state_difference, pm, d, ds_local, **kwargs)


def get_full_controller_name(pm, ctrl, tsf_name):
    name = pm.dyn_net.name if isinstance(pm, prior.NNPrior) else "{}_{}".format(tsf_name, pm.__class__.__name__)
    class_names = "{}".format(ctrl.__class__.__name__)
    if isinstance(ctrl, controller.MPC):
        class_names += "_{}".format(ctrl.dynamics.__class__.__name__)
        if isinstance(ctrl.dynamics, online_model.OnlineGPMixing) and not ctrl.dynamics.use_independent_outputs:
            class_names += "_full"
    name = "{}_{}".format(class_names, name)
    return name


# --- pushing specific data structures
def constrain_state(state):
    # yaw gets normalized
    state[:, 2] = math_utils.angle_normalize(state[:, 2])
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
    def freespace(seed=4, trials=20, trial_length=40, force_gui=False):
        env = BlockPushGetter.env(p.GUI if force_gui else p.DIRECT, 0)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(BlockPushGetter.env_dir, 0)
        sim = block_push.InteractivePush(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                         stop_when_done=False, save_dir=save_dir)
        rand.seed(seed)
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
        env = BlockPushGetter.env(p.GUI, 1, log_video=True)
        init_block_pos = [-0.6, 0.15]
        init_block_yaw = -1.2 * math.pi / 4
        env.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw)

        u = []
        seed = rand.seed(3)
        # different friction between wall and block leads to very different behaviour
        high_friction = False
        if high_friction:
            for _ in range(15):
                u.append([-1.0 + rn(0.1), 0.8 + rn(0.1), -1.0 + rn(0.2)])
            for _ in range(13):
                u.append([-0.9 + rn(0.1), 0.8 + rn(0.1), -0.7 + rn(0.2)])
        else:
            for _ in range(15):
                u.append([0.8 + rn(0.2), 0.7 + rn(0.3), 0.7 + rn(0.4)])
            for _ in range(30):
                u.append([-1.0 + rn(0.1), 0.8 + rn(0.1), -0.9 + rn(0.1)])
            for _ in range(10):
                u.append([0.1 + rn(0.2), 0.7 + rn(0.1), 0.1 + rn(0.3)])

        ctrl = controller.PreDeterminedController(np.array(u), *env.get_control_bounds())
        sim = block_push.InteractivePush(env, ctrl, num_frames=len(u), plot=False, save=True, stop_when_done=False)
        sim.run(seed, 'predetermined_bug_trap')

    @staticmethod
    def model_selector_evaluation(seed=5, level=1, relearn_dynamics=False,
                                  prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior):
        # load a reasonable model
        env = BlockPushGetter.env(p.GUI, level=level, log_video=True)
        ds = BlockPushGetter.ds(env, BlockPushGetter.data_dir(0), validation_ratio=0.1)

        logger.info("initial random seed %d", rand.seed(seed))

        untransformed_config, tsf_name, _ = update_ds_with_transform(env, ds, UseTsf.COORD,
                                                                     BlockPushGetter.pre_invariant_preprocessor,
                                                                     evaluate_transform=False)
        pm = BlockPushGetter.loaded_prior(prior_class, ds, tsf_name, relearn_dynamics)

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


def verify_coordinate_transform(seed=6, use_tsf=UseTsf.COORD):
    # comparison tolerance
    tol = 2e-4
    name = 'same_action_repeated'
    env = BlockPushGetter.env(p.DIRECT, level=0)
    ds = BlockPushGetter.ds(env, BlockPushGetter.data_dir(0), validation_ratio=0.1)

    tsf = get_transform(env, ds, use_tsf=use_tsf)
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
            ds_test = BlockPushGetter.ds(env, 'pushing/{}.mat'.format(name), validation_ratio=0.)
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
    env = BlockPushGetter.env(p.DIRECT, level=0)
    ds = BlockPushGetter.ds(env, BlockPushGetter.data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))

    invariant_tsf = CoordTransform.factory(env, ds)
    transformer = invariant.InvariantTransformer
    preprocessor = preprocess.Compose(
        [transformer(invariant_tsf),
         preprocess.PytorchTransformer(preprocess.MinMaxScaler())])

    ds.update_preprocessor(preprocessor)

    prior_name = 'coord_prior'

    mw = TranslationNetworkWrapper(
        model.DeterministicUser(make.make_sequential_network(ds.original_config()).to(device=d)), ds, name=prior_name)

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


def evaluate_freespace_control(seed=1, level=0, use_tsf=UseTsf.COORD, relearn_dynamics=False,
                               override=False, plot_model_error=False, enforce_model_rollout=False,
                               full_evaluation=True, online_adapt=OnlineAdapt.NONE,
                               prior_class: typing.Type[prior.OnlineDynamicsPrior] = prior.NNPrior, **kwargs):
    d = get_device()
    if plot_model_error:
        env = BlockPushGetter.env(p.DIRECT, level=level)
    else:
        env = BlockPushGetter.env(p.GUI, level=level, log_video=True)

    ds = BlockPushGetter.ds(env, BlockPushGetter.data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))

    untransformed_config, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf,
                                                                 BlockPushGetter.pre_invariant_preprocessor,
                                                                 evaluate_transform=False)

    pm = BlockPushGetter.loaded_prior(prior_class, ds, tsf_name, relearn_dynamics)

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
        XU, _, _ = ds.validation_set()
        Yhatn = pm.dyn_net.user.sample(XU).detach()
        # convert back to untransformed space to allow fair comparison
        XU, Y, _ = ds.validation_set(original=True)
        X = XU[:, :env.nx]
        Y = Y.cpu().numpy()
        Yhatn = ds.preprocessor.invert_transform(Yhatn, X).cpu().numpy()
        E = Yhatn - Y
        # relative error (compared to the mean magnitude)
        Er = E / np.mean(np.abs(Y), axis=0)
        ylabels = ['d' + label for label in env.state_names()]
        config = ds.original_config()
        f, ax = plt.subplots(config.ny, 3, constrained_layout=True)
        for i in range(config.ny):
            ax[i, 0].plot(Y[:, i])
            ax[i, 0].set_ylabel(ylabels[i])
            ax[i, 1].plot(E[:, i])
            ax[i, 1].set_ylabel("$e_{}$".format(i))
            ax[i, 2].plot(Er[:, i])
            ax[i, 2].set_ylabel("$er_{}$".format(i))
        plt.show()
        return

    common_wrapper_opts, mpc_opts = BlockPushGetter.controller_options(env)
    nominal_kwargs = {"online_adapt": online_adapt, }
    nominal_kwargs.update(kwargs)
    hybrid_dynamics = hybrid_model.HybridDynamicsModel([ds], pm, env.state_difference, [use_tsf.name],
                                                       device=d,
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs=nominal_kwargs)
    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, untransformed_config, **common_wrapper_opts,
                                        mpc_opts=mpc_opts)

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


def test_local_model_sufficiency_for_escaping_wall(seed=1, level=1, plot_model_eval=True, plot_online_update=False,
                                                   use_gp=True, allow_update=False,
                                                   recover_adjust=True,
                                                   gating=None,
                                                   use_tsf=UseTsf.COORD, test_traj=None, **kwargs):
    if plot_model_eval:
        env = BlockPushGetter.env(p.DIRECT)
    else:
        env = BlockPushGetter.env(p.GUI, level=level, log_video=True)

    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = BlockPushGetter.prior(env, use_tsf)

    # data from predetermined policy for getting into and out of bug trap
    ds_wall = BlockPushGetter.ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    ds_wall.update_preprocessor(ds.preprocessor)

    dss = [ds, ds_wall]

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, [use_tsf.name],
                                                       preprocessor=no_tsf_preprocessor(),
                                                       local_model_kwargs={
                                                           "allow_update": allow_update or plot_online_update,
                                                           "online_adapt": OnlineAdapt.GP_KERNEL if use_gp else OnlineAdapt.LINEARIZE_LIKELIHOOD
                                                       }.update(kwargs))

    gating = hybrid_dynamics.get_gating() if gating is None else gating

    common_args = [env.state_difference, pm, get_device(), ds_wall, ds.preprocessor]
    dynamics_gp = hybrid_model.HybridDynamicsModel.get_local_model(*common_args,
                                                                   allow_update=allow_update or plot_online_update,
                                                                   online_adapt=OnlineAdapt.GP_KERNEL)
    dynamics_lin = hybrid_model.HybridDynamicsModel.get_local_model(*common_args,
                                                                    allow_update=allow_update or plot_online_update,
                                                                    online_adapt=OnlineAdapt.LINEARIZE_LIKELIHOOD)

    if plot_model_eval:
        if not test_traj:
            test_traj = ("pushing/predetermined_bug_trap.mat", "pushing/physical0.mat")
        ds_test = BlockPushGetter.ds(env, test_traj, validation_ratio=0.)
        config = ds_test.original_config()
        ds_test.update_preprocessor(ds.preprocessor)
        test_slice = slice(0, 150)

        # visualize data and linear fit onto it
        xu, y, info = (v[test_slice] for v in ds_test.training_set())
        t = np.arange(test_slice.start, test_slice.stop)
        reaction_forces = info[:, :2]
        model_errors = info[:, 2:]

        xuo, _, _ = (v[test_slice] for v in ds_test.training_set(original=True))
        x, u = torch.split(xuo, env.nx, dim=1)
        classes = gating.sample_class(x, u)

        yhat_freespace = pm.dyn_net.user.sample(xu)
        cx, cu = xu[:, :config.nx], xu[:, config.nx:]
        # an actual linear fit on data
        dynamics_lin.const_local_weight = True
        dynamics_lin.local_weight_scale = 1000
        yhat_linear = dynamics_lin._dynamics_in_transformed_space(None, None, cx, cu)

        # our mixed model
        dynamics_lin.const_local_weight = False
        dynamics_lin.local_weight_scale = 50
        dynamics_lin.characteristic_length = 10
        yhat_linear_mix = dynamics_lin._dynamics_in_transformed_space(None, None, cx, cu)
        weight_linear_mix = dynamics_lin.get_local_weight(xu)
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
            yhat_linear_online[i] = dynamics_lin._dynamics_in_transformed_space(px, pu, cx.view(1, -1), cu.view(1, -1))
            yhat_gp_online[i] = dynamics_gp._dynamics_in_transformed_space(px, pu, cx.view(1, -1), cu.view(1, -1))
            with torch.no_grad():
                lower_online[i], upper_online[i], yhat_gp_online_mean[i] = dynamics_gp.get_last_prediction_statistics()
            dynamics_lin._update(cx, cu, y[i])
            dynamics_gp._update(cx, cu, y[i])
            px, pu = cx, cu
            gp_online_fit_loss[i], gp_online_fit_last_loss_diff[i] = dynamics_gp.last_loss, dynamics_gp.last_loss_diff

        XU, Y, Yhat_freespace, Yhat_linear, Yhat_linear_online, reaction_forces = (v.cpu().numpy() for v in (
            xu, y, yhat_freespace, yhat_linear, yhat_linear_online, reaction_forces))

        to_plot_y_dims = [0, 2, 3, 4]
        axis_name = ['d$v_{}$'.format(dim) for dim in range(env.nx)]
        num_plots = len(to_plot_y_dims) + 2  # additional reaction force magnitude and dynamics_class selector
        if not use_gp:
            num_plots += 1  # weights for local model (marginalized likelihood of data)
        f, axes = plt.subplots(num_plots, 1, sharex='all')
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
                m = yhat_linear_mix[:, dim].cpu().numpy()
                axes[i].plot(t, m, label='mix', color='g')

            # axes[i].plot(t, Yhat_linear_online[:, dim], label='online mix')
            if dim in [4, 5]:
                # axes[i].set_ybound(-10, 10)
                pass
            else:
                # axes[i].set_ybound(-0.2, 1.2)
                pass
            axes[i].set_ylabel(axis_name[dim])

        if not use_gp:
            w = weight_linear_mix.cpu().numpy()
            axes[-2].plot(t, w, color='g')
            axes[-2].set_ylabel('local model weight')

        axes[0].legend()
        axes[-1].plot(t, np.linalg.norm(reaction_forces, axis=1))
        axes[-1].set_ylabel('|r|')
        axes[-1].set_ybound(0., 50)
        # axes[-1].axvspan(train_slice.start, train_slice.stop, alpha=0.3)
        # axes[-1].text((train_slice.start + train_slice.stop) * 0.5, 20, 'local train interval')
        axes[-2].plot(t, classes.cpu())
        axes[-2].set_ylabel('dynamics class')

        # plot dynamics in original space
        axis_name = ['d{}'.format(name) for name in env.state_names()]
        num_plots = len(to_plot_y_dims) + 1
        f, axes = plt.subplots(num_plots, 1, sharex=True)
        xu_orig, y_orig, _ = (v[test_slice] for v in ds_test.training_set(original=True))
        x_orig, u_orig = torch.split(xu_orig, env.nx, dim=1)
        y_orig_tsf, yhat_freespace_orig, yhat_gp_mean_orig, yhat_linear_mix_orig = (
            ds.preprocessor.invert_transform(v, x_orig) for v in
            (y, yhat_freespace, yhat_gp_mean, yhat_linear_mix))
        gp_samples_orig = ds.preprocessor.invert_transform(gp_samples, x_orig.repeat(samples, 1, 1))
        for i, dim in enumerate(to_plot_y_dims):
            axes[i].scatter(t, y_orig[:, dim].cpu(), label='truth', alpha=0.4)
            axes[i].scatter(t, y_orig_tsf[:, dim].cpu(), label='truth transformed', alpha=0.4, color='k', marker='*')
            axes[i].plot(t, yhat_freespace_orig[:, dim].cpu(), label='nominal', alpha=0.4, linewidth=3)
            if use_gp:
                axes[i].plot(t, yhat_gp_mean_orig[:, dim].cpu().numpy(), label='gp')
                axes[i].scatter(np.tile(t, samples), gp_samples_orig[:, :, dim].view(-1).cpu().numpy(),
                                label='gp sample', marker='.', color='r', alpha=0.10)
            else:
                axes[i].plot(t, yhat_linear_mix_orig[:, dim].cpu(), label='mix', color='g')
            axes[i].set_ylabel(axis_name[dim])
        axes[0].legend()
        axes[-1].plot(t, classes.cpu())
        axes[-1].set_ylabel('dynamics class')

        if plot_online_update:
            f, axes = plt.subplots(2, 1, sharex=True)
            axes[0].plot(gp_online_fit_loss)
            axes[0].set_ylabel('online fit loss')
            axes[1].plot(gp_online_fit_last_loss_diff)
            axes[1].set_ylabel('loss last gradient')

        e = (yhat_linear_mix - y).norm(dim=1)
        logger.info('linear mix scale %f length %f mse %.4f', dynamics_lin.local_weight_scale,
                    dynamics_lin.characteristic_length,
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

    common_wrapper_opts, mpc_opts = BlockPushGetter.controller_options(env)
    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                        **common_wrapper_opts, constrain_state=constrain_state, mpc_opts=mpc_opts)
    ctrl.set_goal(env.goal)

    name = get_full_controller_name(pm, ctrl, use_tsf.name)

    env.draw_user_text(name, 14, left_offset=-1.5)
    env.draw_user_text(gating.name, 13, left_offset=-1.5)
    sim = block_push.InteractivePush(env, ctrl, num_frames=250, plot=False, save=True, stop_when_done=False)
    seed = rand.seed(seed)
    run_name = 'test_sufficiency_{}_{}_{}_{}'.format(level, use_tsf.name, gating.name, seed)
    if allow_update:
        run_name += "_online_adapt"
    sim.run(seed, run_name)
    logger.info("last run cost %f", np.sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


def run_controller(default_run_prefix, pre_run_setup, seed=1, level=1, gating=None,
                   use_tsf=UseTsf.COORD, nominal_adapt=OnlineAdapt.NONE,
                   autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE,
                   adaptive_control_baseline=False,
                   use_demo=False,
                   use_trap_cost=True,
                   reuse_escape_as_demonstration=False, num_frames=250, run_name=None,
                   assume_all_nonnominal_dynamics_are_traps=False,
                   rep_name=None,
                   visualize_rollout=False,
                   override_tampc_params=None,
                   override_mpc_params=None,
                   **kwargs):
    if adaptive_control_baseline:
        use_tsf = UseTsf.NO_TRANSFORM

    env = BlockPushGetter.env(p.GUI, level=level, log_video=True)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = BlockPushGetter.prior(env, use_tsf, rep_name=rep_name)
    dss = [ds]
    demo_trajs = []
    if use_demo:
        demo_trajs = ["pushing/predetermined_bug_trap.mat"]
    for demo in demo_trajs:
        ds_local = BlockPushGetter.ds(env, demo, validation_ratio=0.)
        ds_local.update_preprocessor(ds.preprocessor)
        dss.append(ds_local)

    tampc_opts, mpc_opts = BlockPushGetter.controller_options(env)
    if override_tampc_params is not None:
        tampc_opts.update(override_tampc_params)
    if override_mpc_params is not None:
        mpc_opts.update(override_mpc_params)

    logger.debug("running with parameters\nhigh level controller: %s\nlow level MPC: %s",
                 pprint.pformat(tampc_opts), pprint.pformat(mpc_opts))

    if adaptive_control_baseline:
        online_dynamics = online_model.OnlineLinearizeMixing(0.1, pm, ds,
                                                             env.state_difference,
                                                             local_mix_weight_scale=0,
                                                             const_local_mix_weight=True, sigreg=1e-10,
                                                             device=get_device())
        ctrl = ilqr.OnlineLQR(online_dynamics, ds.original_config(), u_min=tampc_opts['u_min'],
                              u_max=tampc_opts['u_max'], Q=np.diag([1, 1, 0, 0, 0]),
                              R=np.diag([0.1, 0.2, 0.1]), device=get_device(), max_timestep=num_frames,
                              horizon=mpc_opts['horizon'])
        ctrl.set_goal(env.goal)
        env.draw_user_text("adaptive baseline", 13, left_offset=-1.5)
    else:
        hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, [use_tsf.name],
                                                           preprocessor=no_tsf_preprocessor(),
                                                           nominal_model_kwargs={'online_adapt': nominal_adapt},
                                                           local_model_kwargs=kwargs,
                                                           device=tampc_opts['device'])

        # we're always going to be in the nominal mode in this case; might as well speed up testing
        if not use_demo and not reuse_escape_as_demonstration:
            gating = AlwaysSelectNominal()
        else:
            gating = hybrid_dynamics.get_gating() if gating is None else gating

        ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                            autonomous_recovery=autonomous_recovery,
                                            assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                            reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                            use_trap_cost=use_trap_cost,
                                            **tampc_opts, constrain_state=constrain_state,
                                            mpc_opts=mpc_opts)
        ctrl.set_goal(env.goal)
        env.draw_user_text(gating.name, 13, left_offset=-1.5)
        env.draw_user_text("run seed {}".format(seed), 12, left_offset=-1.5)
        env.draw_user_text("recovery {}".format(autonomous_recovery.name), 11, left_offset=-1.6)
        if reuse_escape_as_demonstration:
            env.draw_user_text("reuse escape", 10, left_offset=-1.6)
        if use_trap_cost:
            env.draw_user_text("trap set cost".format(autonomous_recovery.name), 9, left_offset=-1.6)

    sim = block_push.InteractivePush(env, ctrl, num_frames=num_frames, plot=False, save=True, stop_when_done=True,
                                     visualize_rollouts=visualize_rollout)
    seed = rand.seed(seed)

    if run_name is None:
        def affix_run_name(*args):
            nonlocal run_name
            for token in args:
                run_name += "__{}".format(token)

        def get_rep_model_name(ds):
            import re
            tsf_name = ""
            try:
                for tsf in ds.preprocessor.tsf.transforms:
                    if isinstance(tsf, invariant.InvariantTransformer):
                        tsf_name = tsf.tsf.name
                        tsf_name = re.match(r".*?s\d+", tsf_name)[0]
            except AttributeError:
                pass
            return tsf_name

        run_name = default_run_prefix
        if adaptive_control_baseline:
            affix_run_name("ADAPTIVE_BASELINE")
        else:
            affix_run_name(nominal_adapt.name)
            affix_run_name(autonomous_recovery.name + ("_WITHDEMO" if use_demo else ""))
        affix_run_name(level)
        affix_run_name(use_tsf.name)
        if not adaptive_control_baseline:
            affix_run_name("ALLTRAP" if assume_all_nonnominal_dynamics_are_traps else "SOMETRAP")
            affix_run_name("REUSE" if reuse_escape_as_demonstration else "NOREUSE")
            affix_run_name(gating.name)
            affix_run_name("TRAPCOST" if use_trap_cost else "NOTRAPCOST")
        affix_run_name(get_rep_model_name(ds))
        affix_run_name(seed)
        affix_run_name(num_frames)

    env.draw_user_text(run_name, 14, left_offset=-1.5)

    pre_run_setup(env, ctrl)

    sim.run(seed, run_name)
    logger.info("last run cost %f", np.sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


def test_autonomous_recovery(*args, **kwargs):
    def default_setup(env, ctrl):
        return

    run_controller('auto_recover', default_setup, *args, **kwargs)


def tune_trap_set_cost(*args, num_frames=100, **kwargs):
    def setup(env, ctrl):
        # setup initial conditions where we are close to a trap and have items in our trap set
        ctrl.nominal_avg_velocity = 0.012
        ctrl.trap_set.append((torch.tensor([0.6147, 0.1381, -1.2658, 6.9630, 14.9701], device=ctrl.d, dtype=ctrl.dtype),
                              torch.tensor([-0.5439, 0.6192, -0.5857], device=ctrl.d, dtype=ctrl.dtype)))

        x = [0.4, 0.15, -math.pi / 8, 0, 0]
        env.set_task_config(init_block=x[:2], init_yaw=x[2])
        x = torch.tensor(x, device=ctrl.d, dtype=ctrl.dtype)
        ctrl.normalize_trapset_cost_to_state(x)

    run_controller('tune_trap_cost', setup, *args, num_frames=num_frames, **kwargs)


def tune_recovery_policy(*args, num_frames=100, **kwargs):
    def setup(env, ctrl: online_controller.OnlineMPPI):
        # setup initial conditions where we are close to a trap and have items in our trap set
        ctrl.nominal_max_velocity = 0.012
        ctrl.trap_set.append((torch.tensor([0.6147, 0.1381, -1.2658, 6.9630, 14.9701], device=ctrl.d, dtype=ctrl.dtype),
                              torch.tensor([-0.5439, 0.6192, -0.5857], device=ctrl.d, dtype=ctrl.dtype)))

        x = [0.063, 0.14, -1.246, 0., 0.]
        env.set_task_config(init_block=x[:2], init_yaw=x[2])

        ctrl.nominal_dynamic_states = [
            [torch.tensor([-0.0525, 0.1609, -0.4940, -3.0287, 15.0308], device=ctrl.d, dtype=ctrl.dtype)]]

        ctrl._start_recovery_mode()

    run_controller('tune_recovery', setup, *args, num_frames=num_frames, use_trap_cost=False, **kwargs)


def evaluate_controller(env: block_push.PushAgainstWallStickyEnv, ctrl: controller.Controller, name,
                        tasks: typing.Union[list, int] = 10, tries=10,
                        start_seed=0,
                        translation=(0, 0),
                        override=False):
    """Fixed set of benchmark tasks to do control over, with the total reward for each task collected and reported"""
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


def evaluate_gating_function(use_tsf=UseTsf.COORD, test_file="pushing/model_selector_evaluation.mat"):
    plot_definite_negatives = False
    num_pos_samples = 100  # start with balanced data
    seed = rand.seed(9)

    _, env, _, ds = BlockPushGetter.free_space_env_init(seed)
    _, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, BlockPushGetter.pre_invariant_preprocessor,
                                                         evaluate_transform=False)
    ds_neg = BlockPushGetter.ds(env, test_file, validation_ratio=0.)
    ds_neg.update_preprocessor(preprocessor)
    ds_recovery = BlockPushGetter.ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
    ds_recovery.update_preprocessor(preprocessor)

    rand.seed(seed)
    gating = get_gating([ds, ds_recovery], tsf_name)

    # get evaluation data by getting definite positive samples from the freespace dataset
    pm = BlockPushGetter.loaded_prior(prior.NNPrior, ds, tsf_name, False)
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
    # label corresponds to the component the dynamics_class selector should pick
    target = torch.zeros(xu.shape[0], dtype=torch.long)
    target[num_pos_samples:] = 1

    x, u = torch.split(xu, env.nx, dim=1)
    output = gating.sample_class(x, u)

    # metrics for binary classification (since it's binary, we can reduce it to prob of not nominal model)
    if torch.is_tensor(gating.relative_weights):
        gating.relative_weights = gating.relative_weights.cpu().numpy()
    scores = torch.from_numpy(gating.relative_weights).transpose(0, 1)
    loss = torch.nn.CrossEntropyLoss()
    m = {'cross entropy': loss(scores, target).item()}

    y_score = gating.relative_weights[1]

    from sklearn import metrics
    precision, recall, pr_thresholds = metrics.precision_recall_curve(target, y_score)
    m['average precision'] = metrics.average_precision_score(target, y_score)
    m['PR AUC'] = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(target, y_score)
    m['ROC AUC'] = metrics.auc(fpr, tpr)
    m['f1 (sample)'] = metrics.f1_score(target, output.cpu())

    print('{} {}'.format(gating.name, num_pos_samples))
    for key, value in m.items():
        print('{}: {:.3f}'.format(key, value))

    # visualize
    N = x.shape[0]
    f, ax1 = plt.subplots()
    t = list(range(N))
    ax1.scatter(t, output.cpu(), label='target', marker='*', color='k', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.stackplot(t, gating.relative_weights,
                  labels=['prob {}'.format(i) for i in range(gating.relative_weights.shape[0])], alpha=0.3)
    plt.legend()
    plt.title('Sample and relative weights {}'.format(gating.name))
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
    plt.title('PR {} (#neg={})'.format(gating.name, num_pos_samples))
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
        f, ax = plt.subplots(len(names) + 1, 1, sharex='all')
        N = me.shape[0]
        for i, name in enumerate(names):
            ax[i].set_ylabel('error {}'.format(name))
            ax[i].plot(me[:, i].abs().cpu(), label='observed')
            ax[i].plot([1, N], [freespace_threshold[i].cpu()] * 2, ':', label='freespace max')
        ax[-1].set_ylabel('num contacts (oracle)')
        ax[-1].plot(contacts.cpu())
        draw.highlight_value_ranges(def_neg.to(dtype=torch.int), ax=ax[-1])
    plt.show()


def evaluate_ctrl_sampler(eval_file, eval_i, seed=1, use_tsf=UseTsf.COORD,
                          do_rollout_best_action=True, assume_all_nonnominal_dynamics_are_traps=False,
                          rollout_prev_xu=True, manual_control=None, step_N_steps=80, **kwargs):
    env = BlockPushGetter.env(p.GUI, level=1, log_video=do_rollout_best_action)
    env.draw_user_text("eval {} seed {}".format(eval_file, seed), 14, left_offset=-1.5)
    env.draw_user_text("eval index {}".format(eval_i), 13, left_offset=-1.5)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = BlockPushGetter.prior(env, use_tsf)
    dss = [ds]

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, [use_tsf.name],
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': OnlineAdapt.NONE},
                                                       local_model_kwargs=kwargs)
    gating = hybrid_dynamics.get_gating()

    common_wrapper_opts, mpc_opts = BlockPushGetter.controller_options(env)
    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                        autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE,
                                        reuse_escape_as_demonstration=False,
                                        use_trap_cost=True,
                                        assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                        **common_wrapper_opts, constrain_state=constrain_state, mpc_opts=mpc_opts)
    ctrl.set_goal(env.goal)

    ds_eval = BlockPushGetter.ds(env, eval_file, validation_ratio=0.)
    ds_eval.update_preprocessor(ds.preprocessor)

    # evaluate on a non-recovery dataset to see if rolling out the actions from the recovery set is helpful
    XU, Y, info = ds_eval.training_set(original=True)
    X, U = torch.split(XU, env.nx, dim=1)

    # put the state right before the evaluated action
    x = X[eval_i].cpu().numpy()
    logger.info(np.array2string(x, separator=', '))
    if rollout_prev_xu:
        # only need to do rollouts; don't need control samples
        T = ctrl.mpc.T
        ctrl.original_horizon = 1
        for i in range(eval_i):
            env.draw_user_text(str(i), 1)
            env.set_state(X[i].cpu().numpy(), U[i].cpu().numpy())
            ctrl.mpc.change_horizon(1)
            ctrl.command(X[i].cpu().numpy())

        ctrl.original_horizon = T
        ctrl.mpc.change_horizon(T)

        # U_mpc_orig = ctrl.mpc.U.clone()
        # if ctrl.recovery_cost:
        #     env.draw_user_text("recovery" if ctrl.autonomous_recovery_mode else "", 3)
        #     env.draw_user_text("goal set yaws" if ctrl.autonomous_recovery_mode else "", 1, -1.5)
        #     for i, goal in enumerate(ctrl.recovery_cost.goal_set):
        #         env.draw_user_text(
        #             "{:.2f}".format(goal[2].item()) if ctrl.autonomous_recovery_mode else "".format(
        #                 goal[2]), 2 + i, -1.5)
        #
        # samples = [500, 1000, 2000]
        # for K in samples:
        #     ctrl.mpc.U = U_mpc_orig.clone()
        #     ctrl.mpc.K = K
        #     # use controller to sample next action
        #     u_best = ctrl.mpc.command(x)
        #
        #     # visualize best actions in simulator
        #     path_cost = ctrl.mpc.cost_total.cpu()
        #     # show best action and its rollouts
        #     env.draw_user_text("{} samples".format(K), 1)
        #     env._draw_action(u_best)
        #     env.draw_user_text("{} cost".format(path_cost.min()), 2)
        #     for i in range(1, 10):
        #         env._draw_action(ctrl.mpc.U[i], debug=i)
        #     env.visualize_rollouts(ctrl.get_rollouts(x))
        #     time.sleep(2)
        # time.sleep(2)
    else:
        env.set_state(x, U[eval_i].cpu().numpy())
        ctrl.command(x)

    # give manual control and visualize rollouts
    if manual_control is not None:
        u = manual_control
        N = u.shape[0]
        U = u.repeat(N, 1)
        ctrl.mpc.U = U.clone()

        env.draw_user_text("manual actions", 1)
        total_cost, _, _ = ctrl.mpc._compute_rollout_costs(U.view(1, N, -1))
        env.draw_user_text("{} cost".format(total_cost.item()), 2)
        env.visualize_rollouts(ctrl.get_rollouts(x))
        # execute those moves and compare against rollout
        for u in U:
            env.step(u.cpu().numpy())

        time.sleep(5)

    else:
        obs = X[eval_i + 1].cpu().numpy()
        # if not given manual control, just do control
        for i in range(step_N_steps):
            action = ctrl.command(obs)
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = env.step(action)

            env.draw_user_text(str(eval_i + i + 1), 1)
            env.draw_user_text("dyn cls {}".format(ctrl.dynamics_class), 2)

            mode_text = "recovery" if ctrl.autonomous_recovery_mode else (
                "local" if ctrl.using_local_model_for_nonnominal_dynamics else "")
            env.draw_user_text(mode_text, 3)
            if ctrl.recovery_cost:
                # plot goal set
                env.visualize_goal_set(ctrl.recovery_cost.goal_set)
            if ctrl.trap_cost is not None and ctrl.trap_set:
                # only show last few to reduce clutter
                env.visualize_trap_set(ctrl.trap_set[-10:])
            env.visualize_rollouts(ctrl.get_rollouts(obs))

    # f, axes = plt.subplots(2, 1, sharex=True)
    # path_sampled_cost = ctrl.mpc.cost_samples[:, ind]
    # t = np.arange(N)
    # axes[0].scatter(np.tile(t, (M, 1)), path_sampled_cost.cpu(), alpha=0.2)
    # axes[0].set_ylabel('cost')
    #
    # modes = [ctrl.dynamics_class_prediction[i].view(20, -1) for i in range(ctrl.mpc.T)]
    # modes = torch.stack(modes, dim=0)
    # modes = (modes == 0).sum(dim=0)
    # axes[1].scatter(np.tile(t, (M, 1)), modes[:, ind].cpu(), alpha=0.2)
    # axes[1].set_ylabel('# nominal dyn cls in traj')
    # axes[-1].set_xlabel('u sample')
    #
    # plt.show()


class Learn:
    @staticmethod
    def invariant(use_tsf=UseTsf.DX_TO_V, seed=1, name="", MAX_EPOCH=3000, BATCH_SIZE=500, resume=False, **kwargs):
        d, env, config, ds = BlockPushGetter.free_space_env_init(seed)
        ds.update_preprocessor(BlockPushGetter.pre_invariant_preprocessor(use_tsf))
        invariant_cls = get_transform(env, ds, use_tsf).__class__
        ds_test = BlockPushGetter.ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
        # ds_test_2 = BlockPushGetter.ds(env, "pushing/test_sufficiency_3_failed_test_140891.mat", validation_ratio=0.)
        common_opts = {'name': "{}_s{}".format(name, seed), 'ds_test': [ds_test]}
        invariant_tsf = invariant_cls(ds, d, **common_opts, **kwargs)
        if resume:
            invariant_tsf.load(invariant_tsf.get_last_checkpoint())
        invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)

    @staticmethod
    def model(use_tsf, seed=1, name="", train_epochs=500, batch_N=500, rep_name=None):
        d, env, config, ds = BlockPushGetter.free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf, BlockPushGetter.pre_invariant_preprocessor,
                                                  rep_name=rep_name)
        # tsf_name = "none_at_all"

        mw = TranslationNetworkWrapper(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                                       name="dynamics_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


class Visualize:
    @staticmethod
    def _state_sequence(env, X, u_0, step):
        X = X.cpu().numpy()
        N = len(X)
        env.set_state(X[-1], u_0)
        final_rgba = np.array(p.getVisualShapeData(env.blockId)[0][7])
        start_rgba = np.array([0.0, 0.1, 0.0, 0.1])

        for i in range(0, N - 1, step):
            block_id = p.loadURDF(os.path.join(cfg.ROOT_DIR, "block_big.urdf"), env.initBlockPos,
                                  p.getQuaternionFromEuler([0, 0, env.initBlockYaw]))
            env.set_state(X[i], block_id=block_id)
            t = float(i) / (N - 1)
            rgba = start_rgba + t * (final_rgba - start_rgba)
            p.changeVisualShape(block_id, -1, rgbaColor=rgba)
        input('wait for input')

    @staticmethod
    def state_sequence(level, file, restrict_slice=None, step=3):
        env = BlockPushGetter.env(mode=p.GUI, level=level)
        ds = BlockPushGetter.ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)
        if restrict_slice:
            X = X[restrict_slice]
        Visualize._state_sequence(env, X, U[0], step)

    @staticmethod
    def _dataset_training_dist(env, ds, z_names=None, v_names=None, fs=(None, None, None), axes=(None, None, None)):

        plt.ioff()

        XU, Y, _ = ds.training_set()

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
            X, U = torch.split(XU, env.nx, dim=1)
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
    def dist_diff_nominal_and_bug_trap(use_tsf, test_file="pushing/predetermined_bug_trap.mat"):
        _, env, _, ds = BlockPushGetter.free_space_env_init()
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf,
                                                                                BlockPushGetter.pre_invariant_preprocessor,
                                                                                evaluate_transform=False)
        coord_z_names = ['p', '\\theta', 'f', '\\beta', '$r_x$', '$r_y$'] if use_tsf in (
            UseTsf.COORD, UseTsf.COORD_LEARN_DYNAMICS) else None
        coord_v_names = ['d{}'.format(n) for n in coord_z_names] if use_tsf in (
            UseTsf.COORD, UseTsf.COORD_LEARN_DYNAMICS) else None

        ds_test = BlockPushGetter.ds(env, test_file, validation_ratio=0.)
        ds_test.update_preprocessor(preprocessor)

        fs, axes = Visualize._dataset_training_dist(env, ds, coord_z_names, coord_v_names)
        Visualize._dataset_training_dist(env, ds_test, coord_z_names, coord_v_names, fs=fs, axes=axes)
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
    def dynamics_stochasticity(use_tsf=UseTsf.COORD):
        _, env, _, ds = BlockPushGetter.free_space_env_init()
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf,
                                                                                BlockPushGetter.pre_invariant_preprocessor,
                                                                                evaluate_transform=False)

        ds_eval = BlockPushGetter.ds(env, "pushing/fixed_p_and_beta.mat", validation_ratio=0.)
        ds_eval.update_preprocessor(preprocessor)

        pm = BlockPushGetter.loaded_prior(prior.NNPrior, ds, tsf_name, False)
        Visualize._conditioned_dataset([None] * env.nx, [0.9, None, 0.8], env, ds_eval, pm)
        plt.show()

    @staticmethod
    def model_actions_at_given_state():
        seed = 1
        env = BlockPushGetter.env(p.GUI, level=1)

        logger.info("initial random seed %d", rand.seed(seed))

        ds, pm = BlockPushGetter.prior(env)
        ds_wall = BlockPushGetter.ds(env, "pushing/predetermined_bug_trap.mat", validation_ratio=0.)
        ds_wall.update_preprocessor(ds.preprocessor)
        dynamics_gp = get_local_model(env, pm, ds_wall)

        XU, Y, info = ds_wall.training_set(original=True)
        X, U = torch.split(XU, env.nx, dim=1)

        i = 15
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

        f, axes = plt.subplots(env.nx, 1, sharex='all')
        for j, name in enumerate(env.state_names()):
            axes[j].scatter(u_dist, var[:, j], alpha=0.3)
            axes[j].set_ylabel('var d{}'.format(name))
        axes[-1].set_xlabel('eucliden dist to trajectory u')
        plt.show()
        input('do visualization')


class EvaluateTask:
    @staticmethod
    def closest_distance_to_goal(file, level, visualize=True, nodes_per_side=150):
        from sklearn.preprocessing import MinMaxScaler
        env = BlockPushGetter.env(p.GUI if visualize else p.DIRECT, level=level)
        ds = BlockPushGetter.ds(env, file, validation_ratio=0., loader_args={'ignore_masks': True})
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        if level is 1:
            min_pos = [-0.9, -0.8]
            max_pos = [1.3, 0.5]
        elif level in [3, 6]:
            min_pos = [-0.5, -1.1]
            max_pos = [1.3, 0.9]
        elif level in [4, 5]:
            min_pos = [-0.7, -0.9]
            max_pos = [1.3, 0.4]
        else:
            raise RuntimeError("Unspecified range for level {}".format(level))

        scaler = MinMaxScaler(feature_range=(0, nodes_per_side - 1))
        scaler.fit(np.array([min_pos, max_pos]))

        reached_states = X[:, :2].cpu().numpy()
        goal_pos = env.goal[:2]

        lower_bound_dist = np.linalg.norm((reached_states - goal_pos), axis=1).min()

        # we expect there not to be walls between us if the minimum distance is this small
        # if lower_bound_dist < 0.2:
        #     return lower_bound_dist

        def node_to_pos(node):
            return scaler.inverse_transform([node])[0]

        def pos_to_node(pos):
            pair = scaler.transform([pos])[0]
            node = tuple(int(round(v)) for v in pair)
            return node

        z = env.initPusherPos[2]
        # draw search boundaries
        rgb = [0, 0, 0]
        p.addUserDebugLine([min_pos[0], min_pos[1], z], [max_pos[0], min_pos[1], z], rgb)
        p.addUserDebugLine([max_pos[0], min_pos[1], z], [max_pos[0], max_pos[1], z], rgb)
        p.addUserDebugLine([max_pos[0], max_pos[1], z], [min_pos[0], max_pos[1], z], rgb)
        p.addUserDebugLine([min_pos[0], max_pos[1], z], [min_pos[0], min_pos[1], z], rgb)

        # draw previous trajectory
        rgb = [0, 0, 1]
        start = reached_states[0, 0], reached_states[0, 1], z
        for i in range(1, len(reached_states)):
            next = reached_states[i, 0], reached_states[i, 1], z
            p.addUserDebugLine(start, next, rgb)
            start = next

        # try to load it if possible
        fullname = os.path.join(cfg.DATA_DIR, 'ok{}_{}.pkl'.format(level, nodes_per_side))
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                ok_nodes = pickle.load(f)
                logger.info("loaded ok nodes from %s", fullname)
        else:
            ok_nodes = [[None for _ in range(nodes_per_side)] for _ in range(nodes_per_side)]
            orientation = p.getQuaternionFromEuler([0, 0, 0])
            pointer = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), (0, 0, z))
            # discretize positions and show goal states
            xs = np.linspace(min_pos[0], max_pos[0], nodes_per_side)
            ys = np.linspace(min_pos[1], max_pos[1], nodes_per_side)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p.resetBasePositionAndOrientation(pointer, (x, y, z), orientation)
                    c = p.getClosestPoints(pointer, env.walls[0], 0)
                    if not c:
                        n = pos_to_node((x, y))
                        ok_nodes[i][j] = n

        with open(fullname, 'wb') as f:
            pickle.dump(ok_nodes, f)
            logger.info("saved ok nodes to %s", fullname)

        # distance 1 step along x
        dxx = (max_pos[0] - min_pos[0]) / nodes_per_side
        dyy = (max_pos[1] - min_pos[1]) / nodes_per_side
        neighbours = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        distances = [dxx, dyy, dxx, dyy]
        # create graph and do search on it based on environment obstacles
        g = util.Graph()
        for i in range(nodes_per_side):
            for j in range(nodes_per_side):
                u = ok_nodes[i][j]
                if u is None:
                    continue
                g.add_node(u)
                for dxy, dist in zip(neighbours, distances):
                    ii = i + dxy[0]
                    jj = j + dxy[1]
                    if ii < 0 or ii >= nodes_per_side:
                        continue
                    if jj < 0 or jj >= nodes_per_side:
                        continue
                    v = ok_nodes[ii][jj]
                    if v is not None:
                        g.add_edge(u, v, dist)

        goal_node = pos_to_node(goal_pos)
        visited, path = util.dijsktra(g, goal_node)
        # find min across visited states
        min_dist = 100
        min_node = None
        dists = []
        for xy in reached_states:
            n = pos_to_node(xy)
            if n not in visited:
                logger.warning("reached state %s node %s not visited", xy, n)
                dists.append(None)
            else:
                dists.append(visited[n])
                if visited[n] < min_dist:
                    min_dist = visited[n]
                    min_node = n

        if min_node is None:
            print('min node outside search region, return lower bound')
            return lower_bound_dist * 1.2
        # display minimum path to goal
        rgb = [1, 0, 0]
        min_xy = node_to_pos(min_node)
        start = min_xy[0], min_xy[1], z
        while min_node != goal_node:
            next_node = path[min_node]
            next_xy = node_to_pos(next_node)
            next = next_xy[0], next_xy[1], z
            p.addUserDebugLine(start, next, rgb)
            start = next
            min_node = next_node

        print('min dist: {} lower bound: {}'.format(min_dist, lower_bound_dist))
        env.close()
        return dists


task_map = {'freespace': 0, 'Block-H': 5, 'Block-D': 6}

parser = argparse.ArgumentParser(description='Experiments on the planar pushing environment')
parser.add_argument('command',
                    choices=['collect', 'learn_representation', 'fine_tune_dynamics', 'run', 'evaluate', 'visualize',
                             'debug'],
                    help='which part of the experiment to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
tsf_map = {'none': UseTsf.NO_TRANSFORM, 'coordinate_transform': UseTsf.COORD, 'learned_rex': UseTsf.REX_EXTRACT,
           'rex_ablation': UseTsf.EXTRACT, 'extractor_ablation': UseTsf.SEP_DEC, 'skip_z': UseTsf.SKIP}
parser.add_argument('--representation', default='none',
                    choices=tsf_map.keys(),
                    help='representation to use for nominal dynamics')
parser.add_argument('--rep_name', default=None, type=str,
                    help='name and seed of a learned representation to use')
parser.add_argument('--gui', action='store_true', help='force GUI for some commands that default to not having GUI')
# learning parameters
parser.add_argument('--batch', metavar='N', type=int, default=500,
                    help='learning parameter: batch size')
# run parameters
parser.add_argument('--task', default=list(task_map.keys())[0], choices=task_map.keys(),
                    help='run parameter: what task to run')
parser.add_argument('--num_frames', metavar='N', type=int, default=500,
                    help='run parameter: number of simulation frames to run')
parser.add_argument('--no_trap_cost', action='store_true', help='run parameter: turn off trap set cost')
parser.add_argument('--nonadaptive_baseline', action='store_true',
                    help='run parameter: use non-adaptive baseline options')
parser.add_argument('--adaptive_baseline', action='store_true', help='run parameter: use adaptive baseline options')
parser.add_argument('--random_ablation', action='store_true', help='run parameter: use random recovery policy options')
parser.add_argument('--visualize_rollout', action='store_true',
                    help='run parameter: visualize MPC rollouts (slows down running)')

# controller parameters
parser.add_argument('--tampc_param', nargs='*', type=util.param_type, default=[],
                    help="run parameter: high level controller parameters")
parser.add_argument('--mpc_param', nargs='*', type=util.param_type, default=[],
                    help="run parameter: low level MPC parameters")
# evaluate parameters
parser.add_argument('--eval_run_prefix', default=None, type=str,
                    help='evaluate parameter: prefix of saved runs to evaluate performance on')

args = parser.parse_args()

if __name__ == "__main__":
    ut = tsf_map[args.representation]
    level = task_map[args.task]

    tampc_params = {}
    for d in args.tampc_param:
        tampc_params.update(d)
    mpc_params = {}
    for d in args.mpc_param:
        mpc_params.update(d)

    if args.command == 'collect':
        OfflineDataCollection.freespace(seed=args.seed[0], trials=200, trial_length=50, force_gui=args.gui)
        OfflineDataCollection.push_against_wall_recovery()
    elif args.command == 'learn_representation':
        for seed in args.seed:
            Learn.invariant(ut, seed=seed, name="", MAX_EPOCH=3000, BATCH_SIZE=args.batch)
    elif args.command == 'fine_tune_dynamics':
        Learn.model(ut, seed=args.seed[0], name="", rep_name=args.rep_name)
    elif args.command == 'run':
        nominal_adapt = OnlineAdapt.NONE
        autonomous_recovery = online_controller.AutonomousRecovery.MAB
        use_trap_cost = not args.no_trap_cost

        if args.adaptive_baseline:
            nominal_adapt = OnlineAdapt.GP_KERNEL_INDEP_OUT
            autonomous_recovery = online_controller.AutonomousRecovery.NONE
            use_trap_cost = False
            ut = UseTsf.NO_TRANSFORM
        elif args.random_ablation:
            autonomous_recovery = online_controller.AutonomousRecovery.RANDOM
        elif args.nonadaptive_baseline:
            autonomous_recovery = online_controller.AutonomousRecovery.NONE
            use_trap_cost = False
            ut = UseTsf.NO_TRANSFORM

        for seed in args.seed:
            test_autonomous_recovery(seed=seed, level=level, use_tsf=ut,
                                     nominal_adapt=nominal_adapt, rep_name=args.rep_name,
                                     reuse_escape_as_demonstration=False, use_trap_cost=use_trap_cost,
                                     assume_all_nonnominal_dynamics_are_traps=False, num_frames=args.num_frames,
                                     visualize_rollout=args.visualize_rollout,
                                     override_tampc_params=tampc_params, override_mpc_params=mpc_params,
                                     autonomous_recovery=autonomous_recovery)
    elif args.command == 'evaluate':
        util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
                                                args.eval_run_prefix, suffix="{}.mat".format(args.num_frames))
    elif args.command == 'visualize':
        task_names = {v: k for k, v in task_map.items()}
        util.plot_task_res_dist({
            'auto_recover__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green', 'label': True},
            # 'auto_recover__NONE__RANDOM__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC random', 'color': 'orange', 'label': True},
            # 'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2': {
            #     'name': 'TAMPC skip z (aggregate)', 'color': 'black', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s2': {
                'name': 'TAMPC skip z (seed 2)', 'color': 'black', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s3': {
                'name': 'TAMPC skip z (seed 3)', 'color': 'red', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s4': {
                'name': 'TAMPC skip z (seed 4)', 'color': 'blue', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s5': {
                'name': 'TAMPC skip z (seed 5)', 'color': 'cyan', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s6': {
                'name': 'TAMPC skip z (seed 6)', 'color': 'magenta', 'label': True},
            'auto_recover__NONE__MAB__5__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_RAL_s7': {
                'name': 'TAMPC skip z (seed 7)', 'color': 'orange', 'label': True},
            # 'auto_recover__NONE__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'non-adapative', 'color': 'purple', 'label': True},
            # 'auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'adaptive baseline++', 'color': 'red', 'label': True},
            # 'sac__5': {'name': 'SAC', 'color': 'cyan', 'label': True},

            # 'auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC', 'color': 'green'},
            # 'auto_recover__NONE__RANDOM__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC random', 'color': 'orange'},
            # 'auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'non-adapative', 'color': 'purple'},
            # 'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'adaptive baseline++', 'color': 'red'},
            # 'sac__6': {'name': 'SAC', 'color': 'cyan'},
        }, 'pushing_task_res.pkl', expected_data_len=args.num_frames - 1, figsize=(5, 7), task_names=task_names,
            success_min_dist=0.5)

    else:
        for seed in range(10):
            Learn.invariant(UseTsf.SKIP, seed=seed, name="prenormalized", MAX_EPOCH=3000, BATCH_SIZE=500)
        # for seed in range(10):
        #     Learn.invariant(UseTsf.REX_SKIP, seed=seed, name="prenormalized", MAX_EPOCH=3000, BATCH_SIZE=2048)
        pass
        # tune_trap_set_cost(seed=0, level=1, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
        #                    use_trap_cost=True,
        #                    autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)

        # tune_recovery_policy(seed=0, level=1, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
        #                      autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)
