import torch
import pickle
import time
import math
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
import pprint
import typing

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from tampc import cfg
from tampc.env import peg_in_hole
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import model, hybrid_model

from arm_pytorch_utilities.model import make

from tampc.dynamics.hybrid_model import OnlineAdapt
from tampc.controller import online_controller
from tampc.controller.gating_function import AlwaysSelectNominal
from tampc import util
from tampc.util import update_ds_with_transform, no_tsf_preprocessor, UseTsf, get_transform, TranslationNetworkWrapper, \
    EnvGetter

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


# --- SHARED GETTERS
class PegGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "peg"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = peg_in_hole.PegInHoleDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

    @staticmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        if use_tsf is UseTsf.COORD:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer())
        elif use_tsf is UseTsf.FEEDFORWARD_BASELINE:
            return util.no_tsf_preprocessor()
        else:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer(),
                                                 preprocess.RobustMinMaxScaler())

    @staticmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        d = get_device()
        u_min, u_max = env.get_control_bounds()
        Q = torch.tensor(env.state_cost(), dtype=torch.double)
        R = 0.01
        sigma = [0.2, 0.2]
        noise_mu = [0, 0]
        u_init = [0, 0]
        sigma = torch.tensor(sigma, dtype=torch.double, device=d)
        common_wrapper_opts = {
            'Q': Q,
            'R': R,
            'u_min': u_min,
            'u_max': u_max,
            'compare_to_goal': env.state_difference,
            'state_dist': env.state_distance,
            'u_similarity': env.control_similarity,
            'device': d,
            'terminal_cost_multiplier': 50,
            'trap_cost_annealing_rate': 0.9,
            'abs_unrecognized_threshold': 15 / 1.2185,  # to account for previous runs with bug in error
            # 'nonnominal_dynamics_penalty_tolerance': 0.1,
            # 'dynamics_minimum_window': 15,
        }
        mpc_opts = {
            'num_samples': 500,
            'noise_sigma': torch.diag(sigma),
            'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
            'lambda_': 1e-2,
            'horizon': 10,
            'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
            'sample_null_action': False,
            'step_dependent_dynamics': True,
            'rollout_samples': 10,
            'rollout_var_cost': 0,
        }
        return common_wrapper_opts, mpc_opts

    @classmethod
    def env(cls, mode=p.GUI, level=0, log_video=False):
        init_peg = [-0.2, 0]
        hole_pos = [0.3, 0.3]

        if level is 2:
            init_peg = [0, -0.2]
            hole_pos = [0, 0.2]

        if level in [3, 5, 8]:
            init_peg = [0, -0.05]
            hole_pos = [0, 0.2]

        if level is 4:
            init_peg = [-0.15, 0.2]
            hole_pos = [0, 0.2]

        if level is 6:
            init_peg = [0.15, 0.07]

        if level is 7:
            init_peg = [0.15 + 10, 0.07 + 10]

        if level is 10:
            init_peg = [0., 0.05]
            hole_pos = [0, 0.2]

        env_opts = {
            'mode': mode,
            'hole': hole_pos,
            'init_peg': init_peg,
            'log_video': log_video,
            'environment_level': level,
        }
        env = peg_in_hole.PegFloatingGripperEnv(**env_opts)
        cls.env_dir = 'peg/floating'
        return env

    @classmethod
    def learn_invariant(cls, use_tsf=UseTsf.REX_EXTRACT, seed=1, name="", MAX_EPOCH=1000, BATCH_SIZE=500, resume=False,
                        **kwargs):
        d, env, config, ds = cls.free_space_env_init(seed)
        ds.update_preprocessor(cls.pre_invariant_preprocessor(use_tsf))
        invariant_cls = get_transform(env, ds, use_tsf).__class__
        ds_test = cls.ds(env, "peg/peg_contact_test_set.mat", validation_ratio=0.)
        common_opts = {'name': "{}_s{}".format(name, seed), 'ds_test': [ds_test]}
        invariant_tsf = invariant_cls(ds, d, **common_opts, **kwargs)
        if resume:
            invariant_tsf.load(invariant_tsf.get_last_checkpoint())
        invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)


class OfflineDataCollection:
    @staticmethod
    def random_config(env):
        hole = (np.random.random((2,)) - 0.5)
        init_peg = (np.random.random((2,)) - 0.5)
        return hole, init_peg

    @staticmethod
    def freespace(seed=4, trials=200, trial_length=50, force_gui=False):
        env = PegGetter.env(p.GUI if force_gui else p.DIRECT, 0)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(PegGetter.env_dir, 0)
        sim = peg_in_hole.PegInHole(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                    stop_when_done=False, save_dir=save_dir)
        rand.seed(seed)
        # randomly distribute data
        for _ in range(trials):
            seed = rand.seed()
            # start at fixed location
            hole, init_peg = OfflineDataCollection.random_config(env)
            env.set_task_config(hole=hole, init_peg=init_peg)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()

    @staticmethod
    def test_set():
        # get data in and around the bug trap we want to avoid in the future
        env = PegGetter.env(p.GUI, task_map['Peg-T'])
        env.set_task_config(init_peg=[0.1, 0.12])

        def rn(scale):
            return np.random.randn() * scale

        u = []
        seed = rand.seed(2)
        for _ in range(5):
            u.append([0.4, 0.7 + rn(0.5)])
        for i in range(15):
            u.append([-0.0 + (i - 7) * 0.1, 0.8 + rn(0.5)])
        for i in range(15):
            u.append([-0.8 + rn(0.2), -0. + (i - 7) * 0.1])
        for i in range(5):
            u.append([-0.1 + rn(0.1), -1.])
        u.append([-0.6, -0.])
        for i in range(10):
            u.append([-0. + rn(0.5), 0.9])

        ctrl = controller.PreDeterminedController(np.array(u), *env.get_control_bounds())
        sim = peg_in_hole.PegInHole(env, ctrl, num_frames=len(u), plot=False, save=True,
                                    stop_when_done=False)
        sim.run(seed, 'peg_contact_test_set')


def run_controller(default_run_prefix, pre_run_setup, seed=1, level=1, gating=None,
                   use_tsf=UseTsf.COORD, nominal_adapt=OnlineAdapt.NONE,
                   autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE,
                   use_demo=False,
                   use_trap_cost=True,
                   reuse_escape_as_demonstration=False, num_frames=200,
                   run_prefix=None, run_name=None,
                   assume_all_nonnominal_dynamics_are_traps=False,
                   rep_name=None,
                   visualize_rollout=False,
                   override_tampc_params=None,
                   override_mpc_params=None,
                   never_estimate_error=False,
                   apfvo_baseline=False,
                   apfsp_baseline=False,
                   **kwargs):
    env = PegGetter.env(p.GUI, level=level, log_video=True)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = PegGetter.prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local = PegGetter.ds(env, demo, validation_ratio=0.)
        ds_local.update_preprocessor(ds.preprocessor)
        dss.append(ds_local)

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, [use_tsf.name],
                                                       device=get_device(),
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': nominal_adapt},
                                                       local_model_kwargs=kwargs)

    # we're always going to be in the nominal mode in this case; might as well speed up testing
    if not use_demo and not reuse_escape_as_demonstration:
        gating = AlwaysSelectNominal()
    else:
        gating = hybrid_dynamics.get_gating() if gating is None else gating

    tampc_opts, mpc_opts = PegGetter.controller_options(env)
    if override_tampc_params is not None:
        tampc_opts.update(override_tampc_params)
    if override_mpc_params is not None:
        mpc_opts.update(override_mpc_params)

    logger.debug("running with parameters\nhigh level controller: %s\nlow level MPC: %s",
                 pprint.pformat(tampc_opts), pprint.pformat(mpc_opts))

    if apfvo_baseline or apfsp_baseline:
        tampc_opts.pop('trap_cost_annealing_rate')
        tampc_opts.pop('abs_unrecognized_threshold')
        if apfvo_baseline:
            rho = 0.05
            if level == task_map['Peg-I']:
                rho = 0.04  # use lower value to prevent obstacle detected below to prevent us from entering the goal
            elif level == task_map['Peg-U']:
                rho = 0.025  # use lower value to place more dense virtual obstacles to increase chance of entering
            ctrl = online_controller.APFVO(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           local_min_threshold=0.005, trap_max_dist_influence=rho, repulsion_gain=0.01,
                                           **tampc_opts)
            env.draw_user_text("APF-VO baseline", 13, left_offset=-1.5)
        if apfsp_baseline:
            # anything lower leads to oscillation between backing up and entering the trap's field of influence
            rho = 0.07
            if level == task_map['Peg-U']:
                rho = 0.055
            ctrl = online_controller.APFSP(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           trap_max_dist_influence=rho, backup_scale=0.7,
                                           **tampc_opts)
            env.draw_user_text("APF-SP baseline", 13, left_offset=-1.5)
    else:
        ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                            autonomous_recovery=autonomous_recovery,
                                            assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                            reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                            use_trap_cost=use_trap_cost,
                                            never_estimate_error_dynamics=never_estimate_error,
                                            **tampc_opts,
                                            mpc_opts=mpc_opts)
        env.draw_user_text(gating.name, 13, left_offset=-1.5)
        env.draw_user_text("recovery {}".format(autonomous_recovery.name), 11, left_offset=-1.6)
        if reuse_escape_as_demonstration:
            env.draw_user_text("reuse escape", 10, left_offset=-1.6)
        if use_trap_cost:
            env.draw_user_text("trap set cost".format(autonomous_recovery.name), 9, left_offset=-1.6)
    env.draw_user_text("run seed {}".format(seed), 12, left_offset=-1.5)

    z = env.initGripperPos[2]
    goal = np.r_[env.hole, z, 0, 0]
    ctrl.set_goal(goal)
    # env._dd.draw_point('hole', env.hole, color=(0, 0.5, 0.8))

    sim = peg_in_hole.PegInHole(env, ctrl, num_frames=num_frames, plot=False, save=True, stop_when_done=True,
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
        if apfvo_baseline:
            run_prefix = 'APFVO'
        elif apfsp_baseline:
            run_prefix = 'APFSP'
        if run_prefix is not None:
            affix_run_name(run_prefix)
        affix_run_name(nominal_adapt.name)
        if not apfvo_baseline and not apfsp_baseline:
            affix_run_name(autonomous_recovery.name + ("_WITHDEMO" if use_demo else ""))
        if never_estimate_error:
            affix_run_name('NO_E')
        affix_run_name(level)
        affix_run_name(use_tsf.name)
        affix_run_name("ALLTRAP" if assume_all_nonnominal_dynamics_are_traps else "SOMETRAP")
        affix_run_name("REUSE" if reuse_escape_as_demonstration else "NOREUSE")
        affix_run_name(gating.name)
        affix_run_name("TRAPCOST" if use_trap_cost else "NOTRAPCOST")
        affix_run_name(get_rep_model_name(ds))
        affix_run_name(seed)
        affix_run_name(num_frames)

    env.draw_user_text(run_name, 14, left_offset=-1.5)

    pre_run_setup(env, ctrl, ds)

    sim.run(seed, run_name)
    logger.info("last run cost %f", np.sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


def test_autonomous_recovery(*args, **kwargs):
    def default_setup(env, ctrl, ds):
        return

    run_controller('auto_recover', default_setup, *args, **kwargs)


def tune_trap_set_cost(*args, num_frames=100, **kwargs):
    def setup(env, ctrl, ds):
        z = env.initGripperPos[2]

        level = kwargs['level']
        if level is 0:
            hole = [0, 0.2]
            x = [hole[0], hole[1] - 0.3, z, 0, 0]
            env.set_task_config(hole=hole, init_peg=x[:2])
            x = torch.tensor(x, device=ctrl.d, dtype=ctrl.dtype)
            goal = np.r_[env.hole, z, 0, 0]
            ctrl.set_goal(goal)

            # setup initial conditions where we are close to a trap and have items in our trap set
            trap_x = torch.tensor([env.hole[0], env.hole[1] - 0.2, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype)
            trap_u = torch.tensor([0, -1], device=ctrl.d, dtype=ctrl.dtype)
            ctrl.trap_set.append((trap_x, trap_u))

            ctrl.trap_set.append(
                (torch.tensor([env.hole[0] - 0.1, env.hole[1] - 0.2, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype),
                 torch.tensor([0, -1], device=ctrl.d, dtype=ctrl.dtype)))
            # test with explicit seeding on nominal trajectory
            # ctrl.mpc.U = torch.tensor([0, 0.5], device=ctrl.d, dtype=ctrl.dtype).repeat(ctrl.original_horizon, 1)

            ctrl.trap_set_weight = ctrl.normalize_trapset_cost_to_state(x)

            perturbations = [[0.05, 0.], [0.1, 0.], [0.15, 0]]
            for perturbation in perturbations:
                shifted_x = trap_x.clone()
                shifted_x[0] += perturbation[0]
                shifted_x[1] += perturbation[1]

                actions = []
                costs = []
                for angle in np.linspace(0, 2 * np.pi, 36):
                    action = [math.cos(angle), math.sin(angle)]
                    actions.append(action)
                    costs.append(
                        ctrl.trap_cost(shifted_x, torch.tensor(action, device=ctrl.d, dtype=ctrl.dtype)).item())

                costs = np.array(costs)
                logger.debug(costs)
                env.draw_user_text("{:.5f}".format(np.max(costs)), 3)
                costs = costs / np.max(costs)

                env.visualize_trap_set(ctrl.trap_set)
                for i in range(len(actions)):
                    normalization = costs[i]
                    u = [a * normalization for a in actions[i]]
                    env._draw_action(u, old_state=shifted_x.cpu().numpy(), debug=i + 2)

                time.sleep(2)

        elif level is 5:
            ctrl.trap_set.extend([(torch.tensor([-3.5530e-03, 1.4122e-01, 1.9547e-01, 7.1541e-01, -1.0235e+01],
                                                device='cuda:0', dtype=torch.float64),
                                   torch.tensor([0.3596, 0.2701], device='cuda:0', dtype=torch.float64)),
                                  (torch.tensor([-0.0775, 0.1415, 0.1956, -9.7136, -10.1923], device='cuda:0',
                                                dtype=torch.float64),
                                   torch.tensor([0.2633, 0.3216], device='cuda:0', dtype=torch.float64)),
                                  (torch.tensor([0.1663, 0.1417, 0.1956, 4.4845, -10.2436], device='cuda:0',
                                                dtype=torch.float64),
                                   torch.tensor([1.0000, 0.8479], device='cuda:0', dtype=torch.float64)),
                                  (torch.tensor([0.0384, 0.1412, 0.1955, 4.6210, -10.4069], device='cuda:0',
                                                dtype=torch.float64),
                                   torch.tensor([0.3978, 0.6424], device='cuda:0', dtype=torch.float64))])
            ctrl.trap_set_weight = torch.tensor([0.0035], device='cuda:0', dtype=torch.float64)
            x = [-3.35635743e-03, 3.24323310e-01, 1.95463138e-01, 5.71478642e+00, -4.01359529e+00]
            env.set_task_config(init_peg=x[:2])

    run_controller('tune_trap_cost', setup, *args, num_frames=num_frames, **kwargs)


def tune_recovery_policy(*args, num_frames=100, **kwargs):
    def setup(env, ctrl: online_controller.OnlineMPPI, ds):
        # setup initial conditions where we are close to a trap and have items in our trap set
        ctrl.nominal_max_velocity = 0.012

        z = env.initGripperPos[2]
        hole = [0, 0.2]
        x = [hole[0], hole[1] - 0.2, z, 0, 0]
        env.set_task_config(hole=hole, init_peg=x[:2])
        goal = np.r_[env.hole, z, 0, 0]
        ctrl.set_goal(goal)

        ctrl.trap_set.append((torch.tensor([env.hole[0], env.hole[1] - 0.1, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype),
                              torch.tensor([0, -1], device=ctrl.d, dtype=ctrl.dtype)))
        ctrl.nominal_dynamic_states = [
            [torch.tensor([hole[0], hole[1] - 0.5, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype)]]

        ctrl._start_recovery_mode()

    run_controller('tune_recovery', setup, *args, num_frames=num_frames, use_trap_cost=False, **kwargs)


def evaluate_after_rollout(rollout_file, rollout_stop_index, *args, num_frames=100, **kwargs):
    def setup(env, ctrl: online_controller.OnlineMPPI, ds):
        ds_eval = PegGetter.ds(env, rollout_file, validation_ratio=0.)
        ds_eval.update_preprocessor(ds.preprocessor)

        # evaluate on a non-recovery dataset to see if rolling out the actions from the recovery set is helpful
        XU, Y, info = ds_eval.training_set(original=True)
        X, U = torch.split(XU, env.nx, dim=1)

        # put the state right before the evaluated action
        x = X[rollout_stop_index].cpu().numpy()
        logger.info(np.array2string(x, separator=', '))
        # only need to do rollouts; don't need control samples
        T = ctrl.mpc.T
        ctrl.original_horizon = 1
        for i in range(rollout_stop_index):
            env.draw_user_text(str(i), 1)
            env.set_state(X[i].cpu().numpy(), U[i].cpu().numpy())
            ctrl.mpc.change_horizon(1)
            ctrl.command(X[i].cpu().numpy())

        ctrl.original_horizon = T
        ctrl.mpc.change_horizon(T)

        # manually evaluate cost near goal when we're not taking actions downwards

        # setup initial conditions where we are close to a trap and have items in our trap set
        z = env.initGripperPos[2]
        x = torch.tensor([env.hole[0], env.hole[1] + 0.1, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype)
        c = ctrl.trap_cost(x, torch.tensor([0, -1], device=ctrl.d, dtype=ctrl.dtype))

    run_controller('evaluate_after_rollout', setup, *args, num_frames=num_frames, **kwargs)


class EvaluateTask:
    @staticmethod
    def closest_distance_to_goal(file, level, visualize=True, nodes_per_side=150):
        from sklearn.preprocessing import MinMaxScaler
        env = PegGetter.env(p.GUI if visualize else p.DIRECT, level=level)
        ds = PegGetter.ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        if level is 1:
            min_pos = (-0.3, -0.3)
            max_pos = (0.5, 0.5)
        elif level is 3 or level is 8:
            min_pos = (-0.2, -0.1)
            max_pos = (0.2, 0.35)
        elif level is 5:
            min_pos = (-0.4, -0.1)
            max_pos = (0.4, 0.4)
        elif level is 6:
            min_pos = (-0.3, -0.1)
            max_pos = (0.3, 0.3)
        elif level is 7:
            translation = 10
            min_pos = (-0.3 + translation, -0.1 + translation)
            max_pos = (0.3 + translation, 0.3 + translation)
        else:
            raise RuntimeError("Unspecified range for level {}".format(level))

        scaler = MinMaxScaler(feature_range=(0, nodes_per_side - 1))
        scaler.fit(np.array([min_pos, max_pos]))

        reached_states = X[:, :2].cpu().numpy()
        goal_pos = env.hole[:2]

        lower_bound_dist = np.linalg.norm((reached_states - goal_pos), axis=1).min()

        def node_to_pos(node):
            return scaler.inverse_transform([node])[0]

        def pos_to_node(pos):
            pair = scaler.transform([pos])[0]
            node = tuple(int(round(v)) for v in pair)
            return node

        z = env.initPeg[2]
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
        fullname = os.path.join(cfg.DATA_DIR, 'ok_peg{}_{}.pkl'.format(level, nodes_per_side))
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                ok_nodes = pickle.load(f)
                logger.info("loaded ok nodes from %s", fullname)
        else:
            ok_nodes = [[None for _ in range(nodes_per_side)] for _ in range(nodes_per_side)]
            orientation = p.getQuaternionFromEuler([0, 0, 0])
            pointer = p.loadURDF(os.path.join(cfg.ROOT_DIR, "peg.urdf"), (0, 0, z))
            # discretize positions and show goal states
            xs = np.linspace(min_pos[0], max_pos[0], nodes_per_side)
            ys = np.linspace(min_pos[1], max_pos[1], nodes_per_side)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p.resetBasePositionAndOrientation(pointer, (x, y, z), orientation)
                    for wall in env.walls:
                        c = p.getClosestPoints(pointer, wall, 0.0)
                        if c:
                            break
                    else:
                        n = pos_to_node((x, y))
                        ok_nodes[i][j] = n

        with open(fullname, 'wb') as f:
            pickle.dump(ok_nodes, f)
            logger.info("saved ok nodes to %s", fullname)

        g = util.grid_to_graph(min_pos, max_pos, nodes_per_side, ok_nodes)

        goal_node = pos_to_node(goal_pos)
        if ok_nodes[goal_node[0]][goal_node[1]] is None:
            goal_node = (goal_node[0], goal_node[1] + 1)

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


task_map = {'freespace': 0, 'Peg-U': 3, 'Peg-I': 5, 'Peg-T': 6, 'Peg-T(T)': 7, 'Peg-H': 8}

parser = argparse.ArgumentParser(description='Experiments on the peg-in-hole environment')
parser.add_argument('command',
                    choices=['collect', 'learn_representation', 'fine_tune_dynamics', 'run', 'evaluate', 'visualize1',
                             'visualize2', 'debug'],
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
parser.add_argument('--run_prefix', default=None, type=str,
                    help='run parameter: prefix to save the run under')
parser.add_argument('--num_frames', metavar='N', type=int, default=500,
                    help='run parameter: number of simulation frames to run')
parser.add_argument('--always_estimate_error', action='store_true',
                    help='run parameter: always online estimate error dynamics using a GP')
parser.add_argument('--never_estimate_error', action='store_true',
                    help='run parameter: never online estimate error dynamics using a GP (always use e=0)')
parser.add_argument('--no_trap_cost', action='store_true', help='run parameter: turn off trap set cost')
parser.add_argument('--nonadaptive_baseline', action='store_true',
                    help='run parameter: use non-adaptive baseline options')
parser.add_argument('--adaptive_baseline', action='store_true', help='run parameter: use adaptive baseline options')
parser.add_argument('--apfvo_baseline', action='store_true',
                    help='run parameter: use artificial potential field virtual obstacles baseline')
parser.add_argument('--apfsp_baseline', action='store_true',
                    help='run parameter: use artificial potential field switched potential baseline')

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
    task_names = {v: k for k, v in task_map.items()}
    tampc_params = {}
    for d in args.tampc_param:
        tampc_params.update(d)
    mpc_params = {}
    for d in args.mpc_param:
        mpc_params.update(d)

    if args.command == 'collect':
        OfflineDataCollection.freespace(seed=args.seed[0], trials=200, trial_length=50, force_gui=args.gui)
        OfflineDataCollection.test_set()
    elif args.command == 'learn_representation':
        for seed in args.seed:
            PegGetter.learn_invariant(ut, seed=seed, name="peg", MAX_EPOCH=1000, BATCH_SIZE=args.batch)
    elif args.command == 'fine_tune_dynamics':
        PegGetter.learn_model(ut, seed=args.seed[0], name="", rep_name=args.rep_name)
    elif args.command == 'run':
        nominal_adapt = OnlineAdapt.NONE
        autonomous_recovery = online_controller.AutonomousRecovery.MAB
        use_trap_cost = not args.no_trap_cost

        if args.always_estimate_error:
            nominal_adapt = OnlineAdapt.GP_KERNEL_INDEP_OUT
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
                                     visualize_rollout=args.visualize_rollout, run_prefix=args.run_prefix,
                                     override_tampc_params=tampc_params, override_mpc_params=mpc_params,
                                     autonomous_recovery=autonomous_recovery,
                                     never_estimate_error=args.never_estimate_error,
                                     apfvo_baseline=args.apfvo_baseline,
                                     apfsp_baseline=args.apfsp_baseline)
    elif args.command == 'evaluate':
        util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
                                                args.eval_run_prefix, suffix="{}.mat".format(args.num_frames),
                                                task_type='peg')
    elif args.command == 'visualize1':
        util.plot_task_res_dist({
            'sac_3': {'name': 'SAC', 'color': 'cyan'},
            'auto_recover__NONE__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},
            'auto_recover__APFVO__NONE__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-VO', 'color': 'black'},
            # 'auto_recover__apflme5__NONE__MAB__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            # 'auto_recover__APFSP__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__RANDOM__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': [0.8, 0.8, 0]},
            'auto_recover__NONE__MAB__NO_E__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            # 'auto_recover__NONE__MAB__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h15_larger_min_window__NONE__MAB__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},

            'sac_5': {'name': 'SAC', 'color': 'cyan'},
            'auto_recover__NONE__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},
            'auto_recover__APFVO__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-VO', 'color': 'black'},
            # 'auto_recover__apflme5__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            # 'auto_recover__APFSP__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__RANDOM__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': [0.8, 0.8, 0]},
            'auto_recover__NONE__MAB__NO_E__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            # 'auto_recover__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h20_less_anneal__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
        }, 'peg_task_res.pkl', task_type='peg', figsize=(5, 7), set_y_label=False,
            task_names=task_names, success_min_dist=0.05)

    elif args.command == 'visualize2':
        util.plot_task_res_dist({
            # 'auto_recover__NONE__MAB__6__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal': {
            #     'name': 'TAMPC skip z', 'color': 'black'},
            'auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},
            'sac_6': {'name': 'SAC', 'color': 'cyan'},
            'auto_recover__APFVO__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-VO', 'color': 'black'},
            # 'auto_recover__apflme5__NONE__MAB__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            # 'auto_recover__APFSP__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__RANDOM__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': [0.8, 0.8, 0]},
            'auto_recover__NONE__MAB__NO_E__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            'auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},

            # 'auto_recover__NONE__MAB__7__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal': {
            #     'name': 'TAMPC skip z', 'color': 'black'},
            'auto_recover__NONE__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},
            'sac__7': {'name': 'SAC', 'color': 'cyan'},
            'auto_recover__APFVO__NONE__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-VO', 'color': 'black'},
            # 'auto_recover__apflme5__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            # 'auto_recover__APFSP__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC original space', 'color': 'olive', 'label': True},
            'auto_recover__NONE__RANDOM__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': [0.8, 0.8, 0]},
            'auto_recover__NONE__MAB__NO_E__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            'auto_recover__NONE__MAB__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
        }, 'peg_task_res.pkl', task_type='peg', figsize=(5, 7), set_y_label=False,
            task_names=task_names, success_min_dist=0.05)

    else:
        for seed in range(2):
            PegGetter.learn_invariant(UseTsf.FEEDFORWARD_BASELINE, seed=seed, name="t12",
                                      MAX_EPOCH=3000, BATCH_SIZE=500,
                                      dynamics_opts={'h_units': (16, 32, 32, 32)})
            # PegGetter.learn_invariant(UseTsf.FEEDFORWARD_BASELINE, seed=seed, name="t9",
            #                           MAX_EPOCH=500, BATCH_SIZE=500,
            #                           dynamics_opts={'h_units': (32, 32)})
        # # for seed in range(10):
        #     PegGetter.learn_invariant(UseTsf.REX_EXTRACT, seed=seed, name="t8",
        #                               MAX_EPOCH=500, BATCH_SIZE=2048)
        # # for seed in range(10):
        #     PegGetter.learn_invariant(UseTsf.SEP_DEC, seed=seed, name="ral",
        #                               MAX_EPOCH=500, BATCH_SIZE=500)
