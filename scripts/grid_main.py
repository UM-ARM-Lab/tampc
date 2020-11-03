import torch
import pickle
import time
import typing
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
import pprint

try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import rospy

    rospy.init_node("tampc_env")
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except:
    print("Proceeding without ROS")

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from tampc import cfg
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import hybrid_model
from tampc.env import gridworld

from tampc.dynamics.hybrid_model import OnlineAdapt
from tampc.controller import online_controller
from tampc.controller.gating_function import AlwaysSelectNominal
from tampc import util
from tampc.util import no_tsf_preprocessor, UseTsf, EnvGetter
from window_recorder import recorder

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

# --- SHARED GETTERS
task_map = {'freespace': 0, 'I': 1, 'I-non-nominal': 4, 'U': 2, 'Non-nominal': 3}


class GridGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "grid"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = gridworld.GridDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

    @staticmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        return preprocess.PytorchTransformer(preprocess.MinMaxScaler(), preprocess.NullSingleTransformer())

    @staticmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        d = get_device()
        u_min, u_max = env.get_control_bounds()
        Q = torch.tensor(env.state_cost(), dtype=torch.double)
        R = 0.00001  # has to be > 0 to measure whether we are inputting control effort
        # sigma = [2.5]
        # noise_mu = [2]
        # u_init = [2]
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
            'trap_cost_annealing_rate': 0.8,
            'abs_unrecognized_threshold': 0.5,
            'dynamics_minimum_window': 2,
            'max_trap_weight': 100,
        }
        mpc_opts = {
            'num_samples': 1000,
            'noise_sigma': torch.diag(sigma),
            'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
            'lambda_': 1e-2,
            'horizon': 5,
            'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
            'sample_null_action': False,
            'step_dependent_dynamics': True,
            'rollout_samples': 10,
            'rollout_var_cost': 0,
        }
        return common_wrapper_opts, mpc_opts

    @classmethod
    def env(cls, mode=0, level=0, log_video=False, **kwargs):
        # env = gridworld.GridEnv(environment_level=level, **kwargs)
        # cls.env_dir = '{}/raw'.format(gridworld.DIR)
        env = gridworld.GridContinuousEnv(environment_level=level, **kwargs)
        cls.env_dir = '{}/continuous'.format(gridworld.DIR)
        if level is task_map['I'] or level is task_map['freespace']:
            env.set_task_config(goal=[1, 6], init=[7, 6])
        elif level is task_map['I-non-nominal']:
            env.set_task_config(goal=[1, 6], init=[7, 6])
        return env


class OfflineDataCollection:
    @staticmethod
    def freespace(seed_offset=0, trials=200, trial_length=50, force_gui=False):
        env = GridGetter.env(level=0, check_boundaries=False)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(GridGetter.env_dir, 0)
        sim = gridworld.ExperimentRunner(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                         pause_s_between_steps=0.01, stop_when_done=False,
                                         save_dir=save_dir)
        rospy.sleep(0.5)
        sim.clear_markers()
        # randomly distribute data
        for offset in range(trials):
            seed = rand.seed(seed_offset + offset)
            # random position
            init = [int(np.random.random() * max_val) for max_val in env.size]
            env.set_task_config(init=init)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            with recorder.WindowRecorder(
                    window_names=("RViz*", "RViz", "gridworld.rviz - RViz", "gridworld.rviz* - RViz"),
                    name_suffix="rviz", frame_rate=30.0,
                    save_dir=cfg.VIDEO_DIR):
                sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()


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
                   **kwargs):
    env = GridGetter.env(level=level)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = GridGetter.prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local = GridGetter.ds(env, demo, validation_ratio=0.)
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

    tampc_opts, mpc_opts = GridGetter.controller_options(env)
    if override_tampc_params is not None:
        tampc_opts.update(override_tampc_params)
    if override_mpc_params is not None:
        mpc_opts.update(override_mpc_params)

    logger.debug("running with parameters\nhigh level controller: %s\nlow level MPC: %s",
                 pprint.pformat(tampc_opts), pprint.pformat(mpc_opts))

    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                        autonomous_recovery=autonomous_recovery,
                                        assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                        reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                        use_trap_cost=use_trap_cost,
                                        **tampc_opts,
                                        mpc_opts=mpc_opts)

    ctrl.set_goal(env.goal)

    sim = gridworld.ExperimentRunner(env, ctrl, num_frames=num_frames, plot=False, save=True, stop_when_done=True)
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
            # TODO also include model name
            return tsf_name

        run_name = default_run_prefix
        if run_prefix is not None:
            affix_run_name(run_prefix)
        affix_run_name(nominal_adapt.name)
        affix_run_name(autonomous_recovery.name + ("_WITHDEMO" if use_demo else ""))
        affix_run_name(level)
        affix_run_name(use_tsf.name)
        affix_run_name("ALLTRAP" if assume_all_nonnominal_dynamics_are_traps else "SOMETRAP")
        affix_run_name("REUSE" if reuse_escape_as_demonstration else "NOREUSE")
        affix_run_name(gating.name)
        affix_run_name("TRAPCOST" if use_trap_cost else "NOTRAPCOST")
        affix_run_name(get_rep_model_name(ds))
        affix_run_name(seed)
        affix_run_name(num_frames)

    time.sleep(1)
    sim.clear_markers()
    time.sleep(1)
    sim.dd.draw_text("seed", "s{}".format(seed), 1, left_offset=-1.4)
    sim.dd.draw_text("recovery_method", "recovery {}".format(autonomous_recovery.name), 2, left_offset=-1.4)
    if reuse_escape_as_demonstration:
        sim.dd.draw_text("resuse", "reuse escape", 3, left_offset=-1.4)
    sim.dd.draw_text("run_name", run_name, 18, left_offset=-0.8)
    with recorder.WindowRecorder(
            window_names=("RViz*", "RViz", "gridworld.rviz - RViz", "gridworld.rviz* - RViz"),
            name_suffix="rviz", frame_rate=30.0,
            save_dir=cfg.VIDEO_DIR):
        pre_run_setup(env, ctrl, ds, sim)

        sim.run(seed, run_name)
        logger.info("last run cost %f", np.sum(sim.last_run_cost))
        time.sleep(2)
    plt.ioff()
    plt.show()


def test_autonomous_recovery(*args, **kwargs):
    def default_setup(env, ctrl, ds, sim):
        return

    run_controller('auto_recover', default_setup, *args, **kwargs)


def compute_traps(*args, **kwargs):
    def default_setup(env, ctrl, ds, sim):
        sim.dd.draw_board(env)
        if ctrl.goal is not None:
            sim.dd.draw_goal(ctrl.goal)

        traps = env.compute_true_trap_difficulty(ctrl)
        sim.dd.draw_trap_difficulty(traps)
        rospy.sleep(1.5)
        input('press enter to start')
        exit(0)

    run_controller('draw_traps', default_setup, *args, **kwargs)


parser = argparse.ArgumentParser(description='Experiments on the 2D grid environment')
parser.add_argument('command',
                    choices=['collect', 'learn_representation', 'fine_tune_dynamics', 'run', 'evaluate', 'visualize1',
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
parser.add_argument('--run_prefix', default=None, type=str,
                    help='run parameter: prefix to save the run under')
parser.add_argument('--num_frames', metavar='N', type=int, default=200,
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
    task_names = {v: k for k, v in task_map.items()}
    tampc_params = {}
    for d in args.tampc_param:
        tampc_params.update(d)
    mpc_params = {}
    for d in args.mpc_param:
        mpc_params.update(d)

    if args.command == 'collect':
        OfflineDataCollection.freespace(seed_offset=0, trials=100, trial_length=30, force_gui=args.gui)
    elif args.command == 'learn_representation':
        for seed in args.seed:
            GridGetter.learn_invariant(ut, seed=seed, name=gridworld.DIR, MAX_EPOCH=1000, BATCH_SIZE=args.batch)
    elif args.command == 'fine_tune_dynamics':
        GridGetter.learn_model(ut, seed=args.seed[0], name="", rep_name=args.rep_name, train_epochs=1000)
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
            compute_traps(seed=seed, level=level, use_tsf=ut,
                          nominal_adapt=nominal_adapt, rep_name=args.rep_name,
                          reuse_escape_as_demonstration=False, use_trap_cost=use_trap_cost,
                          assume_all_nonnominal_dynamics_are_traps=False, num_frames=args.num_frames,
                          visualize_rollout=args.visualize_rollout, run_prefix=args.run_prefix,
                          override_tampc_params=tampc_params, override_mpc_params=mpc_params,
                          autonomous_recovery=autonomous_recovery)
    elif args.command == 'evaluate':
        task_type = gridworld.DIR
        trials = ["{}/{}".format(task_type, filename) for filename in os.listdir(os.path.join(cfg.DATA_DIR, task_type))
                  if filename.startswith(args.eval_run_prefix)]
        # get all the trials to visualize for choosing where the obstacles are
        # EvaluateTask.closest_distance_to_goal(trials, level=level, just_get_ok_nodes=True)
        # util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
        #                                         args.eval_run_prefix, task_type=task_type)
    elif args.command == 'visualize1':
        pass
    else:
        use_tsf = UseTsf.NO_TRANSFORM
        d, env, config, ds = GridGetter.free_space_env_init(0)
        ds.update_preprocessor(GridGetter.pre_invariant_preprocessor(use_tsf=use_tsf))
        xu, y, trial = ds.training_set(original=True)
        ds, pm = GridGetter.prior(env, use_tsf)
        yhat = pm.dyn_net.predict(xu, get_next_state=False, return_in_orig_space=True)
        u = xu[:, env.nx:]
        f, axes = plt.subplots(2, 1, figsize=(10, 9))
        dims = ['x', 'y']
        for i, dim in enumerate(dims):
            axes[i].scatter(u[:, i].cpu(), yhat[:, i].cpu(), color="red")
            axes[i].scatter(u[:, i].cpu(), y[:, i].cpu())
            axes[i].set_ylabel('d{}'.format(dim))
            axes[i].set_xlabel('u{}'.format(i))

        plt.show()
