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

import tampc.env.real_env

try:
    import rospy

    rospy.init_node("tampc_env")
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from tampc import cfg
from tampc.env import peg_in_hole_real
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import hybrid_model

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
class PegRealGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "pegr"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = peg_in_hole_real.PegRealDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

    @staticmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        if use_tsf is UseTsf.COORD:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer())
        elif use_tsf in [UseTsf.SKIP, UseTsf.REX_SKIP]:
            # normalize position and force dimensions separately using shared scales
            return preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler(dims_share_scale=[[0, 1], [3, 4]]),
                                                 preprocess.RobustMinMaxScaler(dims_share_scale=[[0, 1], [3, 4]]))
        else:
            return preprocess.PytorchTransformer(preprocess.NullSingleTransformer(),
                                                 preprocess.RobustMinMaxScaler(dims_share_scale=[[0, 1], [3, 4]]))

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
            'trap_cost_annealing_rate': 0.8,
            'abs_unrecognized_threshold': 15,
            # 'nonnominal_dynamics_penalty_tolerance': 0.1,
            'dynamics_minimum_window': 2,
            # 'trap_cost_init_normalization': 1.0,
            # 'manual_init_trap_weight': 0.02,
            'max_trap_weight': 0.01,
        }
        mpc_opts = {
            'num_samples': 1000,
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
    def env(cls, mode=0, level=0, log_video=False, **kwargs):
        env = peg_in_hole_real.RealPegEnv(environment_level=level, **kwargs)
        if level is task_map['Real Peg-T']:
            x = 1.73472827  # 1.74962708 - 0.001
            y = -0.00480442  # -0.02913485 + 0.011
            env.set_task_config(hole=[x, y], init_peg=[1.64363362, 0.05320179])
            # for tuning close to goal behaviour (spiral exploration vs going straight to goal)
            # env.set_task_config(hole=[x, y], init_peg=[x + 0.01, y + 0.01])
        elif level is task_map['Peg-U(W)'] or level is task_map['Real Peg-U']:
            x = 1.61988168 + 0.001 + 0.002
            y = 0.04864363 - 0.002  # 0.04154706 + 0.013
            env.set_task_config(hole=[x, y], init_peg=[1.53700509, 0.08727498])
            # for tuning close to goal behaviour (spiral exploration vs going straight to goal)
            # env.set_task_config(hole=[x, y], init_peg=[x + 0.0, y + 0.0])

        cls.env_dir = '{}/real'.format(peg_in_hole_real.DIR)
        return env


class OfflineDataCollection:
    @staticmethod
    def freespace(seed_offset=0, trials=200, trial_length=50, force_gui=False):
        env = PegRealGetter.env(level=0, stub=False)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(PegRealGetter.env_dir, 0)
        sim = peg_in_hole_real.ExperimentRunner(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                                stop_when_done=False, save_dir=save_dir)
        # randomly distribute data
        for offset in range(trials):
            seed = rand.seed(seed_offset + offset)
            move = input('specify dx and dy to move to')
            dx, dy = [float(dim) for dim in move.split()]
            env.reset([dx, dy])
            obs = env.state

            run_name = "{}_{}_{}_{}".format(seed, obs[0].round(3), obs[1].round(3), obs[2].round(3))
            # start at fixed location
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            with tampc.env.real_env.VideoLogger():
                sim.run(seed, run_name=run_name)

        env.close()
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
                   never_estimate_error=False,
                   apfvo_baseline=False,
                   apfsp_baseline=False,
                   **kwargs):
    env = PegRealGetter.env(level=level, stub=False)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = PegRealGetter.prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local = PegRealGetter.ds(env, demo, validation_ratio=0.)
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

    tampc_opts, mpc_opts = PegRealGetter.controller_options(env)
    if override_tampc_params is not None:
        tampc_opts.update(override_tampc_params)
    if override_mpc_params is not None:
        mpc_opts.update(override_mpc_params)

    logger.debug("running with parameters\nhigh level controller: %s\nlow level MPC: %s",
                 pprint.pformat(tampc_opts), pprint.pformat(mpc_opts))

    if apfvo_baseline or apfsp_baseline:
        tampc_opts.pop('trap_cost_annealing_rate')
        tampc_opts.pop('abs_unrecognized_threshold')
        tampc_opts.pop('dynamics_minimum_window')
        tampc_opts.pop('max_trap_weight')
        if apfvo_baseline:
            ctrl = online_controller.APFVO(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           local_min_threshold=0.005, trap_max_dist_influence=0.02,
                                           repulsion_gain=0.01,
                                           **tampc_opts)
        if apfsp_baseline:
            ctrl = online_controller.APFSP(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           trap_max_dist_influence=0.045,
                                           **tampc_opts)
    else:
        ctrl = online_controller.TAMPC(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                       autonomous_recovery=autonomous_recovery,
                                       assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                       reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                       never_estimate_error_dynamics=never_estimate_error,
                                       use_trap_cost=use_trap_cost,
                                       **tampc_opts)
        mpc = controller.ExperimentalMPPI(ctrl.mpc_apply_dynamics, ctrl.mpc_running_cost, ctrl.nx,
                                          u_min=ctrl.u_min, u_max=ctrl.u_max,
                                          terminal_state_cost=ctrl.mpc_terminal_cost,
                                          device=ctrl.d, **mpc_opts)
        ctrl.register_mpc(mpc)

    z = 0.98
    goal = np.r_[env.hole, z, 0, 0]
    ctrl.set_goal(goal)

    sim = peg_in_hole_real.ExperimentRunner(env, ctrl, num_frames=num_frames, plot=False, save=True,
                                            stop_when_done=True)
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

    time.sleep(1)
    sim.clear_markers()
    time.sleep(1)
    sim.dd.draw_text("seed", "s{}".format(seed), 1, left_offset=-1.4)
    sim.dd.draw_text("recovery_method", "recovery {}".format(autonomous_recovery.name), 2, left_offset=-1.4)
    if reuse_escape_as_demonstration:
        sim.dd.draw_text("resuse", "reuse escape", 3, left_offset=-1.4)
    sim.dd.draw_text("run_name", run_name, 18, left_offset=-0.8, scale=3)
    with tampc.env.real_env.VideoLogger():
        pre_run_setup(env, ctrl, ds)

        sim.run(seed, run_name)
        logger.info("last run cost %f", np.sum(sim.last_run_cost))
        time.sleep(2)
    plt.ioff()
    plt.show()


def test_autonomous_recovery(*args, **kwargs):
    def default_setup(env, ctrl, ds):
        return

    run_controller('auto_recover', default_setup, *args, **kwargs)


class EvaluateTask:
    @staticmethod
    def closest_distance_to_goal(file, level, just_get_ok_nodes=False, visualize=True, nodes_per_side=150):
        from sklearn.preprocessing import MinMaxScaler
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        env = PegRealGetter.env(level=level)
        ds = PegRealGetter.ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        if level is task_map['Real Peg-T']:
            min_pos = (1.5, -0.14)
            max_pos = (1.85, 0.18)
        elif level is task_map['Peg-U(W)'] or level is task_map['Real Peg-U']:
            min_pos = (1.49, -0.14)
            max_pos = (1.83, 0.188)
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

        dd = tampc.env.real_env.DebugRvizDrawer()
        z = X[0, 2].item()
        # draw search boundaries
        marker = dd.make_marker(marker_type=Marker.LINE_STRIP)
        marker.ns = "boundary"
        marker.id = 0
        marker.color.a = 1
        marker.color.r = 0
        marker.color.g = 0
        marker.color.b = 0
        marker.points = [Point(x=min_pos[0], y=min_pos[1], z=z), Point(x=max_pos[0], y=min_pos[1], z=z),
                         Point(x=max_pos[0], y=max_pos[1], z=z), Point(x=min_pos[0], y=max_pos[1], z=z),
                         Point(x=min_pos[0], y=min_pos[1], z=z)]
        dd.marker_pub.publish(marker)

        # draw previous trajectory
        marker = dd.make_marker(marker_type=Marker.POINTS)
        marker.ns = "state_trajectory"
        marker.id = 0
        for i in range(len(X)):
            p = reached_states[i]
            marker.points.append(Point(x=p[0], y=p[1], z=z))
        marker.color.a = 1
        marker.color.b = 1
        dd.marker_pub.publish(marker)

        # try to load it if possible
        fullname = os.path.join(cfg.DATA_DIR, 'ok_{}{}_{}.pkl'.format(peg_in_hole_real.DIR, level, nodes_per_side))
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                ok_nodes = pickle.load(f)
                logger.info("loaded ok nodes from %s", fullname)
        else:
            ok_nodes = [[None for _ in range(nodes_per_side)] for _ in range(nodes_per_side)]
            # discretize positions and show goal states
            xs = np.linspace(min_pos[0], max_pos[0], nodes_per_side)
            ys = np.linspace(min_pos[1], max_pos[1], nodes_per_side)
            # publish to rviz
            marker = dd.make_marker(scale=dd.BASE_SCALE * 0.3)
            marker.ns = "nodes"
            marker.id = 0
            marker.color.a = 1
            marker.color.g = 1
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    n = pos_to_node((x, y))
                    ok_nodes[i][j] = n
                    marker.points.append(Point(x=x, y=y, z=z))
            dd.marker_pub.publish(marker)

            while True:
                ij = input(
                    "enter i_start-i_end [0,{}], j_start-j_end [0,{}] to toggle node or q to finish".format(len(xs) - 1,
                                                                                                            len(
                                                                                                                ys) - 1))
                if ij.strip() == 'q':
                    break
                try:
                    i, j = ij.split(',')
                    # see if things can be broken down into interval
                    if '-' in i:
                        v_min, v_max = tuple(int(v) for v in i.split('-'))
                        i_interval = range(v_min, v_max + 1)
                        if v_min < 0 or v_min > v_max or v_max >= len(xs):
                            raise RuntimeError()
                    else:
                        i_interval = [int(i)]
                    if '-' in j:
                        v_min, v_max = tuple(int(v) for v in j.split('-'))
                        j_interval = range(v_min, v_max + 1)
                        if v_min < 0 or v_min > v_max or v_max >= len(ys):
                            raise RuntimeError()
                    else:
                        j_interval = [int(j)]
                except RuntimeError:
                    print("did not enter in correct format, try again")
                    continue

                marker = dd.make_marker(scale=dd.BASE_SCALE * 0.3)
                marker.ns = "nodes"
                marker.id = i_interval[0] * nodes_per_side + j_interval[0]

                for i in i_interval:
                    for j in j_interval:
                        xy = node_to_pos((i, j))
                        marker.points.append(Point(x=xy[0], y=xy[1], z=z + 0.0001))
                        # toggle whether this node is OK or not
                        if ok_nodes[i][j] is None:
                            ok_nodes[i][j] = (i, j)
                            marker.colors.append(ColorRGBA(r=0, g=1, b=0, a=1))
                        else:
                            ok_nodes[i][j] = None
                            marker.colors.append(ColorRGBA(r=1, g=0, b=0, a=1))
                dd.marker_pub.publish(marker)

        with open(fullname, 'wb') as f:
            pickle.dump(ok_nodes, f)
            logger.info("saved ok nodes to %s", fullname)

        if just_get_ok_nodes:
            return

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
        min_xy = node_to_pos(min_node)
        marker = dd.make_marker(marker_type=Marker.LINE_STRIP)
        marker.ns = "mindist"
        marker.id = 0
        marker.color.a = 1
        marker.color.r = 1
        marker.points = [Point(x=min_xy[0], y=min_xy[1], z=z)]
        while min_node != goal_node:
            next_node = path[min_node]
            next_xy = node_to_pos(next_node)
            marker.points.append(Point(x=next_xy[0], y=next_xy[1], z=z))
            min_node = next_node
        dd.marker_pub.publish(marker)

        print('min dist: {} lower bound: {}'.format(min_dist, lower_bound_dist))
        return dists


task_map = {'freespace': 0, 'Peg-U(W)': 3, 'Peg-I': 5, 'Real Peg-T': 6, 'Peg-T(T)': 7, 'Real Peg-U': 8}

parser = argparse.ArgumentParser(description='Experiments on the real peg-in-hole environment')
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
parser.add_argument('--num_frames', metavar='N', type=int, default=300,
                    help='run parameter: number of simulation frames to run')
parser.add_argument('--no_trap_cost', action='store_true', help='run parameter: turn off trap set cost')
parser.add_argument('--never_estimate_error', action='store_true',
                    help='run parameter: never online estimate error dynamics using a GP (always use e=0)')

parser.add_argument('--nonadaptive_baseline', action='store_true',
                    help='run parameter: use non-adaptive baseline options')
parser.add_argument('--adaptive_baseline', action='store_true', help='run parameter: use adaptive baseline options')
parser.add_argument('--random_ablation', action='store_true', help='run parameter: use random recovery policy options')
parser.add_argument('--apfvo_baseline', action='store_true',
                    help='run parameter: use artificial potential field virtual obstacles baseline')
parser.add_argument('--apfsp_baseline', action='store_true',
                    help='run parameter: use artificial potential field switched potential baseline')

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
        OfflineDataCollection.freespace(seed_offset=20, trials=5, trial_length=30, force_gui=args.gui)
    elif args.command == 'learn_representation':
        for seed in args.seed:
            PegRealGetter.learn_invariant(ut, seed=seed, name="pegr", MAX_EPOCH=1000, BATCH_SIZE=args.batch)
    elif args.command == 'fine_tune_dynamics':
        PegRealGetter.learn_model(ut, seed=args.seed[0], name="", rep_name=args.rep_name)
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
                                     visualize_rollout=args.visualize_rollout, run_prefix=args.run_prefix,
                                     override_tampc_params=tampc_params, override_mpc_params=mpc_params,
                                     autonomous_recovery=autonomous_recovery,
                                     never_estimate_error=args.never_estimate_error,
                                     apfvo_baseline=args.apfvo_baseline,
                                     apfsp_baseline=args.apfsp_baseline)
    elif args.command == 'evaluate':
        task_type = peg_in_hole_real.DIR
        trials = ["{}/{}".format(task_type, filename) for filename in os.listdir(os.path.join(cfg.DATA_DIR, task_type))
                  if filename.startswith(args.eval_run_prefix)]
        # get all the trials to visualize for choosing where the obstacles are
        # EvaluateTask.closest_distance_to_goal(trials, level=level, just_get_ok_nodes=True)

        util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
                                                args.eval_run_prefix, task_type=task_type)
    elif args.command == 'visualize1':
        util.plot_task_res_dist({
            'auto_recover__NONE__MAB__6__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_pegr_s1': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h15__NONE__MAB__NO_E__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            # 'auto_recover__NONE__RANDOM__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC random', 'color': 'orange'},
            'auto_recover__APFVO__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},

            'auto_recover__h15__NONE__MAB__8__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__skipz_2_pegr_s0': {
                'name': 'TAMPC', 'color': 'green'},
        }, '{}_task_res.pkl'.format(peg_in_hole_real.DIR), task_type=peg_in_hole_real.DIR, figsize=(5, 7),
            set_y_label=False, max_t=300, expected_data_len=298, success_min_dist=0.02,
            task_names=task_names)

    elif args.command == 'visualize2':
        util.plot_task_res_dist({
            'auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h15__NONE__MAB__NO_E__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            'auto_recover__APFVO__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},

            'auto_recover__h15__NONE__MAB__8__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h15__NONE__MAB__NO_E__8__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC e=0', 'color': [0.8, 0.5, 0]},
            'auto_recover__APFVO__NONE__8__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-VO', 'color': 'black'},
            'auto_recover__APFSP__NONE__8__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__rex_extract_2_pegr_s1': {
                'name': 'APF-SP', 'color': [0.5, 0.5, 0.5]},
            'auto_recover__NONE__NONE__8__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__8__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive MPC++', 'color': 'red'},
        }, '{}_task_res.pkl'.format(peg_in_hole_real.DIR), task_type=peg_in_hole_real.DIR, figsize=(5, 7),
            set_y_label=False, max_t=500, expected_data_len=498, success_min_dist=0.02,
            task_names=task_names)

    else:
        use_tsf = UseTsf.SKIP
        rep_name = "pegr_s0"
        d, env, config, ds = PegRealGetter.free_space_env_init(0)
        ds.update_preprocessor(PegRealGetter.pre_invariant_preprocessor(use_tsf=use_tsf))
        xu, y, trial = ds.training_set(original=True)
        ds, pm = PegRealGetter.prior(env, use_tsf, rep_name=rep_name)
        yhat = pm.dyn_net.predict(xu, get_next_state=False, return_in_orig_space=True)
        u = xu[:, env.nx:]
        forces = y[:, 3:]
        f, axes = plt.subplots(4, 1, figsize=(10, 9))
        axes[0].scatter(u[:, 0].cpu(), yhat[:, 0].cpu(), color="red")
        axes[0].scatter(u[:, 0].cpu(), y[:, 0].cpu())
        axes[0].set_ylabel('dx')
        axes[0].set_xlabel('commanded dx')

        axes[1].scatter(u[:, 1].cpu(), yhat[:, 1].cpu(), color="red")
        axes[1].scatter(u[:, 1].cpu(), y[:, 1].cpu())
        axes[1].set_ylabel('dy')
        axes[1].set_xlabel('commanded dy')

        axes[2].scatter(u[:, 0].cpu(), forces[:, 0].cpu())
        axes[2].scatter(u[:, 0].cpu(), yhat[:, 3].cpu(), color="red")
        axes[2].set_ylabel('$dr_x$')
        axes[2].set_xlabel('commanded dx')

        axes[3].scatter(u[:, 1].cpu(), forces[:, 1].cpu())
        axes[3].scatter(u[:, 1].cpu(), yhat[:, 4].cpu(), color="red")
        axes[3].set_ylabel('$dr_y$')
        axes[3].set_xlabel('commanded dy')
        plt.show()
