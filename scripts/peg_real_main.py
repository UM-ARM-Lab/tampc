import enum
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

try:
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
from tampc.env import peg_in_hole_real
from tampc.controller import controller
from tampc.transform.peg_in_hole import CoordTransform, translation_generator
from tampc.transform.block_push import LearnedTransform
from tampc.transform import invariant
from tampc.dynamics import online_model, model, prior, hybrid_model

from arm_pytorch_utilities.model import make

from tampc.dynamics.hybrid_model import OnlineAdapt
from tampc.controller import online_controller
from tampc.controller.gating_function import AlwaysSelectNominal
from tampc import util

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

env_dir = None


# --- SHARED GETTERS
def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_env(level=0, **kwargs):
    global env_dir
    env = peg_in_hole_real.RealPegEnv(environment_level=level, **kwargs)
    if level is task_map['Peg-T']:
        x = 1.74962708 - 0.001
        y = -0.02913485 + 0.011
        # env.set_task_config(hole=[x, y], init_peg=[1.64363362, 0.05320179])
        # for tuning close to goal behaviour (spiral exploration vs going straight to goal)
        env.set_task_config(hole=[x, y], init_peg=[x + 0.03, y])
    env_dir = '{}/real'.format(peg_in_hole_real.DIR)
    return env


def get_ds(env, data_dir, **kwargs):
    d = get_device()
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = peg_in_hole_real.PegRealDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
    return ds, config


def get_free_space_env_init(seed=1, **kwargs):
    d = get_device()
    env = get_env(kwargs.pop('mode', p.DIRECT), **kwargs)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))
    return d, env, config, ds


class UseTsf(enum.Enum):
    NO_TRANSFORM = 0
    COORD = 1
    SEP_DEC = 12
    EXTRACT = 13
    REX_EXTRACT = 14
    SKIP = 15
    REX_SKIP = 16


def get_transform(env, ds, use_tsf, override_name=None):
    # add in invariant transform here
    d = get_device()
    if use_tsf is UseTsf.NO_TRANSFORM:
        return None
    elif use_tsf is UseTsf.COORD:
        return CoordTransform.factory(env, ds)
    elif use_tsf is UseTsf.SEP_DEC:
        return LearnedTransform.SeparateDecoder(ds, d, nz=5, nv=5, name=override_name or "peg_s0")
    elif use_tsf is UseTsf.REX_EXTRACT:
        return LearnedTransform.RexExtract(ds, d, nz=5, nv=5, name=override_name or "peg_s0")
    elif use_tsf is UseTsf.SKIP:
        return LearnedTransform.SkipLatentInput(ds, d, name=override_name or "peg_s0")
    elif use_tsf is UseTsf.REX_SKIP:
        return LearnedTransform.RexSkip(ds, d, name=override_name or "peg_s0")
    else:
        raise RuntimeError("Unrecgonized transform {}".format(use_tsf))


# TODO move these shared script methods into shared module
def update_ds_with_transform(env, ds, use_tsf, evaluate_transform=True, rep_name=None):
    invariant_tsf = get_transform(env, ds, use_tsf, override_name=rep_name)

    if invariant_tsf:
        # load transform (only 1 function for learning transform reduces potential for different learning params)
        if use_tsf is not UseTsf.COORD and not invariant_tsf.load(invariant_tsf.get_last_checkpoint()):
            raise RuntimeError("Transform {} should be learned before using".format(invariant_tsf.name))

        if evaluate_transform:
            losses = invariant_tsf.evaluate_validation(None)
            logger.info("tsf on validation %s",
                        "  ".join(
                            ["{} {:.5f}".format(name, loss.mean().cpu().item()) if loss is not None else "" for
                             name, loss
                             in zip(invariant_tsf.loss_names(), losses)]))

        components = [get_pre_invariant_tsf_preprocessor(use_tsf), invariant.InvariantTransformer(invariant_tsf)]
        if use_tsf not in [UseTsf.SKIP, UseTsf.REX_SKIP]:
            components.append(preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler()))
        preprocessor = preprocess.Compose(components)
    else:
        preprocessor = no_tsf_preprocessor()
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)
    return untransformed_config, use_tsf.name, preprocessor


def no_tsf_preprocessor():
    return preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler())


def get_loaded_prior(prior_class, ds, tsf_name, relearn_dynamics, seed=0):
    d = get_device()
    if prior_class is prior.NNPrior:
        mw = PegNetwork(model.DeterministicUser(make.make_sequential_network(ds.config).to(device=d)), ds,
                        name="pegr_{}_{}".format(tsf_name, seed))

        train_epochs = 500
        pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(
            sort_by_time=False), train_epochs=train_epochs)
    elif prior_class is prior.PassthroughLatentDynamicsPrior:
        pm = prior.PassthroughLatentDynamicsPrior(ds)
    elif prior_class is prior.NoPrior:
        pm = prior.NoPrior()
    else:
        pm = prior_class.from_data(ds)
    return pm


def get_prior(env, use_tsf=UseTsf.COORD, prior_class=prior.NNPrior, rep_name=None):
    if use_tsf in [UseTsf.SKIP, UseTsf.REX_SKIP]:
        prior_class = prior.PassthroughLatentDynamicsPrior
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)
    untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False,
                                                                            rep_name=rep_name)
    pm = get_loaded_prior(prior_class, ds, tsf_name, False)
    return ds, pm


def get_controller_options(env):
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
        'abs_unrecognized_threshold': 15,
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


class OfflineDataCollection:
    @staticmethod
    def freespace(seed_offset=0, trials=200, trial_length=50, force_gui=False):
        env = get_env(level=0, stub=False)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(env_dir, 0)
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
            with peg_in_hole_real.VideoLogger():
                sim.run(seed, run_name=run_name)

        env.close()
        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()


class PegNetwork(model.NetworkModelWrapper):
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


def get_pre_invariant_tsf_preprocessor(use_tsf):
    if use_tsf is UseTsf.COORD:
        return preprocess.PytorchTransformer(preprocess.NullSingleTransformer())
    else:
        # normalize position and force dimensions separately using shared scales
        return preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler(dims_share_scale=[[0, 1], [3, 4]]),
                                             preprocess.RobustMinMaxScaler(dims_share_scale=[[0, 1], [3, 4]]))


class Learn:
    @staticmethod
    def invariant(use_tsf=UseTsf.REX_EXTRACT, seed=1, name="", MAX_EPOCH=1000, BATCH_SIZE=500, resume=False,
                  **kwargs):
        d, env, config, ds = get_free_space_env_init(seed)

        ds.update_preprocessor(get_pre_invariant_tsf_preprocessor(use_tsf))
        invariant_cls = get_transform(env, ds, use_tsf).__class__
        common_opts = {'name': "{}_s{}".format(name, seed)}
        invariant_tsf = invariant_cls(ds, d, **common_opts, **kwargs)
        if resume:
            invariant_tsf.load(invariant_tsf.get_last_checkpoint())
        invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)

    @staticmethod
    def model(use_tsf, seed=1, name="", train_epochs=500, batch_N=500, rep_name=None):
        d, env, config, ds = get_free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf, rep_name=rep_name)
        # tsf_name = "none_at_all"

        mw = PegNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                        name="pegr_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


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
    env = get_env(level=level, stub=False)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = get_prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local, config = get_ds(env, demo, validation_ratio=0.)
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

    tampc_opts, mpc_opts = get_controller_options(env)
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

    z = 0.98
    goal = np.r_[env.hole, z, 0, 0]
    ctrl.set_goal(goal)

    sim = peg_in_hole_real.ExperimentRunner(env, ctrl, num_frames=num_frames, plot=False, save=True,
                                            stop_when_done=True)
    seed = rand.seed(seed)
    sim.dd.draw_text("seed", "run seed {}".format(seed), 12, left_offset=-1.5)
    sim.dd.draw_text("recovery method", "recovery {}".format(autonomous_recovery.name), 11, left_offset=-1.6)
    if reuse_escape_as_demonstration:
        sim.dd.draw_text("resuse", "reuse escape", 10, left_offset=-1.6)

    if run_name is None:
        def affix_run_name(*args):
            nonlocal run_name
            for token in args:
                run_name += "__{}".format(token)

        def get_rep_model_name(ds):
            import re
            tsf = ds.preprocessor.tsf.transforms[-1]
            tsf_name = tsf.tsf.name
            tsf_name = re.match(r".*?s\d+", tsf_name)[0]
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

    with peg_in_hole_real.VideoLogger():
        sim.dd.draw_text("run name", run_name, 14, left_offset=-0.8)
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
    def closest_distance_to_goal(file, level, visualize=True, nodes_per_side=150):
        from sklearn.preprocessing import MinMaxScaler
        env = get_env(p.GUI if visualize else p.DIRECT, level=level)
        ds, _ = get_ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        # TODO consider making a simulation of the environment to allow visualization and validation
        if level is 1:
            min_pos = [-0.3, -0.3]
            max_pos = [0.5, 0.5]
        elif level is 3:
            min_pos = [-0.2, -0.1]
            max_pos = [0.2, 0.35]
        elif level is 5:
            min_pos = [-0.4, -0.1]
            max_pos = [0.4, 0.4]
        elif level is 6:
            min_pos = [-0.3, -0.1]
            max_pos = [0.3, 0.3]
        elif level is 7:
            translation = 10
            min_pos = [-0.3 + translation, -0.1 + translation]
            max_pos = [0.3 + translation, 0.3 + translation]
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
        return dists


task_map = {'freespace': 0, 'Peg-U': 3, 'Peg-I': 5, 'Peg-T': 6, 'Peg-T(T)': 7}

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
    task_names = {v: k for k, v in task_map.items()}
    tampc_params = {}
    for d in args.tampc_param:
        tampc_params.update(d)
    mpc_params = {}
    for d in args.mpc_param:
        mpc_params.update(d)

    if args.command == 'collect':
        OfflineDataCollection.freespace(seed_offset=15, trials=5, trial_length=30, force_gui=args.gui)
    elif args.command == 'learn_representation':
        for seed in args.seed:
            Learn.invariant(ut, seed=seed, name="pegr", MAX_EPOCH=1000, BATCH_SIZE=args.batch)
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
                                     visualize_rollout=args.visualize_rollout, run_prefix=args.run_prefix,
                                     override_tampc_params=tampc_params, override_mpc_params=mpc_params,
                                     autonomous_recovery=autonomous_recovery)
    elif args.command == 'evaluate':
        util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
                                                args.eval_run_prefix, suffix="{}.mat".format(args.num_frames),
                                                task_type='peg')
    elif args.command == 'visualize1':
        util.plot_task_res_dist({
            'auto_recover__NONE__MAB__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h15_larger_min_window__NONE__MAB__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC tuned', 'color': 'blue', 'label': True},
            'auto_recover__NONE__RANDOM__3__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': 'orange'},
            'auto_recover__NONE__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive baseline++', 'color': 'red'},
            'sac__3': {'name': 'SAC', 'color': 'cyan'},
            'sac__9': {'name': 'SAC', 'color': 'cyan'},

            'auto_recover__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__h20_less_anneal__NONE__MAB__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC tuned', 'color': 'blue', 'label': True},
            'auto_recover__NONE__RANDOM__5__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC random', 'color': 'orange'},
            'auto_recover__NONE__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'non-adapative', 'color': 'purple'},
            'auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
                'name': 'adaptive baseline++', 'color': 'red'},
            'sac__5': {'name': 'SAC', 'color': 'cyan'},
        }, 'peg_task_res.pkl', task_type='peg', figsize=(5, 7), set_y_label=False,
            task_names=task_names)

    elif args.command == 'visualize2':
        util.plot_task_res_dist({
            'auto_recover__NONE__MAB__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
            # 'auto_recover__NONE__RANDOM__6__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC random', 'color': 'orange'},
            'auto_recover__NONE__MAB__6__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal': {
                'name': 'TAMPC skip z', 'color': 'black'},
            # 'auto_recover__NONE__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'non-adapative', 'color': 'purple'},
            # 'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'adaptive baseline++', 'color': 'red'},
            # 'sac__6': {'name': 'SAC', 'color': 'cyan'},

            'auto_recover__NONE__MAB__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'TAMPC', 'color': 'green'},
            'auto_recover__NONE__MAB__7__SKIP__SOMETRAP__NOREUSE__AlwaysSelectNominal': {
                'name': 'TAMPC skip z', 'color': 'black'},
            # 'auto_recover__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC original space', 'color': 'olive', 'label': True},
            # 'auto_recover__NONE__RANDOM__7__REX_EXTRACT__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'TAMPC random', 'color': 'orange'},
            # 'auto_recover__NONE__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'non-adapative', 'color': 'purple'},
            # 'auto_recover__GP_KERNEL_INDEP_OUT__NONE__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
            #     'name': 'adaptive baseline++', 'color': 'red'},
            # 'sac__7': {'name': 'SAC', 'color': 'cyan'},
        }, 'peg_task_res.pkl', task_type='peg', figsize=(5, 7), set_y_label=False,
            task_names=task_names)

    else:
        use_tsf = UseTsf.SKIP
        rep_name = "pegr_s1"
        d, env, config, ds = get_free_space_env_init(0)
        ds.update_preprocessor(get_pre_invariant_tsf_preprocessor(use_tsf=use_tsf))
        xu, y, trial = ds.training_set(original=True)
        ds, pm = get_prior(env, use_tsf, rep_name=rep_name)
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
