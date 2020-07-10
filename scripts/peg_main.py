import enum
import math
import torch
import pickle
import re
import time
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import seaborn as sns

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from meta_contact import cfg
from meta_contact.env import peg_in_hole
from meta_contact.controller import controller
from meta_contact.transform.peg_in_hole import CoordTransform, translation_generator
from meta_contact.transform import invariant
from meta_contact.dynamics import online_model, model, prior, hybrid_model

from arm_pytorch_utilities.model import make
from meta_contact.controller.online_controller import NominalTrajFrom

from meta_contact.dynamics.hybrid_model import OnlineAdapt, get_gating
from meta_contact.controller import online_controller
from meta_contact.controller.gating_function import AlwaysSelectNominal

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


def get_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_peg = [-0.2, 0]
    hole_pos = [0.3, 0.3]

    if level is 2:
        init_peg = [0, -0.2]
        hole_pos = [0, 0.2]

    if level is 3:
        init_peg = [0, -0.05]
        hole_pos = [0, 0.2]

    if level is 4:
        init_peg = [-0.15, 0.2]
        hole_pos = [0, 0.2]

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
    env_dir = 'peg/floating'
    return env


def get_ds(env, data_dir, **kwargs):
    d = get_device()
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = peg_in_hole.PegInHoleDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
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


def get_transform(env, ds, use_tsf):
    # add in invariant transform here
    d = get_device()
    if use_tsf is UseTsf.NO_TRANSFORM:
        return None
    elif use_tsf is UseTsf.COORD:
        return CoordTransform.factory(env, ds)
    else:
        raise RuntimeError("Unrecgonized transform {}".format(use_tsf))


def update_ds_with_transform(env, ds, use_tsf, evaluate_transform=True):
    invariant_tsf = get_transform(env, ds, use_tsf)

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

        # wrap the transform as a data preprocessor
        preprocessor = preprocess.Compose(
            [invariant.InvariantTransformer(invariant_tsf),
             preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler())])
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
                        name="peg_{}_{}".format(tsf_name, seed))

        train_epochs = 500
        pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(),
                                     train_epochs=train_epochs)
    elif prior_class is prior.NoPrior:
        pm = prior.NoPrior()
    else:
        pm = prior_class.from_data(ds)
    return pm


def get_prior(env, use_tsf=UseTsf.COORD, prior_class=prior.NNPrior):
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)
    untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf, evaluate_transform=False)
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
        'num_samples': 500,
        'noise_sigma': torch.diag(sigma),
        'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
        'lambda_': 1e-2,
        'horizon': 20,
        'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
        'sample_null_action': False,
        'step_dependent_dynamics': True,
        'rollout_samples': 10,
        'rollout_var_cost': 0,
    }
    return common_wrapper_opts, mpc_opts


class OfflineDataCollection:
    @staticmethod
    def random_config(env):
        hole = (np.random.random((2,)) - 0.5)
        init_peg = (np.random.random((2,)) - 0.5)
        return hole, init_peg

    @staticmethod
    def freespace(trials=200, trial_length=50, mode=p.DIRECT):
        env = get_env(mode, 0)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(env_dir, level)
        sim = peg_in_hole.PegInHole(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                    stop_when_done=False, save_dir=save_dir)
        rand.seed(4)
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


class Learn:
    @staticmethod
    def model(use_tsf, seed=1, name="", train_epochs=600, batch_N=500):
        d, env, config, ds = get_free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf)
        # tsf_name = "none_at_all"

        mw = PegNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                        name="peg_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


def test_autonomous_recovery(seed=1, level=1, recover_adjust=True, gating=None,
                             use_tsf=UseTsf.COORD, nominal_adapt=OnlineAdapt.NONE,
                             autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE,
                             use_demo=False,
                             use_trap_cost=True,
                             reuse_escape_as_demonstration=False, num_frames=200, run_name=None,
                             assume_all_nonnominal_dynamics_are_traps=False,
                             ctrl_opts=None,
                             **kwargs):
    if ctrl_opts is None:
        ctrl_opts = {}

    env = get_env(p.GUI, level=level, log_video=True)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = get_prior(env, use_tsf)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local, config = get_ds(env, demo, validation_ratio=0.)
        ds_local.update_preprocessor(ds.preprocessor)
        dss.append(ds_local)

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, [use_tsf.name],
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': nominal_adapt},
                                                       local_model_kwargs=kwargs)

    # we're always going to be in the nominal mode in this case; might as well speed up testing
    if not use_demo and not reuse_escape_as_demonstration:
        gating = AlwaysSelectNominal()
    else:
        gating = hybrid_dynamics.get_gating() if gating is None else gating

    common_wrapper_opts, mpc_opts = get_controller_options(env)
    ctrl_opts.update({'trap_cost_per_dim': 30.})
    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                        abs_unrecognized_threshold=30,
                                        autonomous_recovery=autonomous_recovery,
                                        assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                        reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                        use_trap_cost=use_trap_cost,
                                        **common_wrapper_opts, **ctrl_opts,
                                        mpc_opts=mpc_opts)

    # TODO set sequence of goals
    z = env.initGripperPos[2]
    goal = np.r_[env.hole, z, 0, 0]
    ctrl.set_goal(goal)
    # env._dd.draw_point('hole', env.hole, color=(0, 0.5, 0.8))
    ctrl.create_recovery_traj_seeder(dss,
                                     nom_traj_from=NominalTrajFrom.RECOVERY_ACTIONS if recover_adjust else NominalTrajFrom.NO_ADJUSTMENT)

    env.draw_user_text(gating.name, 13, left_offset=-1.5)
    env.draw_user_text("run seed {}".format(seed), 12, left_offset=-1.5)
    env.draw_user_text("recovery {}".format(autonomous_recovery.name), 11, left_offset=-1.6)
    if reuse_escape_as_demonstration:
        env.draw_user_text("reuse escape", 10, left_offset=-1.6)
    if use_trap_cost:
        env.draw_user_text("trap set cost".format(autonomous_recovery.name), 9, left_offset=-1.6)

    sim = peg_in_hole.PegInHole(env, ctrl, num_frames=num_frames, plot=False, save=True, stop_when_done=True)
    seed = rand.seed(seed)

    if run_name is None:
        def affix_run_name(*args):
            nonlocal run_name
            for token in args:
                run_name += "__{}".format(token)

        run_name = 'auto_recover'
        affix_run_name(nominal_adapt.name)
        affix_run_name(autonomous_recovery.name + ("_WITHDEMO" if use_demo else ""))
        affix_run_name(level)
        affix_run_name(use_tsf.name)
        affix_run_name("ALLTRAP" if assume_all_nonnominal_dynamics_are_traps else "SOMETRAP")
        affix_run_name("REUSE" if reuse_escape_as_demonstration else "NOREUSE")
        affix_run_name(gating.name)
        affix_run_name("TRAPCOST" if use_trap_cost else "NOTRAPCOST")
        affix_run_name(seed)
        affix_run_name(num_frames)

    env.draw_user_text(run_name, 14, left_offset=-1.5)
    sim.run(seed, run_name)
    logger.info("last run cost %f", np.sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


class EvaluateTask:
    class Graph:
        def __init__(self):
            from collections import defaultdict
            self.nodes = set()
            self.edges = defaultdict(list)
            self.distances = {}

        def add_node(self, value):
            self.nodes.add(value)

        def add_edge(self, from_node, to_node, distance):
            self.edges[from_node].append(to_node)
            self.distances[(from_node, to_node)] = distance

    @staticmethod
    def dijsktra(graph, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(graph.nodes)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in graph.edges[min_node]:
                weight = current_weight + graph.distances[(min_node, edge)]
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node

        return visited, path

    @staticmethod
    def _closest_distance_to_goal(file, level, visualize=True, nodes_per_side=100):
        from sklearn.preprocessing import MinMaxScaler
        env = get_env(p.GUI if visualize else p.DIRECT, level=level)
        ds, _ = get_ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        if level is 1:
            min_pos = [-0.3, -0.3]
            max_pos = [0.5, 0.5]
        elif level is 3:
            min_pos = [-0.3, -0.1]
            max_pos = [0.3, 0.5]
        else:
            raise RuntimeError("Unspecified range for level {}".format(level))

        scaler = MinMaxScaler(feature_range=(0, nodes_per_side - 1))
        scaler.fit(np.array([min_pos, max_pos]))

        reached_states = X[:, :2].cpu().numpy()
        goal_pos = env.hole[:2]

        lower_bound_dist = np.linalg.norm((reached_states - goal_pos), axis=1).min()

        # we expect there not to be walls between us if the minimum distance is this small
        # if lower_bound_dist < 0.2:
        #     return lower_bound_dist

        def node_to_pos(node):
            return scaler.inverse_transform([node])[0]
            # return float(node[0]) / nodes_per_side + min_pos[0], float(node[1]) / nodes_per_side + min_pos[1]

        def pos_to_node(pos):
            pair = scaler.transform([pos])[0]
            node = tuple(int(round(v)) for v in pair)
            return node
            # return int(round((pos[0] - min_pos[0]) * nodes_per_side)), int(
            #     round((pos[1] - min_pos[1]) * nodes_per_side))

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
        g = EvaluateTask.Graph()
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
        visited, path = EvaluateTask.dijsktra(g, goal_node)
        # find min across visited states
        min_dist = 100
        min_node = None
        for xy in reached_states:
            n = pos_to_node(xy)
            if n in visited and visited[n] < min_dist:
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
        return min_dist

    @staticmethod
    def closest_distance_to_goal_whole_set(prefix, **kwargs):
        m = re.search(r"\d+", prefix)
        if m is not None:
            level = int(m.group())
        else:
            raise RuntimeError("Prefix has no level information in it")

        fullname = os.path.join(cfg.DATA_DIR, 'peg_task_res.pkl')
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                runs = pickle.load(f)
                logger.info("loaded runs from %s", fullname)
        else:
            runs = {}

        if prefix not in runs:
            runs[prefix] = {}

        trials = [filename for filename in os.listdir(os.path.join(cfg.DATA_DIR, "peg")) if
                  filename.startswith(prefix)]
        dists = []
        for i, trial in enumerate(trials):
            d = EvaluateTask._closest_distance_to_goal("peg/{}".format(trial), visualize=i == 0, level=level,
                                                       **kwargs)
            dists.append(d)
            runs[prefix][trial] = d

        logger.info(dists)
        logger.info("mean {:.2f} std {:.2f} cm".format(np.mean(dists) * 10, np.std(dists) * 10))
        with open(fullname, 'wb') as f:
            pickle.dump(runs, f)
            logger.info("saved runs to %s", fullname)
        time.sleep(0.5)


class Visualize:
    @staticmethod
    def task_res_dist(filter_function=None):
        def name_to_tokens(name):
            tk = {'name': name}
            tokens = name.split('__')
            # legacy fallback
            if len(tokens) < 5:
                tokens = name.split('_')
                # skip prefix
                tokens = tokens[2:]
                if tokens[0] == "NONE":
                    tk['adaptation'] = tokens.pop(0)
                else:
                    tk['adaptation'] = "{}_{}".format(tokens[0], tokens[1])
                    tokens = tokens[2:]
                if tokens[0] in ("RANDOM", "NONE"):
                    tk['recovery'] = tokens.pop(0)
                else:
                    tk['recovery'] = "{}_{}".format(tokens[0], tokens[1])
                    tokens = tokens[2:]
                tk['level'] = int(tokens.pop(0))
                tk['tsf'] = tokens.pop(0)
                tk['reuse'] = tokens.pop(0)
                tk['optimism'] = "ALLTRAP"
                tk['trap_use'] = "NOTRAPCOST"
            else:
                tokens.pop(0)
                tk['adaptation'] = tokens[0]
                tk['recovery'] = tokens[1]
                tk['level'] = int(tokens[2])
                tk['tsf'] = tokens[3]
                tk['optimism'] = tokens[4]
                tk['reuse'] = tokens[5]
                if len(tokens) > 7:
                    tk['trap_use'] = tokens[7]
                else:
                    tk['trap_use'] = "NOTRAPCOST"

            return tk

        fullname = os.path.join(cfg.DATA_DIR, 'peg_task_res.pkl')
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                runs = pickle.load(f)
                logger.info("loaded runs from %s", fullname)
        else:
            raise RuntimeError("missing cached task results file {}".format(fullname))

        tasks = {}
        for prefix, dists in sorted(runs.items()):
            m = re.search(r"\d+", prefix)
            if m is not None:
                level = int(m.group())
            else:
                raise RuntimeError("Prefix has no level information in it")
            if level not in tasks:
                tasks[level] = {}
            if prefix not in tasks[level]:
                tasks[level][prefix] = dists

        for level, res in tasks.items():
            min_dist = 100
            max_dist = 0

            res_list = {k: list(v.values()) for k, v in res.items()}
            for dists in res_list.values():
                min_dist = min(min(dists), min_dist)
                max_dist = max(max(dists), max_dist)

            series = []
            for i, (series_name, dists) in enumerate(res_list.items()):
                tokens = name_to_tokens(series_name)
                if filter_function is None or filter_function(tokens):
                    series.append((series_name, tokens, dists))

            f, ax = plt.subplots(len(series), 1, figsize=(8, 9))
            f.suptitle("task {}".format(level))

            for i, data in enumerate(series):
                series_name, tk, dists = data
                logger.info("%s with %d runs mean {:.2f} ({:.2f})".format(np.mean(dists) * 10, np.std(dists) * 10),
                            series_name, len(dists))
                sns.distplot(dists, ax=ax[i], hist=True, kde=False, bins=np.linspace(min_dist, max_dist, 20))
                ax[i].set_title((tk['adaptation'], tk['recovery'], tk['optimism']))
                ax[i].set_xlim(min_dist, max_dist)
                ax[i].set_ylim(0, int(0.6 * len(dists)))
            ax[-1].set_xlabel('closest dist to goal [m]')
            f.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()


if __name__ == "__main__":
    level = 0
    ut = UseTsf.COORD

    # OfflineDataCollection.freespace(trials=200, trial_length=50, mode=p.DIRECT)

    # for seed in range(1):
    #     Learn.model(ut, seed=seed, name="")

    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__NONE__RETURN_STATE__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST')
    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__NONE__MAB__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST')
    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__GP_KERNEL__NONE__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST')
    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__NONE__RANDOM__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST')
    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__NONE__RANDOM__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST')
    #
    # Visualize.task_res_dist()

    for seed in range(0, 5):
        test_autonomous_recovery(seed=seed, level=3, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
                                 reuse_escape_as_demonstration=False, use_trap_cost=True,
                                 assume_all_nonnominal_dynamics_are_traps=False,
                                 autonomous_recovery=online_controller.AutonomousRecovery.MAB)

    # for seed in range(0, 5):
    #     test_autonomous_recovery(seed=seed, level=3, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
    #                              reuse_escape_as_demonstration=False, use_trap_cost=False,
    #                              assume_all_nonnominal_dynamics_are_traps=False,
    #                              autonomous_recovery=online_controller.AutonomousRecovery.RANDOM)
    #
    for seed in range(0, 5):
        test_autonomous_recovery(seed=seed, level=3, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
                                 reuse_escape_as_demonstration=False, use_trap_cost=True,
                                 assume_all_nonnominal_dynamics_are_traps=False,
                                 autonomous_recovery=online_controller.AutonomousRecovery.RANDOM)

    # for seed in range(0, 5):
    #     test_autonomous_recovery(seed=seed, level=3, use_tsf=ut, nominal_adapt=OnlineAdapt.GP_KERNEL,
    #                              reuse_escape_as_demonstration=False, use_trap_cost=False,
    #                              assume_all_nonnominal_dynamics_are_traps=False,
    #                              autonomous_recovery=online_controller.AutonomousRecovery.NONE)
