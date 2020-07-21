import enum
import torch
import pickle
import re
import time
import math
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
from meta_contact.transform.block_push import LearnedTransform
from meta_contact.transform import invariant
from meta_contact.dynamics import online_model, model, prior, hybrid_model

from arm_pytorch_utilities.model import make

from meta_contact.dynamics.hybrid_model import OnlineAdapt
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

    if level in [3, 5]:
        init_peg = [0, -0.05]
        hole_pos = [0, 0.2]

    if level is 4:
        init_peg = [-0.15, 0.2]
        hole_pos = [0, 0.2]

    if level is 6:
        init_peg = [0.15, 0.07]

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
    elif use_tsf is UseTsf.SEP_DEC:
        return LearnedTransform.SeparateDecoder(ds, d, nz=5, nv=5, name="peg_s0")
    elif use_tsf is UseTsf.REX_EXTRACT:
        return LearnedTransform.RexExtract(ds, d, nz=5, nv=5, name="peg_s0")
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
        'trap_spread': 0.1,
        'trap_cost_annealing_rate': 0.9,
        'abs_unrecognized_threshold': 30,
        'adjust_model_pred_with_prev_error': False,
        'use_orientation_terminal_cost': False,
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


def get_pre_invariant_tsf_preprocessor(use_tsf):
    if use_tsf is UseTsf.COORD:
        return preprocess.PytorchTransformer(preprocess.NullSingleTransformer())
    else:
        return preprocess.PytorchTransformer(preprocess.NullSingleTransformer(),
                                             preprocess.RobustMinMaxScaler())


class Learn:
    @staticmethod
    def invariant(use_tsf=UseTsf.REX_EXTRACT, seed=1, name="", MAX_EPOCH=10, BATCH_SIZE=10, resume=False,
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
    def model(use_tsf, seed=1, name="", train_epochs=600, batch_N=500):
        d, env, config, ds = get_free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf)
        # tsf_name = "none_at_all"

        mw = PegNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                        name="peg_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


def run_controller(default_run_prefix, pre_run_setup, seed=1, level=1, recover_adjust=True, gating=None,
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
                                                       device=get_device(),
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': nominal_adapt},
                                                       local_model_kwargs=kwargs)

    # we're always going to be in the nominal mode in this case; might as well speed up testing
    if not use_demo and not reuse_escape_as_demonstration:
        gating = AlwaysSelectNominal()
    else:
        gating = hybrid_dynamics.get_gating() if gating is None else gating

    common_wrapper_opts, mpc_opts = get_controller_options(env)
    ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                        autonomous_recovery=autonomous_recovery,
                                        assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                                        reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                                        use_trap_cost=use_trap_cost,
                                        **common_wrapper_opts, **ctrl_opts,
                                        mpc_opts=mpc_opts)

    z = env.initGripperPos[2]
    goal = np.r_[env.hole, z, 0, 0]
    ctrl.set_goal(goal)
    # env._dd.draw_point('hole', env.hole, color=(0, 0.5, 0.8))

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

        run_name = default_run_prefix
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

            # ctrl.trap_set.append(
            #     (torch.tensor([env.hole[0] - 0.1, env.hole[1] - 0.2, z, 0, 0], device=ctrl.d, dtype=ctrl.dtype),
            #      torch.tensor([0, -1], device=ctrl.d, dtype=ctrl.dtype)))
            # test with explicit seeding on nominal trajectory
            # ctrl.mpc.U = torch.tensor([0, 0.5], device=ctrl.d, dtype=ctrl.dtype).repeat(ctrl.original_horizon, 1)

            ctrl.normalize_trapset_cost_to_state(x)

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
        ctrl.nominal_avg_velocity = 0.012

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
        ds_eval, _ = get_ds(env, rollout_file, validation_ratio=0.)
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
    def _closest_distance_to_goal(file, level, visualize=True, nodes_per_side=150):
        from sklearn.preprocessing import MinMaxScaler
        env = get_env(p.GUI if visualize else p.DIRECT, level=level)
        ds, _ = get_ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

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
        # rgb = [0, 0, 1]
        # start = reached_states[0, 0], reached_states[0, 1], z
        # for i in range(1, len(reached_states)):
        #     next = reached_states[i, 0], reached_states[i, 1], z
        #     p.addUserDebugLine(start, next, rgb)
        #     start = next

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
        if ok_nodes[goal_node[0]][goal_node[1]] is None:
            goal_node = (goal_node[0], goal_node[1] + 1)
        visited, path = EvaluateTask.dijsktra(g, goal_node)
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

    @staticmethod
    def closest_distance_to_goal_whole_set(prefix, suffix=".mat", **kwargs):
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
                  filename.startswith(prefix) and filename.endswith(suffix)]
        dists = []
        for i, trial in enumerate(trials):
            d = EvaluateTask._closest_distance_to_goal("peg/{}".format(trial), visualize=i == 0, level=level,
                                                       **kwargs)
            dists.append(min([dd for dd in d if dd is not None]))
            runs[prefix][trial] = d

        logger.info(dists)
        logger.info("mean {:.2f} std {:.2f} cm".format(np.mean(dists) * 10, np.std(dists) * 10))
        with open(fullname, 'wb') as f:
            pickle.dump(runs, f)
            logger.info("saved runs to %s", fullname)
        time.sleep(0.5)


class Visualize:
    @staticmethod
    def task_res_dist(series_to_plot, res_file, plot_cumulative_distribution=True, max_t=500,
                      expected_data_len=498,
                      plot_min_distribution=False):
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

        fullname = os.path.join(cfg.DATA_DIR, res_file)
        if os.path.exists(fullname):
            with open(fullname, 'rb') as f:
                runs = pickle.load(f)
                logger.info("loaded runs from %s", fullname)
        else:
            raise RuntimeError("missing cached task results file {}".format(fullname))

        tasks = {}
        for prefix, dists in runs.items():
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
            series = []

            for series_name, dists in res_list.items():
                tokens = name_to_tokens(series_name)
                if series_name in series_to_plot:
                    # remove any non-list elements (historical)
                    dists = [dlist for dlist in dists if type(dlist) is list]
                    # process the dists so they are all valid (replace nones)
                    for dhistory in dists:
                        min_dist_up_to_now = 100
                        for i, d in enumerate(dhistory):
                            if d is None:
                                dhistory[i] = min_dist_up_to_now
                            else:
                                min_dist_up_to_now = min(min_dist_up_to_now, d)
                                dhistory[i] = min(min_dist_up_to_now, d)

                        # if list is shorter than expected that means it finished so should have 0 dist
                        dhistory.extend([0] * (expected_data_len - len(dhistory)))
                        min_dist = min(min(dhistory), min_dist)
                        max_dist = max(max(dhistory), max_dist)

                    series.append((series_name, tokens, np.stack(dists)))

            if plot_min_distribution:
                f, ax = plt.subplots(len(series), 1, figsize=(8, 9))
                f.suptitle("task {}".format(level))

                for i, data in enumerate(series):
                    series_name, tk, dists = data
                    dists = np.min(dists, axis=1)
                    logger.info("%s with %d runs mean {:.2f} ({:.2f})".format(np.mean(dists) * 10, np.std(dists) * 10),
                                series_name, len(dists))
                    sns.distplot(dists, ax=ax[i], hist=True, kde=False, bins=np.linspace(min_dist, max_dist, 20))
                    ax[i].set_title((tk['adaptation'], tk['recovery'], tk['optimism']))
                    ax[i].set_xlim(min_dist, max_dist)
                    ax[i].set_ylim(0, int(0.6 * len(dists)))
                ax[-1].set_xlabel('closest dist to goal [m]')
                f.tight_layout(rect=[0, 0.03, 1, 0.95])
            if plot_cumulative_distribution:
                f, ax = plt.subplots(1, figsize=(8, 9))
                f.suptitle("task {}".format(level))

                for i, data in enumerate(series):
                    series_name, tk, dists = data
                    plot_info = series_to_plot[series_name]

                    t = np.arange(dists.shape[1])
                    m = np.median(dists, axis=0)
                    lower = np.percentile(dists, 10, axis=0)
                    upper = np.percentile(dists, 90, axis=0)

                    c = plot_info['color']
                    ax.plot(t, m, color=c, label=plot_info['name'])
                    plt.fill_between(t, lower, upper, facecolor=c, alpha=0.3)

                ax.legend()
                ax.set_xlim(0, max_t)
                ax.set_ylim(0, max_dist * 1.05)
                ax.set_ylabel('closest dist to goal')
                ax.set_xlabel('control step')
                f.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()


if __name__ == "__main__":
    level = 0
    ut = UseTsf.COORD

    # OfflineDataCollection.freespace(trials=200, trial_length=50, mode=p.DIRECT)

    # for seed in range(1):
    #     Learn.invariant(ut, seed=seed, name="peg", MAX_EPOCH=6000, BATCH_SIZE=500)
    # for seed in range(1):
    #     Learn.model(ut, seed=seed, name="")

    # EvaluateTask.closest_distance_to_goal_whole_set('auto_recover__GP_KERNEL_INDEP_OUT__NONE__3__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST', suffix='500.mat')
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__GP_KERNEL_INDEP_OUT__NONE__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST',
    #     suffix='500.mat', nodes_per_side=150)
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST',
    #     suffix='500.mat')

    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__RETURN_STATE__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__RETURN_STATE__5__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__RETURN_STATE__6__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')

    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__MAB__3__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__MAB__5__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')
    # EvaluateTask.closest_distance_to_goal_whole_set(
    #     'auto_recover__NONE__MAB__6__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST',
    #     suffix='500.mat')

    # Visualize.task_res_dist({
    #     'auto_recover__NONE__MAB__6__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
    #         'name': 'MAB', 'color': 'green'},
    #     'auto_recover__NONE__RETURN_STATE__6__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
    #         'name': 'return state', 'color': 'blue'},
    #     'auto_recover__GP_KERNEL_INDEP_OUT__NONE__6__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__NOTRAPCOST': {
    #         'name': 'adaptive baseline++', 'color': 'red'},
    # }, 'peg_task_res.pkl')

    for seed in range(1):
        tune_trap_set_cost(seed=seed, level=0, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
                           use_trap_cost=True,
                           autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)

    # tune_recovery_policy(seed=0, level=0, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
    #                      autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)

    # evaluate_after_rollout(
    #     'peg/auto_recover__NONE__RETURN_STATE__5__COORD__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST__3__200.mat',
    #     184, seed=3, level=5, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
    #     use_trap_cost=True,
    #     autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)

    # for level in [3, 5, 6]:
    #     for seed in range(0, 10):
    #         test_autonomous_recovery(seed=seed, level=level, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
    #                                  reuse_escape_as_demonstration=False, use_trap_cost=True,
    #                                  assume_all_nonnominal_dynamics_are_traps=False, num_frames=500,
    #                                  autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE)
    #
    # for level in [3, 5, 6]:
    #     for seed in range(0, 10):
    #         test_autonomous_recovery(seed=seed, level=level, use_tsf=ut, nominal_adapt=OnlineAdapt.NONE,
    #                                  reuse_escape_as_demonstration=False, use_trap_cost=True,
    #                                  assume_all_nonnominal_dynamics_are_traps=False, num_frames=500,
    #                                  autonomous_recovery=online_controller.AutonomousRecovery.MAB)
    #
    # # baseline ++
    # for level in [3, 5, 6]:
    #     for seed in range(10):
    #         test_autonomous_recovery(seed=seed, level=level, use_tsf=UseTsf.NO_TRANSFORM,
    #                                  nominal_adapt=OnlineAdapt.GP_KERNEL_INDEP_OUT,
    #                                  gating=AlwaysSelectNominal(),
    #                                  num_frames=500,
    #                                  reuse_escape_as_demonstration=False, use_trap_cost=False,
    #                                  assume_all_nonnominal_dynamics_are_traps=False,
    #                                  autonomous_recovery=online_controller.AutonomousRecovery.NONE)
    #
    # # baseline non-adaptive
    # for level in [3, 5, 6]:
    #     for seed in range(10):
    #         test_autonomous_recovery(seed=seed, level=level, use_tsf=UseTsf.NO_TRANSFORM,
    #                                  nominal_adapt=OnlineAdapt.NONE,
    #                                  gating=AlwaysSelectNominal(),
    #                                  num_frames=500,
    #                                  reuse_escape_as_demonstration=False, use_trap_cost=False,
    #                                  assume_all_nonnominal_dynamics_are_traps=False,
    #                                  autonomous_recovery=online_controller.AutonomousRecovery.NONE)
