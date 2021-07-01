try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import time
import random
import torch
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
import pprint

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import draw

from tampc import cfg
from cottun import contact
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import hybrid_model
from tampc.env import arm
from tampc.env.arm import task_map, Levels

from tampc.dynamics.hybrid_model import OnlineAdapt
from tampc.controller import online_controller
from tampc.controller.gating_function import AlwaysSelectNominal
from tampc import util
from tampc.util import no_tsf_preprocessor, UseTsf, Baseline
from tampc.env_getters.arm import ArmGetter

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


class OfflineDataCollection:
    @staticmethod
    def freespace(seed_offset=0, trials=200, trial_length=50, force_gui=False):
        env = ArmGetter.env(level=0, mode=p.GUI if force_gui else p.DIRECT)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(ArmGetter.env_dir, 0)
        sim = arm.ExperimentRunner(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                   stop_when_done=False, save_dir=save_dir)
        # randomly distribute data
        for offset in range(trials):
            seed = rand.seed(seed_offset + offset)
            # random position
            if isinstance(env, arm.PlanarArmEnv) and not isinstance(env, arm.FloatingGripperEnv):
                y_sign = -1 if np.random.random() < 0.5 else 1
                init = [(np.random.random() + 0.2) * 1.0, (np.random.random() + 0.3) * 0.6 * y_sign, arm.FIXED_Z]
            else:
                init = [(np.random.random() - 0.5) * 1.7, (np.random.random() - 0.5) * 1.7, np.random.random() * 0.5]
            env.set_task_config(init=init)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()

    @staticmethod
    def tracking(level, seed_offset=0, trials=50, trial_length=300, force_gui=True):
        env = ArmGetter.env(level=level, mode=p.GUI if force_gui else p.DIRECT)
        contact_params = ArmGetter.contact_parameters(env)

        def cost_to_go(state, goal):
            return env.state_distance_two_arg(state, goal)

        def create_contact_object():
            return contact.ContactUKF(None, contact_params)

        ds, pm = ArmGetter.prior(env, use_tsf=UseTsf.NO_TRANSFORM)

        ctrl = controller.Controller()
        save_dir = '{}{}'.format(ArmGetter.env_dir, level)
        sim = arm.ExperimentRunner(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                   stop_when_done=True, save_dir=save_dir)

        # randomly distribute data
        for offset in range(trials):
            u_min, u_max = env.get_control_bounds()

            # use mode p.GUI to see what the trials look like
            seed = rand.seed(seed_offset + offset)

            contact_set = contact.ContactSet(contact_params, contact_object_factory=create_contact_object)
            ctrl = controller.GreedyControllerWithRandomWalkOnContact(env.nu, pm.dyn_net, cost_to_go, contact_set,
                                                                      u_min,
                                                                      u_max,
                                                                      force_threshold=contact_params.force_threshold,
                                                                      walk_length=6)
            # random position
            intersects_existing_objects = True
            while intersects_existing_objects:
                init = [random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)]
                init_state = np.array(init + [0, 0])
                goal = [random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)]

                # reject if init and goal is too close
                if np.linalg.norm(np.subtract(init, goal)) < 0.7:
                    continue

                env.set_task_config(init=init, goal=goal)
                env.set_state(env.goal)

                # want both goal and start to be free from collision
                p.performCollisionDetection()
                goal_intersection = False
                for obj in env.movable + env.immovable:
                    c = env.get_ee_contact_info(obj)
                    if len(c):
                        goal_intersection = True
                        break
                if goal_intersection:
                    continue

                env.set_state(init_state)
                ctrl.set_goal(env.goal)

                p.performCollisionDetection()
                for obj in env.movable + env.immovable:
                    c = env.get_ee_contact_info(obj)
                    if len(c):
                        break
                else:
                    intersects_existing_objects = False

            sim.ctrl = ctrl
            env.draw_user_text(f"seed {seed}", xy=(0.5, 0.8, -1))
            sim.run(seed)
            env.clear_debug_trajectories()

        env.close()
        # wait for it to fully close; otherwise could skip next run due to also closing that when it's created
        time.sleep(2.)


from pytorch_rrt import UniformActionSpace, ActionDescription, \
    UniformStateSpace, State, StateDescription, \
    KinodynamicRRT
from typing import Iterable


class MyAS(UniformActionSpace):
    MAX_ACTION = 1

    @classmethod
    def description(cls) -> Iterable[ActionDescription]:
        return [ActionDescription("dx", -cls.MAX_ACTION, cls.MAX_ACTION),
                ActionDescription("dy", -cls.MAX_ACTION, cls.MAX_ACTION)]


class MySS(UniformStateSpace):
    MAX_STATE = 1

    @classmethod
    def description(cls) -> Iterable[StateDescription]:
        return [StateDescription("x", -cls.MAX_STATE, cls.MAX_STATE),
                StateDescription("y", -cls.MAX_STATE, cls.MAX_STATE),
                StateDescription("rx", 0, 0),
                StateDescription("ry", 0, 0)]

    def distance(self, s1: State, s2: State) -> torch.tensor:
        return (s1 - s2).view(-1, self.dim()).norm(dim=1)


class RRTMPCWrapper:
    def __init__(self, dynamics, running_cost, goal, batch_size=512, dtype=torch.float32, device='cpu'):
        # not horizon based
        self.T = 0
        self.batch_size = batch_size
        self.state_space = MySS(dtype=dtype, device=device)
        self.action_space = MyAS(dtype=dtype, device=device)
        self.last_traj = None
        self.F = dynamics
        self.running_cost = running_cost
        self.goal = torch.tensor(goal, dtype=dtype, device=device)

        self.contact_set = None
        self.contact_data = None
        self.contact_cost = None
        self.action_id = None

        self.rrt = KinodynamicRRT(self.state_space, self.action_space, self._dynamics, self.traj_cost,
                                  update_environment=self._update_contact_data)

    def change_horizon(self, horizon):
        pass

    def _update_contact_data(self, state_id, action_id):
        center_points, points, actions = self.contact_data
        self.action_id = action_id
        if center_points is not None:
            center_points[:, :, :] = center_points[:, self.action_id, :]
            new_points = []
            new_actions = []
            for pt in points:
                pt[:, :] = pt[:, self.action_id]
                new_points.append(pt)
            for act in actions:
                act[:, :, :] = act[:, self.action_id, :]
                new_actions.append(act)
            points = new_points
            actions = new_actions

            self.contact_data = center_points, points, actions

    def _dynamics(self, state, u, environment):
        # batch process contact
        state, without_contact, self.contact_data = self.contact_set.dynamics(state, u, self.contact_data)
        # only rollout state that's not affected by contact set normally
        state[without_contact] = self.F(state[without_contact], u[without_contact])
        return state

    def command_augmented(self, state, contact_set, contact_cost):
        self.contact_set = contact_set
        self.contact_data = self.contact_set.get_batch_data_for_dynamics(self.batch_size)
        self.contact_cost = contact_cost
        res = self.rrt.plan(state, self.goal_check, goal=self.goal)
        self.last_traj = res.trajectory
        action = res.trajectory.actions[0]

        return action

    def traj_cost(self, trajectory, goal):
        states = torch.stack(trajectory.states[1:])
        actions = torch.stack(trajectory.actions)
        c = self.running_cost(states, actions)
        c = c.sum()
        # d = self.state_space.distance(states, goal)
        # return d.min()
        c_contact = self.contact_cost(self.contact_set, self.contact_data)
        if torch.is_tensor(c_contact) and self.action_id is not None:
            c_contact = c_contact[self.action_id]
        return c + c_contact

    def goal_check(self, trajectory):
        states = torch.stack(trajectory.states)
        d = self.state_space.distance(states, self.goal)
        return d.min() < 0.01

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        if self.last_traj is None or len(self.last_traj.states) < 3:
            return None
        states = torch.stack(self.last_traj.states[1:])
        return [states.view(-1, self.state_space.dim())]


def run_controller(default_run_prefix, pre_run_setup, seed=1, level=1, gating=None,
                   use_tsf=UseTsf.COORD, nominal_adapt=OnlineAdapt.NONE,
                   autonomous_recovery=online_controller.AutonomousRecovery.RETURN_STATE,
                   use_demo=False,
                   use_trap_cost=True,
                   reuse_escape_as_demonstration=False, num_frames=200,
                   run_prefix=None, run_name=None,
                   assume_all_nonnominal_dynamics_are_traps=False,
                   rep_name=None,
                   override_tampc_params=None,
                   override_mpc_params=None,
                   never_estimate_error=False,
                   project_state=True,
                   baseline=Baseline.NONE,
                   low_level_mpc=controller.ExperimentalMPPI,
                   **kwargs):
    env = ArmGetter.env(level=level, mode=p.GUI)
    logger.info("initial random seed %d", rand.seed(seed))

    ds, pm = ArmGetter.prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    demo_trajs = []
    for demo in demo_trajs:
        ds_local = ArmGetter.ds(env, demo, validation_ratio=0.)
        ds_local.update_preprocessor(ds.preprocessor)
        dss.append(ds_local)

    ensemble = []
    for s in range(10):
        _, pp = ArmGetter.prior(env, ut, rep_name=rep_name, seed=s)
        ensemble.append(pp.dyn_net)

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, env.state_distance_two_arg,
                                                       [use_tsf.name],
                                                       device=get_device(),
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': nominal_adapt},
                                                       ensemble=ensemble,
                                                       project_by_default=project_state,
                                                       local_model_kwargs=kwargs)

    # we're always going to be in the nominal mode in this case; might as well speed up testing
    if not use_demo and not reuse_escape_as_demonstration:
        gating = AlwaysSelectNominal()
    else:
        gating = hybrid_dynamics.get_gating() if gating is None else gating

    tampc_opts, mpc_opts = ArmGetter.controller_options(env)
    if override_tampc_params is not None:
        tampc_opts.update(override_tampc_params)
    if override_mpc_params is not None:
        mpc_opts.update(override_mpc_params)

    logger.debug("running with parameters\nhigh level controller: %s\nlow level MPC: %s",
                 pprint.pformat(tampc_opts), pprint.pformat(mpc_opts))

    if baseline in (Baseline.APFVO, Baseline.APFSP):
        tampc_opts.pop('trap_cost_annealing_rate')
        tampc_opts.pop('abs_unrecognized_threshold')
        tampc_opts.pop('dynamics_minimum_window')
        tampc_opts.pop('max_trap_weight')
        if baseline == Baseline.APFVO:
            rho = 0.4
            ctrl = online_controller.APFVO(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           local_min_threshold=0.03, trap_max_dist_influence=rho, repulsion_gain=0.5,
                                           **tampc_opts)
            env.draw_user_text("APF-VO baseline", xy=(0.5, 0.7, -1))
        else:
            # anything lower leads to oscillation between backing up and entering the trap's field of influence
            rho = 0.07
            ctrl = online_controller.APFSP(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                                           trap_max_dist_influence=rho, backup_scale=0.7,
                                           **tampc_opts)
            env.draw_user_text("APF-SP baseline", xy=(0.5, 0.7, -1))
    else:
        ctrl_cls = online_controller.TAMPC
        if baseline == Baseline.RANDOM_ON_CONTACT:
            ctrl_cls = online_controller.TAMPCRandomActionOnContact
        elif baseline == Baseline.AWAY_FROM_CONTACT:
            ctrl_cls = online_controller.TAMPCMoveAwayFromContact
        elif baseline == Baseline.TANGENT_TO_CONTACT:
            ctrl_cls = online_controller.TAMPCMoveTangentToContact
        ctrl = ctrl_cls(ds, hybrid_dynamics, ds.original_config(), gating=gating,
                        autonomous_recovery=autonomous_recovery,
                        assume_all_nonnominal_dynamics_are_traps=assume_all_nonnominal_dynamics_are_traps,
                        reuse_escape_as_demonstration=reuse_escape_as_demonstration,
                        use_trap_cost=use_trap_cost,
                        never_estimate_error_dynamics=never_estimate_error,
                        known_immovable_obstacles=env.immovable,
                        contact_params=ArmGetter.contact_parameters(env),
                        **tampc_opts, )
        if low_level_mpc is controller.ExperimentalMPPI:
            mpc = controller.ExperimentalMPPI(ctrl.mpc_apply_dynamics, ctrl.mpc_running_cost, ctrl.nx,
                                              u_min=ctrl.u_min, u_max=ctrl.u_max,
                                              terminal_state_cost=ctrl.mpc_terminal_cost,
                                              device=ctrl.d, **mpc_opts)
        elif low_level_mpc is RRTMPCWrapper:
            mpc = RRTMPCWrapper(ctrl.mpc_apply_dynamics, ctrl.mpc_running_cost, env.goal, dtype=ctrl.dtype,
                                device=ctrl.d)
        else:
            raise RuntimeError("Invalid low level MPC specified")
        ctrl.register_mpc(mpc)

    # x = torch.tensor([[0, 0, 10, 10], [0.7, -0.4, -6, 8], [-0.2, 0.6, 6, -5], [0.3, 0.4, -8, -3], [0.55, 0.1, 30, -30]],
    #                  dtype=ctrl.dtype, device=ctrl.d)
    # u = torch.tensor([-1, 0], dtype=ctrl.dtype, device=ctrl.d).repeat(x.shape[0], 1)
    # hybrid_dynamics.project_input_to_training_distribution(x, u, ctrl._state_dist_two_args, plot=True)

    env.draw_user_text("run seed {}".format(seed), xy=(0.5, 0.8, -1))
    ctrl.set_goal(env.goal)

    sim = arm.ExperimentRunner(env, ctrl, num_frames=num_frames, plot=False, save=True, stop_when_done=True)
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
        if baseline != Baseline.NONE:
            baseline_name = str(baseline).split('.')[1]
            affix_run_name(baseline_name)
        if run_prefix is not None:
            affix_run_name(run_prefix)
        affix_run_name(nominal_adapt.name)
        if not baseline in (Baseline.APFVO, Baseline.APFSP):
            affix_run_name(autonomous_recovery.name + ("_WITHDEMO" if use_demo else ""))
        if not project_state:
            affix_run_name("NO_PROJ")
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

    env.draw_user_text(run_name, xy=(-1.3, 0.9, -1))
    pre_run_setup(env, ctrl, ds)

    sim.run(seed, run_name)
    logger.info("last run cost %f", sum(sim.last_run_cost))
    plt.ioff()
    plt.show()

    env.close()


def test_autonomous_recovery(*args, **kwargs):
    def default_setup(env, ctrl, ds):
        return

    run_controller('auto_recover', default_setup, *args, **kwargs)


def visualize_clustering_sets(*args, num_frames=100, **kwargs):
    """Visualize clustering results"""

    def setup(env, ctrl, ds):
        goal = [0.0, 0.]
        init = [1, 0]
        # init = [0.5657+0.1, -0.0175]
        env.set_task_config(goal=goal, init=init)
        ctrl.set_goal(env.goal)

        # previous contacts made
        from torch import tensor
        from cottun.contact import ContactObject
        from tampc.dynamics import online_model
        d = 'cuda:0'
        dt = torch.float64
        if level is Levels.STRAIGHT_LINE:
            p.resetBasePositionAndOrientation(env.movable[0],
                                              (0.3877745438935845, 0.06599132022739296, 0.07692224061451539),
                                              (-0.0019374005104293949, -0.0025102144283983864, 0.018303218072636906,
                                               0.999827453869402))
            xs = [tensor([0.6668, -0.0734, 0.0000, 0.0000], device=d, dtype=dt),
                  tensor([0.6368, -0.0434, 9.3426, -9.3518], device=d, dtype=dt),
                  tensor([0.6159, -0.0396, 13.4982, -1.2811], device=d, dtype=dt),
                  tensor([0.5957, -0.0147, 9.8322, -6.9042], device=d, dtype=dt),
                  tensor([0.5657, -0.0175, 12.4257, -0.6292], device=d, dtype=dt)]
            us = [tensor([-1.0000, 1.0000], device=d, dtype=dt),
                  tensor([-0.6976, 0.1258], device=d, dtype=dt),
                  tensor([-0.6750, 0.8334], device=d, dtype=dt),
                  tensor([-1.0000, -0.0950], device=d, dtype=dt),
                  tensor([-0.4818, 0.7293], device=d, dtype=dt)]
            dxs = [tensor([-0.0300, 0.0300, 9.3426, -9.3518], device=d, dtype=dt),
                   tensor([-2.0901e-02, 3.7697e-03, 4.1557e+00, 8.0707e+00], device=d, dtype=dt),
                   tensor([-0.0202, 0.0250, -3.6660, -5.6231], device=d, dtype=dt),
                   tensor([-2.9964e-02, -2.8484e-03, 2.5935e+00, 6.2750e+00], device=d, dtype=dt),
                   tensor([-0.0144, 0.0219, -1.2938, -5.4011], device=d, dtype=dt)]
        elif level is Levels.WALL_BEHIND:
            p.resetBasePositionAndOrientation(env.movable[0],
                                              (0.4855159140630499, 0.021602734077042208, 0.07608311271110159),
                                              (-0.00015439536921398102, -0.0003589100213072153, 0.041949776970746026,
                                               0.9991196442657762))
            xs = [tensor([0.6764, 0.0672, 0.0000, 0.0000], device=d, dtype=dt),
                  tensor([0.6636, 0.0882, 10.7653, -6.6299], device=d, dtype=dt),
                  tensor([0.6637, 0.0865, 29.9954, 3.5119], device=d, dtype=dt),
                  tensor([0.6637, 0.0613, 30.0026, 1.7917], device=d, dtype=dt),
                  tensor([0.6635, 0.0558, 30.0201, 1.3798], device=d, dtype=dt),
                  tensor([0.6635, 0.0552, 30.0013, 1.4532], device=d, dtype=dt),
                  tensor([0.6726, 0.1155, 0.0000, 0.0000], device=d, dtype=dt)]
            us = [tensor([-0.6988, 0.7021], device=d, dtype=dt),
                  tensor([-1.0000, -0.0576], device=d, dtype=dt),
                  tensor([-1.0000, -0.8388], device=d, dtype=dt),
                  tensor([-0.1927, -0.1861], device=d, dtype=dt),
                  tensor([-1.0000, -0.0202], device=d, dtype=dt),
                  tensor([-0.6233, 0.9467], device=d, dtype=dt),
                  tensor([-0.9784, 0.8822], device=d, dtype=dt)]
            dxs = [tensor([-0.0127, 0.0210, 10.7653, -6.6299], device=d, dtype=dt),
                   tensor([8.7072e-05, -1.7250e-03, 1.9230e+01, 1.0142e+01], device=d, dtype=dt),
                   tensor([-2.8276e-05, -2.5146e-02, 7.2015e-03, -1.7201e+00], device=d, dtype=dt),
                   tensor([-2.2353e-04, -5.5784e-03, 1.7508e-02, -4.1194e-01], device=d, dtype=dt),
                   tensor([4.3680e-05, -6.0653e-04, -1.8815e-02, 7.3396e-02], device=d, dtype=dt),
                   tensor([5.1479e-04, 2.8380e-02, -3.8951e-02, -3.5810e+00], device=d, dtype=dt),
                   tensor([-1.4298e-02, 2.6445e-02, 3.0248e+01, 1.1995e+00], device=d, dtype=dt)]
        else:
            raise NotImplementedError(f"This task {level} is not considered")

        c = contact.ContactUKF(ctrl.dynamics.create_empty_local_model(use_prior=ctrl.contact_use_prior,
                                                                      preprocessor=ctrl.contact_preprocessing,
                                                                      nom_projection=False),
                               ArmGetter.contact_parameters(env))

        ctrl.contact_set.append(c)
        for i in range(len(xs)):
            c.add_transition(xs[i], us[i], dxs[i])

        # duplicate contact point at translated location
        import copy
        cc = copy.deepcopy(c)
        # add more points so that they're not the same size
        cc.add_transition(xs[0], us[1], dxs[1])
        cc.move_all_points(torch.tensor([0.2, 0], dtype=dt, device=d))
        ctrl.contact_set.append(cc)

        ctrl.contact_set.updated()

        # train dynamics for more iterations
        # rand.seed(0)
        # c.dynamics._recreate_all()
        # c.dynamics._fit_params(100)

        env.reset()
        env.visualize_contact_set(ctrl.contact_set)

        # evaluate the local model at certain points
        xs_eval = []

        if level is Levels.STRAIGHT_LINE:
            last_state = np.array([0.5657, -0.0175, 0, 0]) + np.array([-0.0144, 0.0219, -1.2938, -5.4011])
            env.set_state(last_state, [-0.4818, 0.7293])
            xs_eval.append(last_state)
            xs_eval.append([0.53, -0.0875, 0, 0])
            xs_eval.append([0.57, 0.0575, 0, 0])
            xs_eval.append([0.45, -0.12, 0, 0])
            xs_eval.append([10., -10, 0, 0])
        elif level is Levels.WALL_BEHIND:
            last_state = np.array([0.6583, 0.1419, 30.2485, 1.1995])
            env.set_state(last_state)
            xs_eval.append(last_state)
            xs_eval.append(last_state + np.array([-0.04, 0.04, 0, 0]))
            xs_eval.append(last_state + np.array([0.052, -0.1, 0, 0]))
            xs_eval.append(last_state + np.array([-0.08, -0.28, 0, 0]))
            xs_eval.append([10., -10, 0, 0])

        u_mag = 1
        N = 51
        t = torch.from_numpy(np.linspace(-3, 3, N)).to(dtype=dt, device=d)
        cu = torch.stack((torch.cos(t) * u_mag, torch.sin(t) * u_mag), dim=1)
        dynamics_gp = c.dynamics
        assert (isinstance(dynamics_gp, online_model.OnlineGPMixing))

        u_train = dynamics_gp.xu[:, -2:]
        t_train = torch.atan2(u_train[:, 1], u_train[:, 0]).cpu().numpy()
        y_train = dynamics_gp.y

        t = t.cpu().numpy()
        for i, x in enumerate(xs_eval):
            pos = env.get_ee_pos(x)
            env._dd.draw_point('state{}'.format(i), pos, label='{}'.format(i))
            cx = torch.tensor(x, dtype=dt, device=d).repeat(N, 1)

            applicable, _ = c.clusters_to_object(cx, cu, ctrl.contact_set.contact_max_linkage_dist,
                                                 ctrl.contact_set.u_sim)
            applicable_u = cu[applicable]
            for k, u in enumerate(applicable_u):
                env._draw_action(u.cpu() * 0.15, old_state=x, debug=k + 1 + i * N)

            applicable = applicable.cpu().numpy()

            yhat, _ = c.predict(cx, cu)
            yhat = yhat - cx
            lower, upper, yhat_mean = dynamics_gp.get_last_prediction_statistics()

            # yhat_mean, lower, upper = (ds.preprocessor.invert_transform(v, cx) for v in (yhat_mean, lower, upper))

            y_names = ['d{}'.format(x_name) for x_name in env.state_names()]
            to_plot_y_dims = [0, 1]
            num_plots = min(len(to_plot_y_dims), yhat_mean.shape[1])
            f, axes = plt.subplots(num_plots, 1, sharex='all')
            f.suptitle('dynamics at point {} ({})'.format(i, pos))
            for j, dim in enumerate(to_plot_y_dims):
                # axes[j].scatter(t, yhat[:, dim].cpu().numpy(), label='sample')
                axes[j].scatter(t_train, y_train[:, dim].cpu().numpy(), color='k', marker='*', label='train')
                axes[j].plot(t, yhat_mean[:, dim].cpu().numpy(), label='mean')
                axes[j].fill_between(t, lower[:, dim].cpu().numpy(), upper[:, dim].cpu().numpy(), alpha=0.3)
                axes[j].set_ylabel(y_names[dim])
                # axes[j].set_ylim(bottom=-0.2, top=0.2)
                draw.highlight_value_ranges(applicable, ax=axes[j], x_values=t)

            axes[0].legend()
            axes[-1].set_xlabel('action theta')

        plt.show()
        logger.info("env setup")

    level = Levels(kwargs.pop('level'))
    assert level in [Levels.STRAIGHT_LINE, Levels.WALL_BEHIND]
    run_controller('tune_avoid_nonnom_action', setup, *args, num_frames=num_frames, level=level, **kwargs)


def test_residual_model_batching(*args, **kwargs):
    """Visualize residual model GP uncertainties"""

    def setup(env, ctrl, ds):
        goal = [0.0, 0.]
        init = [1, 0]
        # init = [0.5657+0.1, -0.0175]
        env.set_task_config(goal=goal, init=init)
        ctrl.set_goal(env.goal)

        # previous contacts made
        from torch import tensor
        d = 'cuda:0'
        dt = torch.float64
        if level is Levels.STRAIGHT_LINE:
            p.resetBasePositionAndOrientation(env.movable[0],
                                              (0.3877745438935845, 0.06599132022739296, 0.07692224061451539),
                                              (-0.0019374005104293949, -0.0025102144283983864, 0.018303218072636906,
                                               0.999827453869402))
            xs = [tensor([0.6668, -0.0734, 0.0000, 0.0000], device=d, dtype=dt),
                  tensor([0.6368, -0.0434, 9.3426, -9.3518], device=d, dtype=dt),
                  tensor([0.6159, -0.0396, 13.4982, -1.2811], device=d, dtype=dt),
                  tensor([0.5957, -0.0147, 9.8322, -6.9042], device=d, dtype=dt),
                  tensor([0.5657, -0.0175, 12.4257, -0.6292], device=d, dtype=dt)]
            us = [tensor([-1.0000, 1.0000], device=d, dtype=dt),
                  tensor([-0.6976, 0.1258], device=d, dtype=dt),
                  tensor([-0.6750, 0.8334], device=d, dtype=dt),
                  tensor([-1.0000, -0.0950], device=d, dtype=dt),
                  tensor([-0.4818, 0.7293], device=d, dtype=dt)]
            dxs = [tensor([-0.0300, 0.0300, 9.3426, -9.3518], device=d, dtype=dt),
                   tensor([-2.0901e-02, 3.7697e-03, 4.1557e+00, 8.0707e+00], device=d, dtype=dt),
                   tensor([-0.0202, 0.0250, -3.6660, -5.6231], device=d, dtype=dt),
                   tensor([-2.9964e-02, -2.8484e-03, 2.5935e+00, 6.2750e+00], device=d, dtype=dt),
                   tensor([-0.0144, 0.0219, -1.2938, -5.4011], device=d, dtype=dt)]
        elif level is Levels.WALL_BEHIND:
            p.resetBasePositionAndOrientation(env.movable[0],
                                              (0.4855159140630499, 0.021602734077042208, 0.07608311271110159),
                                              (-0.00015439536921398102, -0.0003589100213072153, 0.041949776970746026,
                                               0.9991196442657762))
            xs = [tensor([0.6764, 0.0672, 0.0000, 0.0000], device=d, dtype=dt),
                  tensor([0.6636, 0.0882, 10.7653, -6.6299], device=d, dtype=dt),
                  tensor([0.6637, 0.0865, 29.9954, 3.5119], device=d, dtype=dt),
                  tensor([0.6637, 0.0613, 30.0026, 1.7917], device=d, dtype=dt),
                  tensor([0.6635, 0.0558, 30.0201, 1.3798], device=d, dtype=dt),
                  tensor([0.6635, 0.0552, 30.0013, 1.4532], device=d, dtype=dt),
                  tensor([0.6726, 0.1155, 0.0000, 0.0000], device=d, dtype=dt)]
            us = [tensor([-0.6988, 0.7021], device=d, dtype=dt),
                  tensor([-1.0000, -0.0576], device=d, dtype=dt),
                  tensor([-1.0000, -0.8388], device=d, dtype=dt),
                  tensor([-0.1927, -0.1861], device=d, dtype=dt),
                  tensor([-1.0000, -0.0202], device=d, dtype=dt),
                  tensor([-0.6233, 0.9467], device=d, dtype=dt),
                  tensor([-0.9784, 0.8822], device=d, dtype=dt)]
            dxs = [tensor([-0.0127, 0.0210, 10.7653, -6.6299], device=d, dtype=dt),
                   tensor([8.7072e-05, -1.7250e-03, 1.9230e+01, 1.0142e+01], device=d, dtype=dt),
                   tensor([-2.8276e-05, -2.5146e-02, 7.2015e-03, -1.7201e+00], device=d, dtype=dt),
                   tensor([-2.2353e-04, -5.5784e-03, 1.7508e-02, -4.1194e-01], device=d, dtype=dt),
                   tensor([4.3680e-05, -6.0653e-04, -1.8815e-02, 7.3396e-02], device=d, dtype=dt),
                   tensor([5.1479e-04, 2.8380e-02, -3.8951e-02, -3.5810e+00], device=d, dtype=dt),
                   tensor([-1.4298e-02, 2.6445e-02, 3.0248e+01, 1.1995e+00], device=d, dtype=dt)]
        else:
            raise RuntimeError("Unsupported level {}".format(level))

        # add data to the local model
        ctrl.dynamics.use_residual_model()
        for i in range(len(xs) - 1):
            next_x = xs[i + 1].clone()
            # don't try to predict next state reaction forces
            next_x[-2:] = xs[i][-2:]
            ctrl.dynamics.update(xs[i], us[i], next_x)

        ctrl.dynamics.nominal_model.plot_dynamics_at_state(xs[-1])
        # train dynamics for more iterations
        # rand.seed(0)
        # c.dynamics._recreate_all()
        # c.dynamics._fit_params(100)

        env.reset()
        env.visualize_contact_set(ctrl.contact_set)

        plt.show()
        logger.info("env setup")

    level = Levels(kwargs.pop('level'))
    assert level in [Levels.STRAIGHT_LINE, Levels.WALL_BEHIND]
    run_controller('tune_avoid_nonnom_action', setup, *args, level=level, **kwargs)


def replay_trajectory(traj_data_name, upto_index, *args, **kwargs):
    def setup(env, ctrl, ds):
        env.reset()
        ds_eval = ArmGetter.ds(env, traj_data_name, validation_ratio=0.)
        ds_eval.update_preprocessor(ds.preprocessor)

        # evaluate on a non-recovery dataset to see if rolling out the actions from the recovery set is helpful
        XU, Y, info = ds_eval.training_set(original=True)
        X, U = torch.split(XU, env.nx, dim=1)

        saved_ctrl_filename = os.path.join(cfg.ROOT_DIR, 'checkpoints', '{}{}'.format(traj_data_name, upto_index))
        # import copy
        # orig_ctrl = copy.deepcopy(ctrl)

        # need_to_compute_commands = True
        # if ctrl.load(saved_ctrl_filename):
        #     need_to_compute_commands = False

        # put the state right before the evaluated action
        x = X[upto_index].cpu().numpy()
        logger.info(np.array2string(x, separator=', '))
        # only need to do rollouts
        T = ctrl.mpc.T
        ctrl.original_horizon = 1
        for i in range(upto_index):
            env.draw_user_text(str(i), 1)
            # if need_to_compute_commands:
            #     ctrl.command(X[i].cpu().numpy())
            if i > 1:
                # will do worse than actual execution because we don't protect against immovable obstacle contact here
                ctrl.contact_set.update(X[i - 1], U[i - 1], ctrl.compare_to_goal(X[i], X[i - 1])[0], X[i, -2:])
                env.visualize_contact_set(ctrl.contact_set)
            env.step(U[i].cpu().numpy())
            # env.set_state(X[i].cpu().numpy(), U[i].cpu().numpy())
            ctrl.mpc.change_horizon(1)

        ctrl.original_horizon = T
        ctrl.mpc.change_horizon(T)

        ctrl.save(saved_ctrl_filename)
        env.set_task_config(init=X[upto_index, :2])

        logger.info("env played up to desired index")

    run_controller('replay_{}'.format(traj_data_name), setup, *args, **kwargs)


class EvaluateTask:
    @staticmethod
    def closest_distance_to_goal(file, level, visualize=True):
        env = ArmGetter.env(mode=p.GUI if visualize else p.DIRECT, level=level)
        ds = ArmGetter.ds(env, file, validation_ratio=0.)
        XU, _, _ = ds.training_set(original=True)
        X, U = torch.split(XU, ds.original_config().nx, dim=1)

        reached_states = X.cpu().numpy()
        goal_pos = env.get_ee_pos(env.goal)
        reached_ee = np.stack([env.get_ee_pos(s) for s in reached_states])

        dists = np.linalg.norm((reached_ee - goal_pos), axis=1)
        lower_bound_dist = dists.min()

        print('min dist: {} lower bound: {}'.format(lower_bound_dist, lower_bound_dist))
        env.close()
        return dists


parser = argparse.ArgumentParser(description='Experiments on the 2D grid environment')
parser.add_argument('command',
                    choices=['collect', 'collect_tracking', 'learn_representation', 'fine_tune_dynamics', 'run',
                             'evaluate', 'visualize', 'debug'],
                    help='which part of the experiment to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--representation', default='none',
                    choices=util.tsf_map.keys(),
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
parser.add_argument('--no_projection', action='store_true',
                    help='run parameter: turn off state projection before passing in to nominal model')

parser.add_argument('--baseline', default='none',
                    choices=util.baseline_map.keys(),
                    help='run parameter: use some other baseline such as (random action when in contact, '
                         'move along reaction force direction when in contact, '
                         'and move tangent to reaction force when in contact)')

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
    ut = util.tsf_map[args.representation]
    level = task_map[args.task]
    baseline = util.baseline_map[args.baseline]
    task_names = {v: k for k, v in task_map.items()}
    tampc_params = {}
    for d in args.tampc_param:
        tampc_params.update(d)
    mpc_params = {}
    for d in args.mpc_param:
        mpc_params.update(d)

    if args.command == 'collect':
        OfflineDataCollection.freespace(seed_offset=0, trials=100, trial_length=30, force_gui=args.gui)
    elif args.command == 'collect_tracking':
        for level in [Levels.SELECT1, Levels.SELECT2, Levels.SELECT3, Levels.SELECT4]:
            for offset in [7]:
                OfflineDataCollection.tracking(level, seed_offset=offset, trials=40, force_gui=True)
    elif args.command == 'learn_representation':
        for seed in args.seed:
            ArmGetter.learn_invariant(ut, seed=seed, name=arm.DIR, MAX_EPOCH=1000, BATCH_SIZE=args.batch)
    elif args.command == 'fine_tune_dynamics':
        ArmGetter.learn_model(ut, seed=args.seed[0], name="", rep_name=args.rep_name, train_epochs=1000)
    elif args.command == 'run':
        nominal_adapt = OnlineAdapt.NONE
        autonomous_recovery = online_controller.AutonomousRecovery.MAB
        use_trap_cost = not args.no_trap_cost

        if args.always_estimate_error:
            nominal_adapt = OnlineAdapt.GP_KERNEL_INDEP_OUT
        if baseline is Baseline.ADAPTIVE:
            nominal_adapt = OnlineAdapt.GP_KERNEL_INDEP_OUT
            autonomous_recovery = online_controller.AutonomousRecovery.NONE
            use_trap_cost = False
            ut = UseTsf.NO_TRANSFORM
        elif args.random_ablation:
            autonomous_recovery = online_controller.AutonomousRecovery.RANDOM
        elif baseline is Baseline.NON_ADAPTIVE:
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
                                     project_state=not args.no_projection,
                                     baseline=baseline)

    elif args.command == 'evaluate':
        task_type = arm.DIR
        # get all the trials to visualize for choosing where the obstacles are
        util.closest_distance_to_goal_whole_set(EvaluateTask.closest_distance_to_goal,
                                                args.eval_run_prefix, task_type=task_type)
    elif args.command == 'visualize':
        util.plot_task_res_dist({
            # 'auto_recover__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'ours V1', 'color': 'green', 'label': True},
            # 'auto_recover__NONE__MAB__NO_PROJ__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'ours no projection', 'color': [0.8, 0.5, 0], 'label': True},
            # 'auto_recover__RANDOM_ON_CONTACT__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'random on contact', 'color': 'red', 'label': True},
            # 'auto_recover__AWAY_FROM_CONTACT__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'away from contact', 'color': 'purple', 'label': True},
            # 'auto_recover__TANGENT_TO_CONTACT__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'tangent to contact', 'color': 'cyan', 'label': True},

            'auto_recover__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'ours V1', 'color': 'green'},
            'auto_recover__NONE__MAB__NO_PROJ__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'ours no projection', 'color': [0.8, 0.5, 0]},
            'auto_recover__RANDOM_ON_CONTACT__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'random on contact', 'color': 'red'},
            'auto_recover__AWAY_FROM_CONTACT__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'away from contact', 'color': 'purple'},
            'auto_recover__TANGENT_TO_CONTACT__NONE__MAB__7__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
                'name': 'tangent to contact', 'color': 'cyan'},

            # 'auto_recover__NONE__MAB__8__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST': {
            #     'name': 'ours V1', 'color': 'green'},
        }, 'arm_task_res.pkl', task_type='arm', figsize=(5, 7), set_y_label=True,
            task_names=task_names, success_min_dist=0.04, plot_success_vs_steps=True, plot_min_scatter=False)

    else:
        # replay_trajectory(
        #     'arm/auto_recover__NONE__MAB__5__NO_TRANSFORM__SOMETRAP__NOREUSE__AlwaysSelectNominal__TRAPCOST____0__500.mat',
        #     45,
        #     seed=0, level=5, use_tsf=ut,
        #     assume_all_nonnominal_dynamics_are_traps=False, num_frames=args.num_frames,
        #     visualize_rollout=args.visualize_rollout, run_prefix=args.run_prefix,
        #     override_tampc_params=tampc_params, override_mpc_params=mpc_params,
        #     autonomous_recovery=online_controller.AutonomousRecovery.MAB,
        #     never_estimate_error=args.never_estimate_error,
        #     other_baseline=baseline)
        visualize_clustering_sets(seed=0, level=Levels.NCB_C, use_tsf=ut)

        # d, env, config, ds = ArmGetter.free_space_env_init(0)
        # ds.update_preprocessor(ArmGetter.pre_invariant_preprocessor(use_tsf=use_tsf))
        # xu, y, trial = ds.training_set(original=True)
        # ds, pm = ArmGetter.prior(env, use_tsf)
        # yhat = pm.dyn_net.predict(xu, get_next_state=False, return_in_orig_space=True)
        # u = xu[:, env.nx:]
        # f, axes = plt.subplots(3, 1, figsize=(10, 9))
        # dims = ['x', 'y', 'z']
        # for i, dim in enumerate(dims):
        #     axes[i].scatter(u[:, i].cpu(), y[:, i].cpu(), alpha=0.1)
        #     axes[i].scatter(u[:, i].cpu(), yhat[:, i].cpu(), color="red", alpha=0.1)
        #     axes[i].set_ylabel('d{}'.format(dim))
        #     axes[i].set_xlabel('u{}'.format(i))
        #
        # plt.show()
