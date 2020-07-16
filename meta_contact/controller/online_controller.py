import logging
import enum
import statistics

import torch
from meta_contact.controller.multi_arm_bandit import KFMANDB

from arm_pytorch_utilities import tensor_utils
from meta_contact.dynamics import hybrid_model
from meta_contact.controller import controller, gating_function
from meta_contact import cost

logger = logging.getLogger(__name__)


def noop_constrain(state):
    return state


class OnlineMPC(controller.MPC):
    """
    Online controller with a pytorch based MPC method (CEM, MPPI)
    """

    def __init__(self, *args, constrain_state=noop_constrain,
                 gating: gating_function.GatingFunction = gating_function.AlwaysSelectNominal(),
                 **kwargs):
        self.constrain_state = constrain_state
        self.mpc = None
        self.gating = gating
        self.dynamics_class = 0
        self.dynamics_class_prediction = {}
        self.dynamics_class_history = []
        self.mpc_cost_history = []
        super().__init__(*args, **kwargs)
        assert isinstance(self.dynamics, hybrid_model.HybridDynamicsModel)

    def update_prior(self, prior):
        self.dynamics.prior = prior

    def reset(self):
        super(OnlineMPC, self).reset()
        self.dynamics_class_prediction = {}
        self.dynamics_class_history = []
        self.mpc_cost_history = []

    def predict_next_state(self, state, control):
        # we'll temporarily ensure usage of the original nominal model for predicting the next state
        current_model = self.dynamics.nominal_model
        self.dynamics.nominal_model = self.dynamics._original_nominal_model

        next_state = self.dynamics(state.view(1, -1), control.view(1, -1),
                                   torch.tensor(gating_function.DynamicsClass.NOMINAL).view(-1)).cpu().numpy()

        self.dynamics.nominal_model = current_model
        return next_state

    @tensor_utils.ensure_2d_input
    def _apply_dynamics(self, state, u, t=0):
        # hack for handling when dynamics blows up
        bad_states = state.norm(dim=1) > 1e5
        x = state
        x[bad_states] = 0
        u[bad_states] = 0

        cls = self.gating.sample_class(x, u)
        cls[bad_states] = -1

        self.dynamics_class_prediction[t] = cls

        # hybrid dynamics
        next_state = self.dynamics(x, u, cls)

        next_state = self._adjust_next_state(next_state, u, t)
        next_state = self.constrain_state(next_state)
        next_state[bad_states] = state[bad_states]

        return next_state

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

        self.mpc_cost_history.append(self.mpc.running_cost(x.view(1, -1), None))
        u = self._compute_action(x)

        return u

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class AutonomousRecovery(enum.IntEnum):
    NONE = 0
    RANDOM = 1
    RETURN_STATE = 2
    MAB = 3


# TODO take this as input instead of hard coding it
def state_displacement(before, after):
    return (before[:2] - after[:2]).norm()


class OnlineMPPI(OnlineMPC, controller.MPPI_MPC):
    def __init__(self, *args, abs_unrecognized_threshold=10,
                 trap_cost_annealing_rate=0.97,
                 assume_all_nonnominal_dynamics_are_traps=True, nonnominal_dynamics_penalty_tolerance=0.6,
                 Q_recovery=None, R_env=None,
                 autonomous_recovery=AutonomousRecovery.RETURN_STATE, reuse_escape_as_demonstration=True, **kwargs):
        super(OnlineMPPI, self).__init__(*args, **kwargs)
        self.abs_unrecognized_threshold = abs_unrecognized_threshold

        self.Q_recovery = Q_recovery.to(device=self.d) if Q_recovery is not None else self.Q
        self.R_env = tensor_utils.ensure_diagonal(R_env, self.nu).to(device=self.d,
                                                                     dtype=self.dtype) if R_env is not None else self.R

        self.assume_all_nonnominal_dynamics_are_traps = assume_all_nonnominal_dynamics_are_traps
        self.trap_cost_annealing_rate = trap_cost_annealing_rate

        # list of strings of nominal states (separated by uses of local dynamics)
        self.nominal_dynamic_states = [[]]

        self.using_local_model_for_nonnominal_dynamics = False
        self.nonnominal_dynamics_start_index = -1
        # window of points we take to calculate the average trend
        self.nonnominal_dynamics_trend_len = 4
        # we have to be decreasing cost at this much compared to before nonnominal dynamics to not be in a trap
        self.nonnominal_dynamics_penalty_tolerance = nonnominal_dynamics_penalty_tolerance

        # heuristic for determining if we're a trap or not, first set when we enter local dynamics
        # assumes we don't start in a trap
        self.nominal_avg_velocity = None

        # avoid these number of actions before entering a trap (inclusive of the transition into the trap)
        self.steps_before_entering_trap_to_avoid = 1

        self.autonomous_recovery_mode = False
        self.autonomous_recovery_start_index = -1
        self.autonomous_recovery_end_index = -1

        # how many consecutive turns of thinking we are out of non-nominal dynamics for it to stick (avoid jitter)
        self.leave_recovery_num_turns = 3

        self.recovery_cost = None
        self.autonomous_recovery = autonomous_recovery
        self.original_horizon = self.mpc.T
        self.reuse_escape_as_demonstration = reuse_escape_as_demonstration

        # MAB specific properties
        # these are all normalized to be relative to 1
        self.obs_noise = torch.ones(1, device=self.d) * 1
        self.process_noise_scaling = 0.1
        self.last_arm_pulled = None
        self.pull_arm_every_n_steps = 3
        self.turns_since_last_pull = self.pull_arm_every_n_steps
        self.num_arms = 100
        self.num_costs = 2
        self.cost_weights = torch.rand((self.num_arms, self.num_costs), device=self.d)
        # each arm is a row of the cost weight; normalize so it sums to 1
        self.cost_weights /= self.cost_weights.sum(dim=1).view(self.num_arms, 1)
        # give special meaning to the first few arms (they are 1-hot)
        self.cost_weights[:self.num_costs, :self.num_costs] = torch.eye(self.num_costs, device=self.d)
        # TODO include a more informed prior (from previous iterations)
        self.mab = KFMANDB(torch.zeros(self.num_arms, device=self.d), torch.eye(self.num_arms, device=self.d))

    def create_recovery_traj_seeder(self, *args, **kwargs):
        # deprecated
        # self.recovery_traj_seeder = RecoveryTrajectorySeeder(self, *args, **kwargs)
        pass

    def _mpc_command(self, obs):
        return OnlineMPC._mpc_command(self, obs)

    def _recovery_running_cost(self, state, action):
        return self.recovery_cost(state, action)

    def _control_effort(self, u):
        return u @ self.R_env @ u

    def _in_non_nominal_dynamics(self):
        return self.diff_predicted is not None and \
               self.dynamics_class == gating_function.DynamicsClass.NOMINAL and \
               self.diff_predicted.norm() > self.abs_unrecognized_threshold and \
               self.autonomous_recovery is not AutonomousRecovery.NONE and \
               len(self.u_history) > 1 and \
               self._control_effort(self.u_history[-1]) > 0

    def _entering_trap(self):
        # already inside trap
        if self.autonomous_recovery_mode:
            return False

        # not in non-nominal dynamics assume not a trap
        if not self.using_local_model_for_nonnominal_dynamics:
            return False

        # heuristic for determining if this a trap and should we enter autonomous recovery mode
        if self.autonomous_recovery is not AutonomousRecovery.NONE and \
                len(self.x_history) >= self.nonnominal_dynamics_trend_len and \
                self._control_effort(self.u_history[-1]) > 0:

            if self.assume_all_nonnominal_dynamics_are_traps:
                return True

            # cooldown on entering and leaving traps (can't use our progress during recovery for calculating whether we
            # enter a trap or not since the recovery policy isn't expected to decrease goal cost)
            cur_index = len(self.x_history) - 1
            if cur_index - self.autonomous_recovery_end_index < (self.nonnominal_dynamics_trend_len - 1):
                return False

            # look at displacement
            before = self.nominal_avg_velocity

            start = max(self.nonnominal_dynamics_start_index, self.autonomous_recovery_end_index - 1)
            lowest_current = before
            lowest_i = start
            for i in range(start, cur_index - self.nonnominal_dynamics_trend_len):
                current = self._avg_displacement(i, cur_index)
                if current < lowest_current:
                    lowest_current = current
                    lowest_i = i
            is_trap = lowest_current < before * self.nonnominal_dynamics_penalty_tolerance
            logger.debug("before displacement %f current lowest (since %d to %d) %f trap? %d", before, lowest_i,
                         cur_index, lowest_current, is_trap)
            return is_trap
        return False

    def _left_trap(self):
        # not in a trap to begin with
        if not self.autonomous_recovery_mode:
            return False

        # can leave if we've left non-nominal dynamics for a while
        consecutive_recognized_dynamics_class = 0
        for i in range(-1, -len(self.u_history), -1):
            if self.dynamics_class_history[i] == gating_function.DynamicsClass.UNRECOGNIZED:
                break
            if self._control_effort(self.u_history[i]) > 0:
                consecutive_recognized_dynamics_class += 1
        if consecutive_recognized_dynamics_class >= self.leave_recovery_num_turns:
            return True

        cur_index = len(self.mpc_cost_history) - 1
        if cur_index - self.autonomous_recovery_start_index < self.nonnominal_dynamics_trend_len:
            return False

        if self.autonomous_recovery is AutonomousRecovery.RETURN_STATE:
            # can also leave if we are as close as we can get to previous states
            before_trend = torch.cat(self.mpc_cost_history[self.autonomous_recovery_start_index:
                                                           self.autonomous_recovery_start_index + self.nonnominal_dynamics_trend_len])
            current_trend = torch.cat(self.mpc_cost_history[max(self.autonomous_recovery_start_index,
                                                                cur_index - self.nonnominal_dynamics_trend_len):])

            before_progress_rate = (before_trend[1:] - before_trend[:-1]).mean()
            current_progress_rate = (current_trend[1:] - current_trend[:-1]).mean()
            left_trap = before_progress_rate * 0.1 < current_progress_rate
            logger.debug("before recovery rate %f current recovery rate %f left trap? %d", before_progress_rate.item(),
                         current_progress_rate.item(), left_trap)
            return left_trap
        elif self.autonomous_recovery is AutonomousRecovery.MAB:
            # reward for MAB (shared across arms) is displacement
            # look at displacement
            # TODO consider displacement from where trap started to avoid rewarding oscillation
            before = self._avg_displacement(
                max(0, self.nonnominal_dynamics_start_index - self.nonnominal_dynamics_trend_len),
                self.nonnominal_dynamics_start_index)
            current = self._avg_displacement(max(-len(self.x_history), -self.nonnominal_dynamics_trend_len - 1), -1)
            # TODO parameterize the ratio of reluctance to leave recovery mode relative to tolerance of entering it
            left_trap = current > before * self.nonnominal_dynamics_penalty_tolerance * 3
            logger.debug("before velocity %f current velocity %f left trap? %d", before, current, left_trap)
            return left_trap
        else:
            return False

    def _avg_displacement(self, start, end):
        total = state_displacement(self.x_history[start], self.x_history[end])
        return total / (end - start)

    def _left_local_model(self):
        # not using local model to begin with
        if not self.using_local_model_for_nonnominal_dynamics:
            return False
        consecutive_recognized_dynamics_class = 0
        i = -1
        for i in range(-1, -len(self.u_history), -1):
            if self.dynamics_class_history[i] == gating_function.DynamicsClass.UNRECOGNIZED:
                break
            if self._control_effort(self.u_history[i]) > 0:
                consecutive_recognized_dynamics_class += 1
        # have to first pass the test of recognizing the dynamics
        dynamics_class_test = consecutive_recognized_dynamics_class >= self.leave_recovery_num_turns

        # if not dynamics_class_test:
        #     return False
        # # we also have to move sufficiently compared to average nominal dynamics
        # recent_movement = self._avg_displacement(len(self.x_history) + i, len(self.x_history) - 1)
        # return recent_movement > self.nominal_avg_velocity * self.nonnominal_dynamics_penalty_tolerance
        return dynamics_class_test

    def _start_local_model(self, x):
        logger.debug("Entering non nominal dynamics")
        logger.debug(self.diff_predicted.norm())

        if self.nominal_avg_velocity is None:
            self.nominal_avg_velocity = self._avg_displacement(0, len(self.x_history) - 1)
            logger.debug("determined nominal avg velocity to be %f", self.nominal_avg_velocity)

        self.using_local_model_for_nonnominal_dynamics = True
        # includes the current observation
        self.nonnominal_dynamics_start_index = len(self.x_history) - 1

        self.dynamics.use_temp_local_nominal_model()
        # update the local model with the last transition for entering the mode
        self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

    def _start_recovery_mode(self):
        logger.debug("Entering autonomous recovery mode")
        self.autonomous_recovery_mode = True
        # TODO also make autonomous recovery index include this observation
        self.autonomous_recovery_start_index = len(self.x_history)

        # avoid these points in the future
        for i in range(min(len(self.u_history), self.steps_before_entering_trap_to_avoid)):
            self.trap_set.append((self.x_history[-i - 2], self.u_history[-i - 1]))
        temp = torch.stack([torch.cat((x, u)) for x, u in self.trap_set])
        logger.debug("trap set updated to be\n%s", temp)

        # different strategies for recovery mode
        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.MAB]:
            # change mpc cost
            # return to last set of nominal states
            nominal_dynamics_set = torch.stack(self.nominal_dynamic_states[-1])
            nominal_return_cost = cost.CostQRSet(nominal_dynamics_set, self.Q_recovery, self.R, self.compare_to_goal)

            if self.autonomous_recovery is AutonomousRecovery.RETURN_STATE:
                self.recovery_cost = nominal_return_cost
            elif self.autonomous_recovery is AutonomousRecovery.MAB:
                self.last_arm_pulled = None
                local_dynamics_set = torch.stack(self.x_history[-5:-1])
                local_return_cost = cost.CostQRSet(local_dynamics_set, self.Q_recovery, self.R, self.compare_to_goal)
                # linear combination of costs over the different cost functions
                # TODO normalize costs by their value at the current state? (to get on the same magnitude)
                self.recovery_cost = cost.ComposeCost([nominal_return_cost, local_return_cost],
                                                      weights=self.recovery_cost_weight)
            else:
                raise RuntimeError("Unhandled recovery strategy")

            self.mpc.running_cost = self._recovery_running_cost
            self.mpc.terminal_state_cost = None
            self.mpc.change_horizon(10)

    def recovery_cost_weight(self):
        return self.cost_weights[self.last_arm_pulled]

    def _end_recovery_mode(self):
        logger.debug("Leaving autonomous recovery mode")
        logger.debug(torch.tensor(self.dynamics_class_history[-self.leave_recovery_num_turns:]))
        self.autonomous_recovery_mode = False
        self.autonomous_recovery_end_index = len(self.x_history)

        # deprecated
        # if we're sure that we've left an unrecognized class, save as recovery
        # if self.reuse_escape_as_demonstration:
        #     x_recovery = []
        #     u_recovery = []
        #     for i in range(self.autonomous_recovery_start_index, len(self.u_history)):
        #         if self._control_effort(self.u_history[i]) > 0:
        #             x_recovery.append(self.x_history[i])
        #             u_recovery.append(self.u_history[i])
        #     x_recovery = torch.stack(x_recovery)
        #     u_recovery = torch.stack(u_recovery)
        #     logger.info("Using data from index %d with len %d for local model",
        #                 self.autonomous_recovery_start_index, x_recovery.shape[0])
        #     self.dynamics.create_local_model(x_recovery, u_recovery)
        #     self.gating = self.dynamics.get_gating()
        #     self.recovery_traj_seeder.update_data(self.dynamics.dss)

        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.MAB]:
            # restore cost functions
            self.mpc.running_cost = self._running_cost
            self.mpc.terminal_state_cost = self._terminal_cost
            self.mpc.change_horizon(self.original_horizon)

            if self.autonomous_recovery is AutonomousRecovery.MAB and self.last_arm_pulled is not None:
                self._update_mab_arm(self.last_arm_pulled.item())

    def _end_local_model(self):
        logger.debug("Leaving local model")
        logger.debug(torch.tensor(self.dynamics_class_history[-10:]))
        self.dynamics.use_normal_nominal_model()
        self.using_local_model_for_nonnominal_dynamics = False
        # start new string of nominal dynamic states
        self.nominal_dynamic_states.append([])

    def _compute_action(self, x):
        # use only state for dynamics_class selection; this way we can get dynamics_class before calculating action
        a = torch.zeros((1, self.nu), device=self.d, dtype=x.dtype)
        self.dynamics_class = self.gating.sample_class(x.view(1, -1), a).item()

        # in non-nominal dynamics
        if self._in_non_nominal_dynamics():
            self.dynamics_class = gating_function.DynamicsClass.UNRECOGNIZED

            if not self.using_local_model_for_nonnominal_dynamics:
                self._start_local_model(x)

        # deprecated
        # if not self.autonomous_recovery_mode:
        #     self.recovery_traj_seeder.update_nominal_trajectory(self.dynamics_class, x)

        self.dynamics_class_history.append(self.dynamics_class)

        if not self.using_local_model_for_nonnominal_dynamics:
            # current_trend = torch.cat(self.orig_cost_history[-self.nonnominal_dynamics_trend_len:])
            # current_progress_rate = (current_trend[1:] - current_trend[:-1]).mean()
            # # only decrease trap set weight if we're not making progress towards goal
            # if current_progress_rate >= 0:
            self.trap_set_weight *= self.trap_cost_annealing_rate

        if self._entering_trap():
            self._start_recovery_mode()

        if self._left_trap():
            self._end_recovery_mode()
            self.normalize_trapset_cost_to_state(self.nominal_dynamic_states[-1][-1])

            normalized_weights = [self.normalize_trapset_cost_to_state(prev_state) for prev_state in
                                  self.x_history[-6:]]
            self.trap_set_weight = statistics.median(normalized_weights)
            logger.debug("tune trap cost weight %f", self.trap_set_weight)

        if self._left_local_model():
            self._end_local_model()

        if self.autonomous_recovery_mode and self.autonomous_recovery is AutonomousRecovery.RANDOM:
            u = torch.rand(self.nu, device=self.d).cuda() * (self.u_max - self.u_min) + self.u_min
        else:
            if self.autonomous_recovery_mode and self.autonomous_recovery is AutonomousRecovery.MAB:
                self.turns_since_last_pull += 1
                if self.last_arm_pulled is None or self.turns_since_last_pull >= self.pull_arm_every_n_steps:
                    # update arms if we've pulled before
                    if self.last_arm_pulled is not None:
                        self._update_mab_arm(self.last_arm_pulled.item())

                    # pull an arm and assign cost weight
                    self.last_arm_pulled = self.mab.select_arm_to_pull()
                    self.turns_since_last_pull = 0
                    logger.debug("pulled arm %d = %s", self.last_arm_pulled.item(), self.recovery_cost_weight())
            u = self.mpc.command(x)

        if self.trap_cost is not None:
            logger.debug("trap set weight %f", self.trap_set_weight)

        if not self.using_local_model_for_nonnominal_dynamics:
            self.nominal_dynamic_states[-1].append(x)

        return u

    def normalize_trapset_cost_to_state(self, x):
        if self.trap_cost is None or len(self.trap_set) is 0:
            return
        x = x.view(1, -1)
        goal_cost_at_state = self.goal_cost(x)
        # relative cost at goal would be -reference, trap cost needs to be lower than reference at goal
        trap_cost_at_goal = self.trap_cost(self.goal)
        trap_cost_at_state = self.trap_cost(x)

        logger.debug("tune trap cost goal at state %f trap at goal %f trap at state %f",
                     goal_cost_at_state, trap_cost_at_goal, trap_cost_at_state)

        # trap cost at last nominal state should have similar magnitude as the goal cost
        return goal_cost_at_state / trap_cost_at_state

    def _update_mab_arm(self, arm):
        # TODO replace this with the fixed expected nominal velocity
        nominal = self._avg_displacement(
            max(0, self.nonnominal_dynamics_start_index - self.nonnominal_dynamics_trend_len),
            self.nonnominal_dynamics_start_index)

        reward = self._avg_displacement(-self.pull_arm_every_n_steps - 1, -1)
        # normalize reward with respect to displacement in nominal environment so that a reward of 1
        # means equivalent to it
        reward /= nominal
        logger.debug("reward %f for pulling arm %d = %s", reward, arm, self.cost_weights[arm])
        self.mab.update_arms(arm, reward,
                             transition_cov=self._calculate_mab_process_noise() *
                                            self.process_noise_scaling, obs_cov=self.obs_noise)

    def _calculate_mab_process_noise(self):
        # TODO consider correlating with randomly sampling states and taking the cosine sim of their cost
        P = torch.eye(self.num_arms, device=self.d)
        for i in range(self.num_arms):
            for j in range(i + 1, self.num_arms):
                sim = torch.cosine_similarity(self.cost_weights[i], self.cost_weights[j], dim=0)
                P[i, j] = P[j, i] = sim
        return P


class NominalTrajFrom(enum.IntEnum):
    RANDOM = 0
    ROLLOUT_FROM_RECOVERY_STATES = 1
    RECOVERY_ACTIONS = 2
    NO_ADJUSTMENT = 3


class RecoveryTrajectorySeeder:
    def __init__(self, ctrl: OnlineMPPI, dss, fixed_recovery_nominal_traj=True, lookup_traj_start=True,
                 nom_traj_from=NominalTrajFrom.RECOVERY_ACTIONS):
        """
        :param ctrl:
        :param dss:
        :param fixed_recovery_nominal_traj: if false, only set nominal traj when entering dynamics_class, otherwise set every step
        """
        self.ctrl = ctrl
        self.fixed_recovery_nominal_traj = fixed_recovery_nominal_traj
        self.last_class = 0
        self.nom_traj_from = nom_traj_from
        self.lookup_traj_start = lookup_traj_start
        self.ds_nominal = dss[0]

        # local trajectories
        self.train_i = {}
        self.Z_train = {}
        self.U_train = {}
        self.update_data(dss)

    def update_data(self, dss):
        self.train_i = {}
        self.Z_train = {}
        self.U_train = {}
        for dyn_cls, ds_local in enumerate(dss):
            if dyn_cls == gating_function.DynamicsClass.NOMINAL:
                continue

            max_rollout_steps = 10
            XU, Y, info = ds_local.training_set(original=True)
            X_train, self.U_train[dyn_cls] = torch.split(XU, self.ds_nominal.original_config().nx, dim=1)
            self.train_i[dyn_cls] = len(X_train) - 1

            if self.nom_traj_from is NominalTrajFrom.ROLLOUT_FROM_RECOVERY_STATES:
                for ii in range(max(0, self.train_i[dyn_cls] - max_rollout_steps), self.train_i[dyn_cls]):
                    self.ctrl.command(X_train[ii].cpu().numpy())
                self.U_train[dyn_cls] = self.ctrl.mpc.U.clone()
            U_zero = torch.zeros_like(self.U_train[dyn_cls])
            self.Z_train[dyn_cls] = self.ds_nominal.preprocessor.transform_x(torch.cat((X_train, U_zero), dim=1))

    def update_nominal_trajectory(self, dyn_cls, state):
        if dyn_cls == gating_function.DynamicsClass.NOMINAL or self.nom_traj_from is NominalTrajFrom.NO_ADJUSTMENT:
            return False
        adjusted_trajectory = False
        # start with random noise
        U = self.ctrl.mpc.noise_dist.sample((self.ctrl.mpc.T,))
        if self.nom_traj_from is NominalTrajFrom.RANDOM:
            adjusted_trajectory = True
        else:
            if dyn_cls not in self.U_train:
                raise RuntimeError(
                    "Unrecgonized dynamics class {} (known {})".format(dyn_cls, list(self.U_train.keys())))

            # if we're not using the fixed recovery nom, then we set it if we're entering from another dynamics_class
            if self.fixed_recovery_nominal_traj or self.last_class != dyn_cls:
                adjusted_trajectory = True

                train_i = self.train_i[dyn_cls]
                # option to select where to start in training automatically from data
                if self.lookup_traj_start:
                    xu = torch.cat(
                        (state.view(1, -1), torch.zeros((1, self.ctrl.nu), device=state.device, dtype=state.dtype)),
                        dim=1)
                    z = self.ds_nominal.preprocessor.transform_x(xu)
                    dists = (self.Z_train[dyn_cls] - z).norm(dim=1)
                    # TODO prioritize points closer to the escape?
                    train_i = dists.argmin()

                ctrl_rollout_steps = min(len(self.U_train[dyn_cls]) - train_i, self.ctrl.mpc.T - 1)
                U[1:1 + ctrl_rollout_steps] = self.U_train[dyn_cls][train_i:train_i + ctrl_rollout_steps]

        if adjusted_trajectory:
            self.ctrl.mpc.U = U
        self.last_class = dyn_cls
        return adjusted_trajectory
