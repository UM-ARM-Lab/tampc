import logging
import enum
import statistics
import typing

import torch
from tampc.controller.multi_arm_bandit import KFMANDB

from arm_pytorch_utilities import tensor_utils, preprocess
from tampc.dynamics import hybrid_model
from tampc.controller import controller, gating_function
from tampc import cost
from tampc import contact

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
        if self.dynamics._uses_local_model_api(self.dynamics.nominal_model):
            self.dynamics.nominal_model = self.dynamics.nominal_model.prior.dyn_net

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

        # self.dynamics_class_prediction[t] = cls

        # hybrid dynamics
        next_state = self.dynamics(x, u, cls)

        next_state = self.constrain_state(next_state)
        next_state[bad_states] = state[bad_states]

        return next_state

    def _compute_action(self, x):
        u = self.mpc.command(x)
        return u


class AutonomousRecovery(enum.IntEnum):
    NONE = 0
    RANDOM = 1
    RETURN_STATE = 2
    RETURN_FASTEST = 3
    MAB = 4


class StateToPositionTransformer(preprocess.SingleTransformer):
    def __init__(self, state_to_pos, nu):
        self.state_to_pos = state_to_pos
        self.nx = None
        self.nu = nu
        self.npos = None

    def fit(self, X):
        self.nx = X.shape[1]
        pos = self.state_to_pos(X[0])
        self.npos = pos.numel()

    def transform(self, X):
        pos = self.state_to_pos(X)
        if len(pos.shape) is 1:
            pos = pos.view(1, -1)
        if self.nu:
            pos = torch.cat((pos, X[:, -self.nu:]), dim=1)
        return pos

    def inverse_transform(self, X):
        return torch.cat((X, torch.zeros((X.shape[0], self.nx + self.nu - self.npos), dtype=X.dtype, device=X.device)),
                         dim=1)

    def data_dim_change(self):
        return self.npos - self.nx + self.nu, 0


class TAMPC(OnlineMPC):
    def __init__(self, *args, abs_unrecognized_threshold=10,
                 trap_cost_annealing_rate=0.97, trap_cost_init_normalization=1, manual_init_trap_weight=None,
                 max_trap_weight=None,
                 nominal_max_velocity=0,
                 nonnominal_dynamics_penalty_tolerance=0.6,
                 dynamics_minimum_window=5,
                 clearance=1e-8,
                 never_estimate_error_dynamics=False,
                 assume_all_nonnominal_dynamics_are_traps=False,
                 recovery_scale=1, recovery_horizon=5, R_env=None,
                 autonomous_recovery=AutonomousRecovery.RETURN_STATE,
                 state_to_position=None,
                 reuse_escape_as_demonstration=True, **kwargs):
        super(TAMPC, self).__init__(*args, **kwargs)
        self.abs_unrecognized_threshold = abs_unrecognized_threshold
        self.never_estimate_error_dynamics = never_estimate_error_dynamics

        self.recovery_scale = recovery_scale
        self.R_env = tensor_utils.ensure_diagonal(R_env, self.nu).to(device=self.d,
                                                                     dtype=self.dtype) if R_env is not None else self.R

        self.assume_all_nonnominal_dynamics_are_traps = assume_all_nonnominal_dynamics_are_traps
        self.trap_cost_annealing_rate = trap_cost_annealing_rate
        self.trap_cost_init_normalization = trap_cost_init_normalization
        self.auto_init_trap_cost = manual_init_trap_weight is None
        if manual_init_trap_weight is not None:
            self.trap_set_weight = manual_init_trap_weight
        self.max_trap_weight = max_trap_weight
        self.min_trap_weight = clearance ** 2 * 20  # C_m is 20 (maximum cost we care about)

        # list of strings of nominal states (separated by uses of local dynamics)
        self.nominal_dynamic_states = [[]]

        # how much single-step distance the corresponding action in u_history moved the state
        self.single_step_move_dist = []

        self.using_local_model_for_nonnominal_dynamics = False
        self.nonnominal_dynamics_start_index = -1
        # window of points we take to calculate the average trend
        self.nonnominal_dynamics_trend_len = dynamics_minimum_window
        # we have to be decreasing cost at this much compared to before nonnominal dynamics to not be in a trap
        self.nonnominal_dynamics_penalty_tolerance = nonnominal_dynamics_penalty_tolerance

        # heuristic for determining if we're a trap or not, first set when we enter local dynamics
        # if given as 0, we assume we don't start in a trap otherwise everything looks like a trap
        self.nominal_max_velocity = nominal_max_velocity

        self.autonomous_recovery_mode = False
        self.autonomous_recovery_start_index = -1
        self.autonomous_recovery_end_index = -1

        # how many consecutive turns of thinking we are out of non-nominal dynamics for it to stick (avoid jitter)
        self.leave_recovery_num_turns = 3
        self.recovery_horizon = recovery_horizon

        self.recovery_cost = None
        self.autonomous_recovery = autonomous_recovery
        self.reuse_escape_as_demonstration = reuse_escape_as_demonstration

        # return fastest properties
        self.fastest_to_choose = 4

        # contact tracking parameters
        # state distance between making contacts for distinguishing separate contacts
        self.contact_set = contact.ContactSet(0.1, self._state_dist_two_args, self.u_sim)
        self.contact_force_threshold = 0.5
        if state_to_position is not None:
            self.contact_preprocessing = preprocess.PytorchTransformer(
                StateToPositionTransformer(state_to_position, self.ds.nu),
                StateToPositionTransformer(state_to_position, 0))
        else:
            self.contact_preprocessing = preprocess.NoTransform()

    # exposed methods for MPC (not protected via _method_name)
    def register_mpc(self, mpc):
        self.mpc = mpc
        self._init_mab()

    def mpc_apply_dynamics(self, state, action, t=0):
        return self._apply_dynamics(state, action, t=t)

    def mpc_running_cost(self, state, action):
        if self.autonomous_recovery_mode:
            return self.recovery_cost(state, action)
        return self._running_cost(state, action)

    def mpc_terminal_cost(self, state, action):
        if self.autonomous_recovery_mode:
            return 0
        return self._terminal_cost(state, action)

    def get_rollouts(self, obs):
        try:
            return self.mpc.get_rollouts(torch.from_numpy(obs).to(dtype=self.dtype, device=self.d))[0].cpu().numpy()
        except AttributeError:
            logger.warning("MPC has no get_rollouts(state) method")

    # internal reasoning methods
    def _change_mpc_horizon(self, new_horizon):
        try:
            self.mpc.change_horizon(new_horizon)
        except RuntimeError:
            logger.warning("No change_horizon method on MPC")

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

        self.mpc_cost_history.append(self.mpc_running_cost(x.view(1, -1), None))
        u = self._compute_action(x)

        return u

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

            if cur_index - self.nonnominal_dynamics_start_index < self.nonnominal_dynamics_trend_len:
                return False

            # look at displacement
            before = self.nominal_max_velocity

            start = max(self.nonnominal_dynamics_start_index, self.autonomous_recovery_end_index - 1)
            lowest_current = 1e6
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

        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.RETURN_FASTEST]:
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
            left_trap = current_progress_rate > 0
            return left_trap
        elif self.autonomous_recovery is AutonomousRecovery.MAB:
            # reward for MAB (shared across arms) is displacement
            # leave if we converged / as close to targets as possible because we're not moving anymore
            # look at displacement
            before = self.nominal_max_velocity
            cur_index = len(self.x_history) - 1
            current = self._avg_displacement(max(0, cur_index - self.nonnominal_dynamics_trend_len), cur_index)
            # TODO parameterize the ratio of reluctance to leave recovery mode relative to tolerance of entering it
            converged = current < before * self.nonnominal_dynamics_penalty_tolerance * 0.05

            # and moved sufficiently far from when we recognized the trap
            moved_from_trap = self.state_dist(
                self.compare_to_goal(self.x_history[cur_index],
                                     self.x_history[self.autonomous_recovery_start_index - 1]))
            moved_suffiently_far = moved_from_trap > 1 * self.nominal_max_velocity  # number of steps at full speed away

            # left_trap = moved_suffiently_far
            left_trap = (moved_suffiently_far and converged) or cur_index - self.autonomous_recovery_start_index > 20

            logger.debug("moved from trap %f left trap? %d", moved_from_trap, left_trap)
            # logger.debug("before velocity %f current velocity %f left trap? %d", before, current, left_trap)
            return left_trap
        else:
            return False

    def _state_dist_two_args(self, xa, xb):
        diff = self.compare_to_goal(xa, xb)
        total = self.state_dist(diff)
        return total

    def _avg_displacement(self, start, end):
        total = self._state_dist_two_args(self.x_history[start], self.x_history[end])[0]
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

        return dynamics_class_test

    def _start_local_model(self, x):
        logger.debug("Entering non nominal dynamics")
        logger.debug(self.diff_predicted.norm())

        self.using_local_model_for_nonnominal_dynamics = True
        # includes the current observation
        self.nonnominal_dynamics_start_index = len(self.x_history) - 1

        if not self.never_estimate_error_dynamics:
            self.dynamics.use_temp_local_nominal_model()
            # update the local model with the last transition for entering the mode
            self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

    def _start_recovery_mode(self):
        logger.debug("Entering autonomous recovery mode")
        self.autonomous_recovery_mode = True
        # maybe also make autonomous recovery index include this observation?
        self.autonomous_recovery_start_index = len(self.x_history)

        min_index = -1
        min_ratio = 1
        start = max(self.nonnominal_dynamics_start_index, self.autonomous_recovery_end_index - 1)
        for i in range(start, len(self.x_history) - 1):
            ii, moved, expected = self.single_step_move_dist[i]
            if self._control_effort(self.u_history[ii]) > 0:
                if moved / expected < min_ratio:
                    min_ratio = moved / expected
                    min_index = i
        if min_index >= 0:
            self.trap_set.append((self.x_history[min_index], self.u_history[min_index]))

        if len(self.trap_set):
            temp = torch.stack([torch.cat((x, u)) for x, u in self.trap_set])
            logger.debug("trap set updated to be\n%s", temp)

        # different strategies for recovery mode
        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.RETURN_FASTEST,
                                        AutonomousRecovery.MAB]:
            # change mpc cost
            # return to last set of nominal states
            nominal_dynamics_set = torch.stack(self.nominal_dynamic_states[-1][-5:])
            # nominal_return_cost = cost.CostQRSet(nominal_dynamics_set, self.Q_recovery, self.R, self.compare_to_goal)
            nominal_return_cost = cost.GoalSetCost(nominal_dynamics_set, self.compare_to_goal, self.state_dist,
                                                   reduce=cost.min_cost, scale=self.recovery_scale)

            # return to last set of states that allowed the greatest single step movement
            last_states_to_consider = self.single_step_move_dist[self.nonnominal_dynamics_start_index:]
            last_states_to_consider = sorted(last_states_to_consider, key=lambda x: x[1], reverse=True)
            fastest_movement_set = [self.x_history[i] for i, moved, expected in
                                    last_states_to_consider[:self.fastest_to_choose]]
            if len(fastest_movement_set):
                fastest_movement_set = torch.stack(fastest_movement_set)
            # fastest_return_cost = cost.CostQRSet(fastest_movement_set, self.Q_recovery, self.R, self.compare_to_goal)
            fastest_return_cost = cost.GoalSetCost(fastest_movement_set, self.compare_to_goal, self.state_dist,
                                                   reduce=cost.min_cost, scale=self.recovery_scale)

            if self.autonomous_recovery is AutonomousRecovery.RETURN_STATE:
                self.recovery_cost = nominal_return_cost
            elif self.autonomous_recovery is AutonomousRecovery.RETURN_FASTEST:
                self.recovery_cost = fastest_return_cost
            elif self.autonomous_recovery is AutonomousRecovery.MAB:
                costs = [nominal_return_cost, fastest_return_cost]
                assert len(costs) == self.num_costs
                self.last_arm_pulled = None
                # linear combination of costs over the different cost functions
                self.recovery_cost = cost.ComposeCost(costs, weights=self.recovery_cost_weight)
            else:
                raise RuntimeError("Unhandled recovery strategy")

            self.original_horizon = self.mpc.T
            self._change_mpc_horizon(self.recovery_horizon)

    def recovery_cost_weight(self):
        return self.cost_weights[self.last_arm_pulled]

    def _end_recovery_mode(self):
        logger.debug("Leaving autonomous recovery mode")
        logger.debug(torch.tensor(self.dynamics_class_history[-self.leave_recovery_num_turns:]))
        self.autonomous_recovery_mode = False
        self.autonomous_recovery_end_index = len(self.x_history)

        if self.autonomous_recovery in [AutonomousRecovery.RETURN_STATE, AutonomousRecovery.RETURN_FASTEST,
                                        AutonomousRecovery.MAB]:
            self._change_mpc_horizon(self.original_horizon)
            if self.autonomous_recovery is AutonomousRecovery.MAB and self.last_arm_pulled is not None:
                self._update_mab_arm(self.last_arm_pulled.item())

    def _end_local_model(self):
        logger.debug("Leaving local model")
        logger.debug(torch.tensor(self.dynamics_class_history[-10:]))
        self.dynamics.use_normal_nominal_model()
        self.using_local_model_for_nonnominal_dynamics = False
        # start new string of nominal dynamic states
        self.nominal_dynamic_states.append([])
        # skip current state since we'll add it later
        for i in range(-2, -len(self.u_history), -1):
            if self.dynamics_class_history[i] != gating_function.DynamicsClass.NOMINAL:
                break
            elif self._control_effort(self.u_history[i]) > 0:
                self.nominal_dynamic_states[-1].insert(0, self.x_history[i])

    def _update_contact_set(self, x, u, dx):
        # associate each contact to a single object (max likelihood estimate on which object it is)
        c, _ = self.contact_set.check_which_object_applies(x, u)
        # couldn't find an existing contact
        if c is None:
            # TODO try linear model?
            # if using object-centered model, don't use preprocessor, else use default
            c = contact.ContactObject(self.dynamics.create_empty_local_model(preprocessor=self.contact_preprocessing))
            self.contact_set.append(c)
        c.add_transition(x, u, dx)
        self.contact_set.updated()

    def _compute_action(self, x):
        # use only state for dynamics_class selection; this way we can get dynamics_class before calculating action
        a = torch.zeros((1, self.nu), device=self.d, dtype=x.dtype)
        self.dynamics_class = self.gating.sample_class(x.view(1, -1), a).item()

        if len(self.x_history) > 1:
            x_index = len(self.x_history) - 2
            last_step_dist = self._avg_displacement(x_index, x_index + 1)
            # also hold the x index to reduce implicit bookkeeping
            # note that these x index is 1 greater than the u index/time index
            predicted_diff = self.compare_to_goal(self.x_history[x_index],
                                                  torch.from_numpy(self.predicted_next_state).to(device=self.d))
            predicted_step_dist = self.state_dist(predicted_diff)[0]
            self.single_step_move_dist.append((x_index, last_step_dist, predicted_step_dist))

        # update tracked contact objects
        # TODO generalize extract reaction forces
        reaction = x[-2:]
        if reaction.norm() > self.contact_force_threshold:
            # get previous x, u, dx
            px = self.x_history[-2]  # note that x is already latest in list
            self._update_contact_set(px, self.u_history[-1], self.compare_to_goal(x, px)[0])

        # in non-nominal dynamics
        if self._in_non_nominal_dynamics():
            self.dynamics_class = gating_function.DynamicsClass.UNRECOGNIZED

            if not self.using_local_model_for_nonnominal_dynamics:
                self._start_local_model(x)

        self.dynamics_class_history.append(self.dynamics_class)

        if not self.using_local_model_for_nonnominal_dynamics:
            self.trap_set_weight = max(self.trap_set_weight * self.trap_cost_annealing_rate, self.min_trap_weight)

        if self._entering_trap():
            self._start_recovery_mode()

        if self._left_trap():
            self._end_recovery_mode()
            if self.trap_cost is not None and len(self.trap_set):
                if self.max_trap_weight is not None:
                    self.trap_set_weight = self.max_trap_weight
                elif self.auto_init_trap_cost:
                    normalized_weights = [self.normalize_trapset_cost_to_state(prev_state) for prev_state in
                                          self.nominal_dynamic_states[-1][-6:]]
                    self.trap_set_weight = statistics.median(normalized_weights) * self.trap_cost_init_normalization
                else:
                    self.trap_set_weight *= (1 / self.trap_cost_annealing_rate) * 5
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
            u = self.mpc.command_augmented(x, self.contact_set)

        if self.trap_cost is not None:
            logger.debug("trap set weight %f", self.trap_set_weight)

        # in nominal dynamics
        if not self.using_local_model_for_nonnominal_dynamics:
            self.nominal_dynamic_states[-1].append(x)
            num_states = len(self.nominal_dynamic_states[-1])
            # update our estimate of max state velocity in nominal dynamics
            if num_states >= 2:
                cur_index = len(self.x_history) - 1
                start = cur_index - min(num_states, self.nonnominal_dynamics_trend_len) + 1
                vel = self._avg_displacement(start, cur_index)
                if vel > self.nominal_max_velocity:
                    self.nominal_max_velocity = vel
                    logger.debug("nominal max state velocity updated to be %f", self.nominal_max_velocity)

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

    # MAB methods
    def _init_mab(self):
        # MAB specific properties
        # these are all normalized to be relative to 1
        self.obs_noise = torch.ones(1, device=self.d) * 0.3
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
        self.mab = KFMANDB(torch.ones(self.num_arms, device=self.d) * 0.1,
                           torch.eye(self.num_arms, device=self.d) * 0.1)

    def _update_mab_arm(self, arm):
        nominal = self.nominal_max_velocity
        cur_index = len(self.x_history) - 1
        reward = self._avg_displacement(cur_index - self.pull_arm_every_n_steps, cur_index)
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


class APFVO(OnlineMPC):
    """Artificial potential field local minima escape using virtual obstacles controller"""

    def __init__(self, *args, samples=5000, trap_max_dist_influence=1, T_a=10, local_min_threshold=0.01,
                 repulsion_gain=1, **kwargs):
        self.samples = samples
        super(APFVO, self).__init__(*args, **kwargs)
        self.u_scale = self.u_max - self.u_min
        self.trap_cost = cost.ArtificialRepulsionCost(self.trap_set, self.compare_to_goal, self.state_dist,
                                                      trap_max_dist_influence, gain=repulsion_gain)
        self.cost = cost.ComposeCost([self.goal_cost, self.trap_cost])
        self.T_a = T_a
        self.local_min_threshold = local_min_threshold

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

        if len(self.x_history) > 1:
            # check if stuck in local minima
            recent_x = torch.stack(self.x_history[-self.T_a:-1])
            d = self.state_dist(self.compare_to_goal(obs, recent_x))
            # logger.debug("min dist %f", d.min())
            if torch.any(d < self.local_min_threshold):
                # place trap points where our model thinks our action will take us
                trap_state = self._apply_dynamics(obs, self.u_history[-1])
                self.trap_set.append(trap_state[0])
                logger.debug("trap added %s", trap_state[0])

        # sample a bunch of actions and run them through the dynamics
        u = torch.rand(self.samples, self.u_scale.shape[0], device=self.u_scale.device)
        u = u * self.u_scale + self.u_min
        # select action that leads to min cost
        next_state = self._apply_dynamics(obs.repeat(self.samples, 1), u)
        c = self.cost(next_state)
        ui = c.argmin()
        return u[ui]


class APFSP(OnlineMPC):
    """Artificial potential field local minima escape using switched potentials controller
    from https://link.springer.com/article/10.1007/s10846-017-0687-2
    We use the same method for detecting local minima as with APFVO, but instead of superposition of potentials
    we switch between global potential and a helical potential meant to direct around obstacles.
    Note that this method assumes a 2D x-y support, so we will take the first 2 dimensions as x and y or robot.
    """

    def __init__(self, *args, samples=5000, trap_max_dist_influence=1, obstacle_reaction=10, backup_scale=1, **kwargs):
        self.samples = samples
        super(APFSP, self).__init__(*args, **kwargs)
        self.u_scale = self.u_max - self.u_min
        self.trap_max_dist_influence = trap_max_dist_influence
        self.obstacle_reaction = obstacle_reaction
        self.backup_scale = backup_scale
        self.preloaded_control = []

    def _mpc_command(self, obs):
        t = len(self.u_history)
        x = obs
        if t > 0:
            self.dynamics.update(self.x_history[-2], self.u_history[-1], x)

        if len(self.x_history) > 1:
            # check if bumped into obstacle
            reaction_magnitude = torch.norm(x[-2:])
            if reaction_magnitude > self.obstacle_reaction:
                # place trap points where our model thinks our action will take us
                trap_state = self._apply_dynamics(obs, self.u_history[-1])
                self.trap_set.append(trap_state[0])
                # need to back up a bit as otherwise the helicoid doesn't work with flat obstacles very well
                # assume negative of control can reverse
                self.preloaded_control.append(-self.u_history[-1] * self.backup_scale)

        if len(self.preloaded_control):
            u = self.preloaded_control.pop()
            return u

        this_cost = self.goal_cost
        # check if there are any traps that are too close along the tube from current pos to goal
        active_obstacle = None
        # specialization to x-y with Euclidean distance
        xy = x[:2]
        for o in self.trap_set:
            oxy = o[:2]
            gxy = self.goal[:2]
            d = torch.norm(xy - oxy)
            if d < self.trap_max_dist_influence:
                # check if closer than previously found one
                if active_obstacle is not None:
                    dd = torch.norm(xy - active_obstacle)
                    if dd < d:
                        continue
                # check if farther than the goal then ignore
                if d > torch.norm(xy - gxy):
                    continue
                # check if obstacle is along the tube to the goal
                obstacle_alignment = torch.cosine_similarity(oxy - xy, gxy - xy, dim=0)
                if obstacle_alignment > 0.3:
                    active_obstacle = oxy

        if active_obstacle is not None:
            this_cost = cost.APFHelicoid2D(xy, active_obstacle)

        # sample a bunch of actions and run them through the dynamics
        u = torch.rand(self.samples, self.u_scale.shape[0], device=self.u_scale.device)
        u = u * self.u_scale + self.u_min
        # select action that leads to min cost
        next_state = self._apply_dynamics(obs.repeat(self.samples, 1), u)
        c = this_cost(next_state)
        ui = c.argmin()
        return u[ui]
