import abc
import logging
from typing import Dict

import numpy as np
import torch

from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import tensor_utils
from arm_pytorch_utilities import serialization
from pytorch_mppi import mppi

from tampc import cost
from cottun import tracking

logger = logging.getLogger(__name__)


class Controller(serialization.Serializable):
    """
    Controller that gives a command for a given observation (public API is ndarrays)
    Internally may keep state represented as ndarrays or tensors
    """

    def __init__(self, compare_to_goal=np.subtract):
        """
        :param compare_to_goal: function (state, goal) -> diff batched difference
        """
        self.goal = None
        self.compare_to_goal = compare_to_goal

    def state_dict(self) -> dict:
        return {'goal': self.goal}

    def load_state_dict(self, state: dict) -> bool:
        self.goal = state['goal']
        return True

    def reset(self):
        """Clear any controller state to be reused in another trial"""

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    @abc.abstractmethod
    def command(self, obs, info=None):
        """Given current observation and misc info, command an action"""

    def get_rollouts(self, obs):
        """Return what the predicted states for the selected action sequence is applied on obs"""
        return None


class ArtificialController(Controller):
    def __init__(self, push_magnitude):
        super().__init__()
        self.block_width = 0.075
        self.push_magnitude = push_magnitude

    def command(self, obs, info=None):
        x, y, xb, yb, yaw = obs
        to_goal = np.subtract(self.goal[2:4], (xb, yb))
        desired_pusher_pos = np.subtract((xb, yb), to_goal / np.linalg.norm(to_goal) * self.block_width)
        dpusher = np.subtract(desired_pusher_pos, (x, y))
        ranMag = 0.2
        return (dpusher / np.linalg.norm(dpusher) + (
            np.random.uniform(-ranMag, ranMag), np.random.uniform(-ranMag, ranMag))) * self.push_magnitude


class RandomController(Controller):
    """Randomly push towards center of block with some angle offset and randomness"""

    def __init__(self, push_magnitude, random_angular_std, random_bias_magnitude=0.5):
        super().__init__()
        self.push_magnitude = push_magnitude
        self.random_angular_std = random_angular_std
        self.fixed_angular_bias = (np.random.random() - 0.5) * random_bias_magnitude

    def command(self, obs, info=None):
        x, y, xb, yb, yaw = obs
        to_block = np.subtract((xb, yb), (x, y))
        u = math_utils.rotate_wrt_origin(to_block / np.linalg.norm(to_block) * np.random.rand() * self.push_magnitude,
                                         np.random.randn() * self.random_angular_std + self.fixed_angular_bias)
        return u


class RandomStraightController(Controller):
    """Randomly push towards block with some angle offset, moving in a straight line"""

    def __init__(self, push_magnitude, random_angular_std, start_pos, block_pos):
        super().__init__()
        self.push_magnitude = push_magnitude
        x, y = start_pos
        xb, yb = block_pos
        to_block = np.subtract((xb, yb), (x, y))
        self.u = math_utils.rotate_wrt_origin(to_block / np.linalg.norm(to_block),
                                              np.random.randn() * random_angular_std)

    def command(self, obs, info=None):
        return np.multiply(self.u, np.random.rand() * self.push_magnitude)


class FullRandomController(Controller):
    """Uniform randomly compute control along all dimensions"""

    def __init__(self, nu, u_min, u_max):
        super().__init__()
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max

    def command(self, obs, info=None):
        u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
        # logger.debug(obs)
        return u


class GreedyControllerWithRandomWalkOnContact(Controller):
    """Sample actions then take one that leads to lowest cost; take a random action after experiencing contact"""

    def __init__(self, nu, dynamics, cost_to_go, contact_set: tracking.ContactSetHard, u_min, u_max, num_samples=100,
                 force_threshold=4, walk_length=3):
        super().__init__()
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = dynamics
        self.cost = cost_to_go
        self.num_samples = num_samples
        self.force_threshold = force_threshold

        self.max_walk_length = walk_length
        self.remaining_random_actions = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set
        self.ground_truth_contact_map: Dict[int, tracking.ContactObject] = {}

    def command(self, obs, info=None):
        d = self.dynamics.device
        dtype = self.dynamics.dtype

        self.x_history.append(obs)

        if info is not None and np.linalg.norm(info['reaction']) > self.force_threshold:
            self.remaining_random_actions = self.max_walk_length
            # contact_id = info['contact_id']
            # if contact_id not in self.ground_truth_contact_map and self.contact_set is not None:
            #     self.ground_truth_contact_map[contact_id] = self.contact_set.contact_object_factory()
            #     self.contact_set.append(self.ground_truth_contact_map[contact_id])
            # self.ground_truth_contact_map[contact_id].add_transition(self.x_history[-2], self.u_history[-1],
            #                                                          self.x_history[-1] - self.x_history[-2])

        if self.remaining_random_actions > 0:
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
            self.remaining_random_actions -= 1
        else:
            # take greedy action if not in contact
            state = torch.from_numpy(obs).to(device=d, dtype=dtype).repeat(self.num_samples, 1)
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=(self.num_samples, self.nu))
            u = torch.from_numpy(u).to(device=d, dtype=dtype)

            next_state = self.dynamics(state, u)
            costs = self.cost(torch.from_numpy(self.goal).to(device=d, dtype=dtype), next_state)
            min_i = torch.argmin(costs)
            u = u[min_i].cpu().numpy()

        self.u_history.append(u)
        return u


class PreDeterminedController(Controller):
    def __init__(self, controls, u_min=None, u_max=None):
        super().__init__()
        self.u = controls
        self.j = 0
        self.u_min = u_min
        self.u_max = u_max

    def command(self, obs, info=None):
        if self.j >= len(self.u):
            return np.zeros_like(self.u[self.j - 1])
        u = self.u[self.j]
        if self.u_min is not None:
            u = np.clip(u, self.u_min, self.u_max)
        self.j += 1
        return u


class ControllerWithModelPrediction(Controller):
    def __init__(self, *args, **kwargs):
        super(ControllerWithModelPrediction, self).__init__(*args, **kwargs)
        # give the predicted state from the last u output from command, or None if command wasn't called yet
        self.predicted_next_state = None

    def state_dict(self):
        return {'predicted_next_state': self.predicted_next_state}

    def load_state_dict(self, state: dict):
        self.predicted_next_state = state['predicted_next_state']
        return True

    def prediction_error(self, observed_x):
        if self.predicted_next_state is None:
            return None
        return self.compare_to_goal(observed_x.reshape(1, -1), self.predicted_next_state)

    @abc.abstractmethod
    def predict_next_state(self, state, control):
        """Given current state and selected control, predict the next state"""

    def reset(self):
        super(ControllerWithModelPrediction, self).reset()
        self.predicted_next_state = None

    def command(self, obs, info=None):
        u = self._command(obs, info=info)
        self.predicted_next_state = self.predict_next_state(obs, u)
        return u

    @abc.abstractmethod
    def _command(self, obs, info=None):
        pass


class PreDeterminedControllerWithPrediction(PreDeterminedController, ControllerWithModelPrediction):
    def __init__(self, controls, dynamics_model, *args, **kwargs):
        super(PreDeterminedControllerWithPrediction, self).__init__(controls, *args, **kwargs)
        self.dynamics = dynamics_model
        self.d = self.dynamics.device()
        self.dtype = self.dynamics.dtype()

    def command(self, obs, info=None):
        return ControllerWithModelPrediction.command(self, obs, info)

    def _command(self, obs, info=None):
        return PreDeterminedController.command(self, obs, info)

    def predict_next_state(self, state, control):
        state, control = tensor_utils.ensure_tensor(self.d, self.dtype, state, control)
        return self.dynamics(state.view(1, -1), control.view(1, -1)).cpu().numpy()


# not really important, just has to be high enough to avoid saturating everywhere
TRAP_MAX_COST = 100000


def trap_state_cost_process(costs):
    return torch.clamp((1 / costs), 0, TRAP_MAX_COST)


def default_state_dist(state_difference):
    return state_difference[:, :2].norm(dim=1)


class MPC(ControllerWithModelPrediction):
    def __init__(self, ds, dynamics, config, Q=1, R=1, compare_to_goal=torch.sub, u_min=None, u_max=None,
                 device='cpu', u_similarity=None, state_dist=default_state_dist,
                 terminal_cost_multiplier=0., use_trap_cost=True):
        super().__init__(compare_to_goal)

        self.ds = ds
        self.nu = config.nu
        self.nx = config.nx
        self.dtype = torch.double
        self.d = device
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)
        if self.u_min is not None:
            self.u_min, self.u_max = tensor_utils.ensure_tensor(self.d, self.dtype, self.u_min, self.u_max)
        self.dynamics = dynamics
        self.state_dist = state_dist
        self.u_sim = u_similarity

        # get error per dimension to scale our expectations of accuracy
        XU, Y, _ = ds.training_set(original=True)
        X, U = torch.split(XU, config.nx, dim=1)
        Yhat = self._apply_dynamics(X, U)
        if config.predict_difference:
            Yhat = Yhat - X
        E = Yhat - Y
        E_per_dim = E.abs().mean(dim=0)
        self.model_error_per_dim = E_per_dim
        logger.info("Expected per dim dynamics error: %s", E_per_dim)

        # cost
        self.trap_set = []
        self.Q = tensor_utils.ensure_diagonal(Q, self.nx).to(device=self.d, dtype=self.dtype)
        self.R = tensor_utils.ensure_diagonal(R, self.nu).to(device=self.d, dtype=self.dtype)
        self.goal_cost = cost.CostQROnlineTorch(self.goal, self.Q, self.R, self.compare_to_goal)
        self.terminal_cost_multiplier = terminal_cost_multiplier

        self.trap_set_weight = 1
        if use_trap_cost:
            self.trap_cost = cost.TrapSetCost(self.trap_set, self.compare_to_goal, self.state_dist, self.u_sim,
                                              self._trap_cost_reduce)
            self.cost = cost.ComposeCost([self.goal_cost, self.trap_cost])
        else:
            self.trap_cost = None
            self.cost = self.goal_cost

        # analysis attributes
        self.pred_error_log = []
        self.diff_predicted = None
        self.x_history = []
        self.u_history = []
        self.orig_cost_history = []
        self.context = None

    def state_dict(self):
        # excludes data given at construction time for simplicity
        state = super().state_dict()
        state.update(
            {'pred_error_log': self.pred_error_log, 'diff_predicted': self.diff_predicted, 'x_history': self.x_history,
             'u_history': self.u_history, 'orig_cost_history': self.orig_cost_history, 'context': self.context,
             'trap_set': self.trap_set})
        return state

    def load_state_dict(self, state):
        if not super().load_state_dict(state):
            return False
        self.pred_error_log = state['pred_error_log']
        self.diff_predicted = state['diff_predicted']
        self.x_history = state['x_history']
        self.u_history = state['u_history']
        self.orig_cost_history = state['orig_cost_history']
        self.context = state['context']
        self.trap_set = state['trap_set']

        self.goal_cost = cost.CostQROnlineTorch(self.goal, self.Q, self.R, self.compare_to_goal)
        if self.trap_cost is not None:
            self.trap_cost = cost.TrapSetCost(self.trap_set, self.compare_to_goal, self.state_dist, self.u_sim,
                                              self._trap_cost_reduce)
            self.cost = cost.ComposeCost([self.goal_cost, self.trap_cost])

        return True

    def set_goal(self, goal):
        goal = torch.tensor(goal, dtype=self.dtype, device=self.d)
        super().set_goal(goal)
        self.goal_cost.goal = goal

    def _trap_cost_reduce(self, costs):
        return costs.sum(dim=0) * self.trap_set_weight

    def _running_cost(self, state, action):
        return self.cost(state, action)

    def _terminal_cost(self, state, action):
        # extract the last state; assume if given 3 dimensions then it's (B x T x nx) or (M x B x T x nx)
        if len(state.shape) == 3:
            state = state[:, -1, :]
        elif len(state.shape) == 4:
            state = state[:, :, -1, :]
        state_loss = self.terminal_cost_multiplier * self.cost(state, terminal=True)
        total_loss = state_loss
        return total_loss

    @abc.abstractmethod
    def _mpc_command(self, obs):
        """
        Calculate the (nu) action to take given observing the (nx) observation
        :param obs:
        :return:
        """

    def _apply_dynamics(self, state, u, t=0):
        return self.dynamics(state, u)

    def reset(self):
        super(MPC, self).reset()
        self.dynamics.reset()
        if len(self.pred_error_log):
            error = torch.cat(self.pred_error_log)
            median, _ = error.median(0)
            logger.debug("median relative error %s", median)
        self.pred_error_log = []
        self.diff_predicted = None
        self.x_history = []
        self.u_history = []
        self.orig_cost_history = []
        self.context = None

    def command(self, obs, info=None):
        original_obs = obs
        obs = tensor_utils.ensure_tensor(self.d, self.dtype, obs)
        self.x_history.append(obs)
        # here so that in command we have access to the latest
        self.orig_cost_history.append(self.goal_cost(obs.view(1, -1)))
        if self.predicted_next_state is not None:
            # scale with model error for each dimension
            self.diff_predicted = torch.tensor(self.prediction_error(original_obs),
                                               device=self.d) / self.model_error_per_dim
            self.pred_error_log.append(self.diff_predicted.abs())
            logger.debug("diff normalized error %.2f", self.diff_predicted.norm())

        self.context = {'diff_predicted': self.diff_predicted}
        self.context.update(info)

        u = self._mpc_command(obs)
        if self.u_max is not None:
            u = math_utils.clip(u, self.u_min, self.u_max)

        self.u_history.append(u if len(u.shape) == 1 else u[0])
        self.predicted_next_state = self.predict_next_state(obs, u)
        return u.cpu().numpy()

    def _command(self, obs, info=None):
        raise RuntimeError("Should not be calling this; command should override directly")

    def predict_next_state(self, state, control):
        return self._apply_dynamics(state.view(1, -1), control.view(1, -1), -1).cpu().numpy()


class ExperimentalMPPI(mppi.MPPI):
    def __init__(self, *args, rollout_samples=20, rollout_var_cost=0.5, rollout_var_discount=0.95, **kwargs):
        super(ExperimentalMPPI, self).__init__(*args, **kwargs)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount
        self.contact_set = None
        self.contact_cost = None

    def change_horizon(self, horizon):
        if horizon < self.T:
            self.U = self.U[:horizon]
        elif horizon > self.T:
            U = self.noise_dist.sample((horizon,))
            U[:self.T] = self.U
            self.U = U
        self.T = horizon

    @tensor_utils.handle_batch_input
    def _dynamics(self, state, u, t):
        return super(ExperimentalMPPI, self)._dynamics(state, u, t)

    @tensor_utils.handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def command_augmented(self, state, contact_set, contact_cost):
        self.contact_set = contact_set
        self.contact_cost = contact_cost
        return super(ExperimentalMPPI, self).command(state)

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        # batch process contact
        total_num = self.M * K
        contact_data = self.contact_set.get_batch_data_for_dynamics(total_num)

        states = []
        actions = []
        center_points = []
        for t in range(T):
            u = perturbed_actions[:, t].repeat(self.M, 1, 1)
            batch_dims = u.shape[:-1]
            # flatten the batch dimensions to allow easier indexing
            u = u.view(-1, u.shape[-1])
            state = state.view(-1, state.shape[-1])
            state, without_contact, contact_data = self.contact_set.dynamics(state, u, contact_data)

            # only rollout state that's not affected by contact set normally
            state[without_contact] = self._dynamics(state[without_contact], u[without_contact], t)
            c = self.running_cost(state, u)
            # TODO generalize this
            c_contact = self.contact_cost(self.contact_set, contact_data)
            c += c_contact

            # restore batch dimensions
            c = c.view(batch_dims)
            u = u.view(*batch_dims, u.shape[-1])
            state = state.view(*batch_dims, state.shape[-1])

            cost_samples += c
            cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            center_points.append(contact_data[0].clone() if contact_data[0] is not None else None)
            states.append(state.clone())
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions, center_points

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

        self.cost_total, self.states, self.actions, _ = self._compute_rollout_costs(self.perturbed_action)

        # action perturbation cost
        perturbation_cost = torch.sum(self.perturbed_action * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def get_rollouts(self, state, num_rollouts=1):
        # modified from original to include contact dynamics and contact info
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        # batch process contact
        contact_data = self.contact_set.get_batch_data_for_dynamics(num_rollouts)

        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        contact_model_active = torch.zeros((num_rollouts, T), device=self.U.device)
        center_points = []

        states[:, 0] = state
        for t in range(T):
            u = self.u_scale * self.U[t].repeat(num_rollouts, 1)
            state = states[:, t].clone().view(num_rollouts, -1)
            state, without_contact, contact_data = self.contact_set.dynamics(state, u, contact_data)

            # only rollout state that's not affected by contact set normally
            state[without_contact] = self._dynamics(state[without_contact], u[without_contact], t)
            states[:, t + 1] = state
            contact_model_active[:, t] = ~without_contact
            center_points.append(contact_data[0].clone() if contact_data[0] is not None else None)

        return states[:, 1:], contact_model_active, center_points


class MPPI_MPC(MPC):
    def __init__(self, *args, mpc_opts=None, **kwargs):
        if mpc_opts is None:
            mpc_opts = {}
        super().__init__(*args, **kwargs)
        # if not given we give it a default value
        noise_sigma = mpc_opts.pop('noise_sigma', None)
        if noise_sigma is None:
            if torch.is_tensor(self.u_max):
                noise_sigma = torch.diag(self.u_max)
            else:
                noise_mult = self.u_max if self.u_max is not None else 1
                noise_sigma = torch.eye(self.nu, dtype=self.dtype) * noise_mult
        self.mpc = ExperimentalMPPI(self._apply_dynamics, self._running_cost, self.nx, u_min=self.u_min,
                                    u_max=self.u_max,
                                    noise_sigma=noise_sigma, device=self.d, terminal_state_cost=self._terminal_cost,
                                    **mpc_opts, **self._mpc_opts())

    def reset(self):
        super().reset()
        self.mpc.reset()

    def _mpc_command(self, obs):
        return self.mpc.command(obs)

    def _mpc_opts(self):
        """For any inheriting class to supply options for contructing the MPC"""
        return {}

    def get_rollouts(self, obs):
        return self.mpc.get_rollouts(torch.from_numpy(obs).to(dtype=self.dtype, device=self.d))[0].cpu().numpy()
