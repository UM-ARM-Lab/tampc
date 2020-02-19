import abc
import logging

import numpy as np
import torch

from arm_pytorch_utilities import math_utils
from arm_pytorch_utilities import tensor_utils
from pytorch_mppi import mppi

from meta_contact import cost

logger = logging.getLogger(__name__)


class Controller(abc.ABC):
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

    def reset(self):
        """Clear any controller state to be reused in another trial"""

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    @abc.abstractmethod
    def command(self, obs):
        """Given current observation, command an action"""

    def get_rollouts(self, obs):
        """Return what the predicted states for the selected action sequence is applied on obs"""
        return None


class ArtificialController(Controller):
    def __init__(self, push_magnitude):
        super().__init__()
        self.block_width = 0.075
        self.push_magnitude = push_magnitude

    def command(self, obs):
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

    def command(self, obs):
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

    def command(self, obs):
        return np.multiply(self.u, np.random.rand() * self.push_magnitude)


class FullRandomController(Controller):
    """Uniform randomly compute control along all dimensions"""

    def __init__(self, nu, u_min, u_max):
        super().__init__()
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max

    def command(self, obs):
        u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
        # logger.debug(obs)
        return u


class PreDeterminedController(Controller):
    def __init__(self, controls):
        super().__init__()
        self.u = controls
        self.j = 0

    def command(self, obs):
        if self.j >= len(self.u):
            return np.zeros_like(self.u[self.j - 1])
        u = self.u[self.j]
        self.j += 1
        return u


class MPC(Controller):
    def __init__(self, dynamics, config, Q=1, R=1, compare_to_goal=torch.sub, u_min=None, u_max=None, device='cpu',
                 terminal_cost_multiplier=0., adjust_model_pred_with_prev_error=False,
                 use_orientation_terminal_cost=False):
        super().__init__(compare_to_goal)

        self.nu = config.nu
        self.nx = config.nx
        self.dtype = torch.double
        self.d = device
        self.u_min, self.u_max = math_utils.get_bounds(u_min, u_max)
        if self.u_min is not None:
            self.u_min, self.u_max = tensor_utils.ensure_tensor(self.d, self.dtype, self.u_min, self.u_max)
        self.dynamics = dynamics
        self.adjust_model_pred_with_prev_error = adjust_model_pred_with_prev_error
        self.use_orientation_terminal_cost = use_orientation_terminal_cost

        # cost
        self.Q = tensor_utils.ensure_diagonal(Q, self.nx).to(device=self.d, dtype=self.dtype)
        self.R = tensor_utils.ensure_diagonal(R, self.nu).to(device=self.d, dtype=self.dtype)
        self.cost = cost.CostQROnlineTorch(self.goal, self.Q, self.R, self.compare_to_goal)
        self.terminal_cost_multiplier = terminal_cost_multiplier

        # analysis
        self.prediction_error = []
        self.prev_predicted_x = None
        self.prev_x = None
        self.prev_u = None
        self.diff_predicted = None

    def set_goal(self, goal):
        goal = torch.tensor(goal, dtype=self.dtype, device=self.d)
        super().set_goal(goal)
        self.cost.eetgt = goal

    def _running_cost(self, state, action):
        return self.cost(state, action)

    def _terminal_cost(self, state, action):
        # extract the last state; assume if given 3 dimensions then it's (B x T x nx)
        if len(state.shape) is 3:
            state = state[:, -1, :]
        state_loss = self.terminal_cost_multiplier * self.cost(state, action, terminal=True)
        total_loss = state_loss
        # TODO specific to block pushing (want final pose to point towards goal) - should push to inherited class
        if self.use_orientation_terminal_cost:
            diff = self.compare_to_goal(state, self.goal)
            angle_to_goal = torch.atan2(-diff[:, 1], -diff[:, 0])
            # between 0 and 10
            orientation_loss = math_utils.angular_diff_batch(angle_to_goal, state[:, 2]) ** 2
            # decrease orientation loss if we're close to the goal
            orientation_loss *= state_loss / 10
            total_loss = state_loss + orientation_loss
        return total_loss

    @abc.abstractmethod
    def _command(self, obs):
        """
        Calculate the (nu) action to take given observing the (nx) observation
        :param obs:
        :return:
        """

    def _apply_dynamics(self, state, u, t=0):
        next_state = self.dynamics(state, u)
        return self._adjust_next_state(next_state, u, t)

    def _adjust_next_state(self, next_state, u, t):
        # correct for next state with previous state's error
        if self.adjust_model_pred_with_prev_error and t is not -1 and self.diff_predicted is not None:
            # TODO generalize beyond addition (what about angles?)
            # adjustment_vector = u @ self.prev_u
            next_state += self.diff_predicted * (0.99 ** t)  # * adjustment_vector.view(-1, 1)
        return next_state

    def reset(self):
        error = torch.cat(self.prediction_error)
        median, _ = error.median(0)
        logger.debug("median relative error %s", median)
        self.prediction_error = []
        self.prev_predicted_x = None
        self.diff_predicted = None

    def command(self, obs):
        obs = tensor_utils.ensure_tensor(self.d, self.dtype, obs)
        if self.prev_predicted_x is not None:
            self.diff_predicted = self.compare_to_goal(obs.view(1, -1), self.prev_predicted_x)
            diff_actual = self.compare_to_goal(obs.view(1, -1), self.prev_x)
            relative_residual = self.diff_predicted / diff_actual
            # ignore along since it can be 0
            self.prediction_error.append(relative_residual[:, :3].abs())

        u = self._command(obs)
        if self.u_max is not None:
            u = math_utils.clip(u, self.u_min, self.u_max)

        self.prev_predicted_x = self._apply_dynamics(obs.view(1, -1), u.view(1, -1), -1)
        self.prev_x = obs
        self.prev_u = u
        return u.cpu().numpy()


class MPPI(MPC):
    def __init__(self, *args, use_bounds=True, mpc_opts=None, **kwargs):
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
        # there's interesting behaviour for MPPI if we don't pass in bounds - it'll be optimistic and try to exploit
        # regions in the dynamics where we don't know the effects of control
        if use_bounds:
            u_min, u_max = self.u_min, self.u_max
        else:
            u_min, u_max = None, None
        self.mpc = mppi.MPPI(self._apply_dynamics, self._running_cost, self.nx, u_min=u_min, u_max=u_max,
                             noise_sigma=noise_sigma, device=self.d, terminal_state_cost=self._terminal_cost,
                             **mpc_opts)

    def reset(self):
        super().reset()
        self.mpc.reset()

    def _command(self, obs):
        return self.mpc.command(obs)

    def get_rollouts(self, obs):
        return self.mpc.get_rollouts(torch.from_numpy(obs).to(dtype=self.dtype, device=self.d))[0].cpu().numpy()
