import abc
import logging

import numpy as np

from arm_pytorch_utilities.math_utils import rotate_wrt_origin

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
        u = rotate_wrt_origin(to_block / np.linalg.norm(to_block) * np.random.rand() * self.push_magnitude,
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
        self.u = rotate_wrt_origin(to_block / np.linalg.norm(to_block), np.random.randn() * random_angular_std)

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
        j = self.j
        u = self.u[j]
        self.j += 1
        return u
