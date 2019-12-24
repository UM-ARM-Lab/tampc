import abc

import numpy as np

from meta_contact.util import rotate_wrt_origin


class Controller(abc.ABC):
    def __init__(self):
        self.goal = None

    def set_goal(self, goal):
        self.goal = goal[2:4]

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
        to_goal = np.subtract(self.goal, (xb, yb))
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
    """Randomly push in any direction"""

    def __init__(self, push_magnitude_max):
        super().__init__()
        self.push_magnitude_max = push_magnitude_max

    def command(self, obs):
        u = (np.random.random((2,)) - 0.5) * self.push_magnitude_max
        return u


class PreDeterminedController(Controller):
    def __init__(self, controls, p):
        super().__init__()
        self.p = p
        self.u = controls
        self.j = 0

    def command(self, obs):
        j = self.j
        u = self.u[j * self.p:(j + 1) * self.p]
        self.j += 1
        return u
