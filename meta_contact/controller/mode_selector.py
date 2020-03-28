import abc
import torch


class ModeSelector(abc.ABC):
    """In mixture of experts we have that the posterior is a sum over weighted components.
    This class allows sampling over weights (as opposed to giving the value of the weights)"""

    @abc.abstractmethod
    def sample_mode(self, state, action, *args):
        """Sample from pi(x) mode distribution and return mode index 0 to K-1

        :param state: (N x nx) states
        :param action: (N x nu) actions
        :param args: additional arguments needed
        :return: (N,) sampled mode, each from 0 to K-1 where K is the total number of modes
        """
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.int)


class ReactionForceHeuristicSelector(ModeSelector):
    def sample_mode(self, state, action, *args):
        # use local model if reaction force is beyond a certain threshold
        # doesn't work when we're using a small force to push into the wall
        r = torch.norm(state[:, 4:6], dim=1)
        mode = r > 16
        return mode.to(dtype=torch.int)
