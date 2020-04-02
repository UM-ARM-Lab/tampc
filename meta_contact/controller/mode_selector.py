import abc
import torch
import numpy as np


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


class AlwaysSelectNominal(ModeSelector):
    def sample_mode(self, state, action, *args):
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.int)


class AlwaysSelectLocal(ModeSelector):
    def sample_mode(self, state, action, *args):
        return torch.ones((state.shape[0],), device=state.device, dtype=torch.int)


class ReactionForceHeuristicSelector(ModeSelector):
    def sample_mode(self, state, action, *args):
        # use local model if reaction force is beyond a certain threshold
        # doesn't work when we're using a small force to push into the wall
        r = torch.norm(state[:, 4:6], dim=1)
        mode = r > 16
        return mode.to(dtype=torch.int)


class KDEProbabilitySelector(ModeSelector):
    def __init__(self, dss):
        self.num_components = len(dss)
        self.pdfs = [self.estimate_density(ds) for ds in dss]
        self.nominal_ds = dss[0]
        self.weights = None
        self.relative_weights = None

    def estimate_density(self, ds):
        from scipy import stats
        XU, _, _ = ds.training_set()
        kernel = stats.gaussian_kde(XU.transpose(0, 1).cpu().numpy())
        return kernel

    def probs_to_relative_weights(self, probs):
        # TODO could use other methods of combining the probabilities (ex. log them first to get smoother)
        normalization = np.sum(probs, axis=0)
        return np.divide(probs, normalization)

    def sample_mode(self, state, action, *args):
        xu = torch.cat((state, action), dim=1)
        if self.nominal_ds.preprocessor:
            xu = self.nominal_ds.preprocessor.transform_x(xu)

        xu = xu.transpose(0, 1).cpu().numpy()
        self.weights = [kernel(xu) for kernel in self.pdfs]

        # scale the relative likelihoods so that we can sample from it by sampling from uniform(0,1)
        self.relative_weights = self.probs_to_relative_weights(self.weights)

        raw_sample = np.random.random(self.relative_weights.shape[1])
        # since there are relatively few components compared to samples, we iterate over components
        sample = np.zeros_like(raw_sample)
        for i, weight in enumerate(self.relative_weights):
            this_value = (raw_sample > 0) & (raw_sample < weight)
            raw_sample -= weight
            sample[this_value] = i

        return torch.tensor(sample, device=state.device, dtype=torch.int)

# TODO implement nearest neighbour, GMM, learned classifier selectors
