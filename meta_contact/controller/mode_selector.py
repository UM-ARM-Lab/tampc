import abc
import torch
import numpy as np

from scipy import stats
from sklearn import mixture


class ModeSelector(abc.ABC):
    """In mixture of experts we have that the posterior is a sum over weighted components.
    This class allows sampling over weights (as opposed to giving the value of the weights)"""

    def name(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def sample_mode(self, state, action, *args):
        """Sample from pi(x) mode distribution and return mode index 0 to K-1

        :param state: (N x nx) states
        :param action: (N x nu) actions
        :param args: additional arguments needed
        :return: (N,) sampled mode, each from 0 to K-1 where K is the total number of modes
        """
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.long)


class AlwaysSelectNominal(ModeSelector):
    def sample_mode(self, state, action, *args):
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.long)


class AlwaysSelectLocal(ModeSelector):
    def sample_mode(self, state, action, *args):
        return torch.ones((state.shape[0],), device=state.device, dtype=torch.long)


class ReactionForceHeuristicSelector(ModeSelector):
    def __init__(self, force_threshold, reaction_force_slice):
        self.force_threshold = force_threshold
        self.reaction_force_slice = reaction_force_slice

    def sample_mode(self, state, action, *args):
        # use local model if reaction force is beyond a certain threshold
        # doesn't work when we're using a small force to push into the wall
        r = torch.norm(state[:, self.reaction_force_slice], dim=1)
        mode = r > self.force_threshold
        self.relative_weights = torch.nn.functional.one_hot(mode.long()).double().transpose(0, 1).cpu().numpy()
        return mode.to(dtype=torch.long)


class DataProbSelector(ModeSelector):
    def __init__(self, dss):
        self.num_components = len(dss)
        self.nominal_ds = dss[0]
        # will end up being (num_components x num_samples)
        self.weights = None
        self.relative_weights = None

    def sample_mode(self, state, action, *args):
        xu = torch.cat((state, action), dim=1)
        if self.nominal_ds.preprocessor:
            xu = self.nominal_ds.preprocessor.transform_x(xu)

        self._get_weights(xu.cpu().numpy())

        raw_sample = np.random.random(self.relative_weights.shape[1])
        # since there are relatively few components compared to samples, we iterate over components
        sample = np.zeros_like(raw_sample)
        for i, weight in enumerate(self.relative_weights):
            this_value = (raw_sample > 0) & (raw_sample < weight)
            raw_sample -= weight
            sample[this_value] = i

        return torch.tensor(sample, device=state.device, dtype=torch.long)

    @abc.abstractmethod
    def _get_weights(self, xu):
        pass


class DistributionLikelihoodSelector(DataProbSelector):
    def __init__(self, dss, component_scale=None):
        super().__init__(dss)
        self.pdfs = [self._estimate_density(ds) for ds in dss]
        self.component_scale = np.array(component_scale).reshape(-1, 1) if component_scale else None

    @abc.abstractmethod
    def _estimate_density(self, ds):
        """Get PDF or something that allows you to evaluate the PDF"""
        return None

    def _probs_to_relative_weights(self, probs):
        # TODO could use other methods of combining the probabilities (ex. log them first to get smoother)
        normalization = np.sum(probs, axis=0)
        return np.divide(probs, normalization)

    @abc.abstractmethod
    def _evaluate_pdf(self, pdf, xu):
        """Get probabilities by evaluating pdf on xu"""

    def _get_weights(self, xu):
        self.weights = np.stack([self._evaluate_pdf(pdf, xu) for pdf in self.pdfs])
        if self.component_scale is not None:
            self.weights *= self.component_scale
        # scale the relative likelihoods so that we can sample from it by sampling from uniform(0,1)
        self.relative_weights = self._probs_to_relative_weights(self.weights)


class KDESelector(DistributionLikelihoodSelector):
    def _estimate_density(self, ds):
        XU, _, _ = ds.training_set()
        kernel = stats.gaussian_kde(XU.transpose(0, 1).cpu().numpy())
        return kernel

    def _evaluate_pdf(self, pdf, xu):
        return pdf(xu.transpose())


class GMMSelector(DistributionLikelihoodSelector):
    def __init__(self, *args, mixture_components=5, **kwargs):
        self.mixture_components = mixture_components
        super().__init__(*args, **kwargs)

    def _estimate_density(self, ds):
        XU, _, _ = ds.training_set()
        gmm = mixture.GaussianMixture(n_components=self.mixture_components, covariance_type='full')
        gmm.fit(XU.cpu().numpy())
        return gmm

    def _evaluate_pdf(self, pdf, xu):
        log_prob = pdf.score_samples(xu)
        return np.exp(log_prob)


class SklearnClassifierSelector(DataProbSelector):
    def __init__(self, dss, classifier):
        super(SklearnClassifierSelector, self).__init__(dss)
        self.classifier = classifier
        xu = []
        component = []
        for i, ds in enumerate(dss):
            XU, _, _ = ds.training_set()
            xu.append(XU.cpu().numpy())
            component.append(np.ones(XU.shape[0], dtype=int) * i)
        xu = np.row_stack(xu)
        component = np.concatenate(component)
        self.classifier.fit(xu, component)

    def name(self):
        return '{}'.format(self.classifier.__class__.__name__)

    def _get_weights(self, xu):
        self.weights = self.classifier.predict_proba(xu).T
        self.relative_weights = self.weights

# TODO implement learned classifier selectors
