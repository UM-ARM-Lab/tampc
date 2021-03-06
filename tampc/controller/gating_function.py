import abc
import torch
import numpy as np
import os
import logging
import copy

from scipy import stats
from sklearn import mixture
from arm_pytorch_utilities.model.common import LearnableParameterizedModel
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.model import make
from tampc import cfg
import enum

logger = logging.getLogger(__name__)


class DynamicsClass(enum.IntEnum):
    NOMINAL = 0
    UNRECOGNIZED = -1


class GatingFunction(abc.ABC):
    """In mixture of experts we have that the posterior is a sum over weighted components.
    This class allows sampling over weights (as opposed to giving the value of the weights)"""

    def __init__(self, use_action=False, input_slice=None):
        self.name = self.__class__.__name__
        self.input_slice = input_slice
        self.use_action = use_action
        self.nominal_ds = None

    def _slice_input(self, xu):
        if self.input_slice:
            xu = xu[:, self.input_slice]
        return xu

    def _get_training_input(self, ds, training=True):
        XU, _, _ = ds.training_set() if training else ds.validation_set()
        if len(XU) and not self.use_action:
            XU, _, _ = ds.training_set(original=True) if training else ds.validation_set(original=True)
            XU = XU.clone()
            XU[:, ds.original_config().nx:] = 0
            if self.nominal_ds is not None and self.nominal_ds.preprocessor:
                XU = self.nominal_ds.preprocessor.transform_x(XU)
        XU = self._slice_input(XU)
        return XU

    @abc.abstractmethod
    def sample_class(self, state, action, *args):
        """Sample from pi(x) class distribution and return dynamics_class index 0 to K-1

        :param state: (N x nx) states
        :param action: (N x nu) actions
        :param args: additional arguments needed
        :return: (N,) sampled class, each from 0 to K-1 where K is the total number of classes
        """
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.long)


class AlwaysSelectNominal(GatingFunction):
    def sample_class(self, state, action, *args):
        return torch.zeros((state.shape[0],), device=state.device, dtype=torch.long)


class AlwaysSelectLocal(GatingFunction):
    def sample_class(self, state, action, *args):
        return torch.ones((state.shape[0],), device=state.device, dtype=torch.long)


class ReactionForceHeuristicSelector(GatingFunction):
    def __init__(self, force_threshold, reaction_force_slice):
        super(ReactionForceHeuristicSelector, self).__init__()
        self.force_threshold = force_threshold
        self.reaction_force_slice = reaction_force_slice

    def sample_class(self, state, action, *args):
        # use local model if reaction force is beyond a certain threshold
        # doesn't work when we're using a small force to push into the wall
        r = torch.norm(state[:, self.reaction_force_slice], dim=1)
        cls = r > self.force_threshold
        self.relative_weights = torch.nn.functional.one_hot(cls.long()).double().transpose(0, 1).cpu().numpy()
        return cls.to(dtype=torch.long)


def sample_discrete_probs(probs, use_numpy=True):
    """Sample from probabilities independently

    :param probs (C x N) where each column is a distribution and so should sum to 1
    """
    C, N = probs.shape
    if use_numpy:
        raw_sample = np.random.random(N)
        sample = np.zeros_like(raw_sample)
    else:
        raw_sample = torch.rand(N, device=probs.device)
        sample = torch.zeros(N, device=probs.device, dtype=torch.long)

    for i, weight in enumerate(probs):
        this_value = (raw_sample > 0) & (raw_sample < weight)
        raw_sample -= weight
        sample[this_value] = i

    return sample


class DataProbSelector(GatingFunction):
    def __init__(self, dss, **kwargs):
        super(DataProbSelector, self).__init__(**kwargs)
        self.num_components = len(dss)
        self.nominal_ds = dss[0]
        # will end up being (num_components x num_samples)
        self.weights = None
        self.relative_weights = None

    def sample_class(self, state, action, *args):
        if not self.use_action:
            action = torch.zeros_like(action)
        xu = torch.cat((state, action), dim=1)
        if self.nominal_ds.preprocessor:
            xu = self.nominal_ds.preprocessor.transform_x(xu)

        xu = self._slice_input(xu)

        self._get_weights(xu.cpu().numpy())

        sample = sample_discrete_probs(self.relative_weights, use_numpy=True)
        return torch.tensor(sample, device=state.device, dtype=torch.long)

    @abc.abstractmethod
    def _get_weights(self, xu):
        pass


class DistributionLikelihoodSelector(DataProbSelector):
    def __init__(self, dss, component_scale=None, **kwargs):
        super().__init__(dss, **kwargs)
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
        XU = self._get_training_input(ds)
        kernel = stats.gaussian_kde(XU.transpose(0, 1).cpu().numpy())
        return kernel

    def _evaluate_pdf(self, pdf, xu):
        return pdf(xu.transpose())


class GMMSelector(DistributionLikelihoodSelector):
    def __init__(self, *args, variational=False, gmm_opts=None, **kwargs):
        self.opts = {'n_components': 5}
        if gmm_opts:
            self.opts.update(gmm_opts)
        self.variational = variational
        super().__init__(*args, **kwargs)
        if self.variational:
            self.name += ' variational'

    def _estimate_density(self, ds):
        XU = self._get_training_input(ds)
        if self.variational:
            gmm = mixture.BayesianGaussianMixture(**self.opts)
        else:
            gmm = mixture.GaussianMixture(**self.opts)
        gmm.fit(XU.cpu().numpy())
        return gmm

    def _evaluate_pdf(self, pdf, xu):
        log_prob = pdf.score_samples(xu)
        return np.exp(log_prob)


class SklearnClassifierSelector(DataProbSelector):
    def __init__(self, dss, classifier, **kwargs):
        super(SklearnClassifierSelector, self).__init__(dss, **kwargs)
        self.classifier = classifier
        xu = []
        component = []
        for i, ds in enumerate(dss):
            XU = self._get_training_input(ds)
            xu.append(XU.cpu().numpy())
            component.append(np.ones(XU.shape[0], dtype=int) * i)
        xu = np.row_stack(xu)
        component = np.concatenate(component)
        self.classifier.fit(xu, component)
        self.name = '{}'.format(self.classifier.__class__.__name__)

    def _get_weights(self, xu):
        self.weights = self.classifier.predict_proba(xu).T
        self.relative_weights = self.weights


# TODO reject samples from the other ds that have low model error
class MLPSelector(LearnableParameterizedModel, GatingFunction):
    def __init__(self, dss, retrain=False, model_opts=None, **kwargs):
        GatingFunction.__init__(self, use_action=kwargs.pop('use_action', False),
                                input_slice=kwargs.pop('input_slice', None))
        self.num_components = len(dss)
        self.nominal_ds = dss[0]
        self.weights = None
        self.relative_weights = None
        self.component_scale = None

        # TODO tune acceptance probability to maximize f1
        self.component_scale = torch.ones(self.num_components, dtype=torch.double, device=self.nominal_ds.d).view(-1, 1)
        for s in range(1, self.num_components):
            self.component_scale[s] = 70

        # we do the softmax ourselves
        opts = {'h_units': (100,), 'activation_factory': torch.nn.LeakyReLU}
        if model_opts is not None:
            opts.update(model_opts)

        config = copy.copy(self.nominal_ds.config)
        if self.input_slice:
            xu = self.nominal_ds.training_set()[0][:, self.input_slice]
            config.nx = xu.shape[1]
            config.n_input = xu.shape[1]
        config.ny = self.num_components

        self.model = make.make_sequential_network(config, **opts).to(device=self.nominal_ds.d)

        LearnableParameterizedModel.__init__(self, cfg.ROOT_DIR, **kwargs)
        self.name = "selector_{}_{}".format(self.name, config)

        if retrain or not self.load(self.get_last_checkpoint()):
            self.learn_model(dss)
        self.eval()

    def modules(self):
        return {'selector': self.model}

    def sample_class(self, state, action, *args):
        if not self.use_action:
            action = torch.zeros_like(action)
        xu = torch.cat((state, action), dim=1)
        if self.nominal_ds.preprocessor:
            xu = self.nominal_ds.preprocessor.transform_x(xu)

        xu = self._slice_input(xu)

        self.weights = self.model(xu).transpose(0, 1)
        # TODO reject cases where weights for all classes are all low (give dynamics_class -1)
        # compute relative weights (just the softmax output of the model)
        self.relative_weights = torch.nn.functional.softmax(2 * self.weights, dim=0)
        if self.component_scale is not None:
            self.relative_weights *= self.component_scale
            normalization = torch.sum(self.relative_weights, dim=0)
            self.relative_weights = self.relative_weights / normalization

        sample = sample_discrete_probs(self.relative_weights, use_numpy=False)
        return sample

    def _cat_data(self, dss, training=True):
        xu = []
        y = []
        for i, ds in enumerate(dss):
            XU = self._get_training_input(ds, training)
            if len(XU):
                # try resampling to get balanced training
                # resamples = round(self.nominal_ds.N / len(XU)) if training else 1
                resamples = 1
                for _ in range(resamples):
                    xu.append(self._slice_input(XU))
                    y.append(torch.ones(XU.shape[0], dtype=torch.long, device=XU.device) * i)
        xu = torch.cat(xu, dim=0)
        y = torch.cat(y, dim=0)
        return xu, y

    def learn_model(self, dss, max_epoch=30, batch_N=500):
        xu, y = self._cat_data(dss, training=True)
        xuv, yv = self._cat_data(dss, training=False)

        train_loader = torch.utils.data.DataLoader(load_data.SimpleDataset(xu, y), batch_size=batch_N, shuffle=True)

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        weight = None
        weight = torch.sqrt(y.shape[0] / y.unique(sorted=True, return_counts=True)[1].double())
        criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            if save_checkpoint_every_n_epochs and epoch % save_checkpoint_every_n_epochs == 0:
                self.save()

            for i_batch, data in enumerate(train_loader):
                self.step += 1

                XU, Y = data

                self.optimizer.zero_grad()
                Yhat = self.model(XU)
                loss = criterion(Yhat, Y)
                # validation and other analysis
                with torch.no_grad():
                    vloss = criterion(self.model(xuv), yv)
                    logger.debug("Epoch %d loss %f vloss %f", epoch, loss.mean().item(), vloss.mean().item())

                loss.mean().backward()
                self.optimizer.step()

        # save after training
        self.save(last=True)
