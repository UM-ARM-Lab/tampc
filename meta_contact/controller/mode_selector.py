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
from meta_contact import cfg
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class ModeSelector(abc.ABC):
    """In mixture of experts we have that the posterior is a sum over weighted components.
    This class allows sampling over weights (as opposed to giving the value of the weights)"""

    def __init__(self):
        self.name = self.__class__.__name__

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
        super(ReactionForceHeuristicSelector, self).__init__()
        self.force_threshold = force_threshold
        self.reaction_force_slice = reaction_force_slice

    def sample_mode(self, state, action, *args):
        # use local model if reaction force is beyond a certain threshold
        # doesn't work when we're using a small force to push into the wall
        r = torch.norm(state[:, self.reaction_force_slice], dim=1)
        mode = r > self.force_threshold
        self.relative_weights = torch.nn.functional.one_hot(mode.long()).double().transpose(0, 1).cpu().numpy()
        return mode.to(dtype=torch.long)


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


class DataProbSelector(ModeSelector):
    def __init__(self, dss):
        super(DataProbSelector, self).__init__()
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

        sample = sample_discrete_probs(self.relative_weights, use_numpy=True)
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
    def __init__(self, *args, variational=False, gmm_opts=None, **kwargs):
        self.opts = {'n_components': 5}
        if gmm_opts:
            self.opts.update(gmm_opts)
        self.variational = variational
        super().__init__(*args, **kwargs)
        if self.variational:
            self.name += ' variational'

    def _estimate_density(self, ds):
        XU, _, _ = ds.training_set()
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
        self.name = '{}'.format(self.classifier.__class__.__name__)

    def _get_weights(self, xu):
        self.weights = self.classifier.predict_proba(xu).T
        self.relative_weights = self.weights


class MLPSelector(LearnableParameterizedModel, ModeSelector):
    def __init__(self, dss, retrain=False, model_opts=None, **kwargs):
        self.num_components = len(dss)
        self.nominal_ds = dss[0]
        self.relative_weights = None

        self.writer = None
        opts = {'end_block_factory': make.make_linear_end_block(activation=torch.nn.Softmax(dim=1))}
        if model_opts is not None:
            opts.update(model_opts)

        config = copy.copy(self.nominal_ds.config)
        config.ny = self.num_components

        self.model = make.make_sequential_network(config, **opts).to(device=self.nominal_ds.d)

        LearnableParameterizedModel.__init__(self, cfg.ROOT_DIR, **kwargs)
        self.name = "selector_{}_{}".format(self.name, self.nominal_ds.config)

        if retrain or not self.load(self.get_last_checkpoint()):
            self.learn_model(dss)
        self.freeze()

    def sample_mode(self, state, action, *args):
        xu = torch.cat((state, action), dim=1)
        if self.nominal_ds.preprocessor:
            xu = self.nominal_ds.preprocessor.transform_x(xu)

        # compute relative weights (just the softmax output of the model)
        self.relative_weights = self.model(xu).transpose(0, 1)

        sample = sample_discrete_probs(self.relative_weights, use_numpy=False)
        return sample

    def parameters(self):
        return self.model.parameters()

    def _model_state_dict(self):
        return self.model.state_dict()

    def _load_model_state_dict(self, saved_state_dict):
        self.model.load_state_dict(saved_state_dict)

    @staticmethod
    def _cat_data(dss, training=True):
        xu = []
        y = []
        for i, ds in enumerate(dss):
            data = ds.training_set() if training else ds.validation_set()
            XU, _, _ = data
            if len(XU):
                xu.append(XU)
                y.append(torch.ones(XU.shape[0], dtype=torch.long, device=XU.device) * i)
        xu = torch.cat(xu, dim=0)
        y = torch.cat(y, dim=0)
        return xu, y

    def learn_model(self, dss, max_epoch=200, batch_N=500):
        xu, y = self._cat_data(dss, training=True)
        xuv, yv = self._cat_data(dss, training=False)

        train_loader = torch.utils.data.DataLoader(load_data.SimpleDataset(xu, y), batch_size=batch_N, shuffle=True)

        save_checkpoint_every_n_epochs = max(max_epoch // 20, 5)

        weight = None
        # weight = torch.sqrt(y.shape[0] / y.unique(sorted=True, return_counts=True)[1].double())
        criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

        for epoch in range(0, max_epoch):  # loop over the dataset multiple times
            if self.writer is None:
                self.writer = SummaryWriter(flush_secs=20, comment=os.path.basename(self.name))
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
                    self.writer.add_scalar('loss/training', loss.mean(), self.step)
                    vloss = criterion(self.model(xuv), yv)
                    self.writer.add_scalar('loss/validation', vloss.mean(), self.step)
                    logger.debug("Epoch %d loss %f vloss %f", epoch, loss.mean().item(), vloss.mean().item())

                loss.mean().backward()
                self.optimizer.step()

        # save after training
        self.save(last=True)
