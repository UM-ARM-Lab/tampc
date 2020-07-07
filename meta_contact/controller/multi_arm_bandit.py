import abc

import torch
from torch.distributions import MultivariateNormal


class MultiArmBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms

    @abc.abstractmethod
    def select_arm_to_pull(self):
        """Select the arm of the bandit to pull"""

    @abc.abstractmethod
    def update_arms(self, arm_pulled, reward, transition_cov=None, obs_cov=None):
        """Update the reward distribution of arms based on observed reward after pulling arm"""


class KFMANDB(MultiArmBandit):
    def __init__(self, prior_mean, prior_covar):
        self._mean = prior_mean
        self._cov = prior_covar
        super(KFMANDB, self).__init__(prior_mean.shape[0])
        assert (self._mean.shape[0] == self._cov.shape[0])
        assert (self._cov.shape[0] == self._cov.shape[1])

    def select_arm_to_pull(self):
        # Thompson sampling on the bandits and select bandit with the largest sample
        arm_dist = MultivariateNormal(self._mean, covariance_matrix=self._cov)
        sample = arm_dist.sample()
        # arm with the highest sample
        return torch.argmax(sample)

    def update_arms(self, arm_pulled, reward, transition_cov=None, obs_cov=None):
        if transition_cov is None or obs_cov is None:
            raise RuntimeError("Transition and observation covariance must be provided to KF-MANDB")
        obs_matrix = torch.zeros((1, self.num_arms), dtype=self._mean.dtype, device=self._mean.device)
        obs_matrix[0, arm_pulled] = 1
        C = obs_matrix

        # Kalman predict
        pred_mean = self._mean  # no change to mean
        pred_cov = self._cov + transition_cov  # add process nosie

        # Kalman update
        innovation = reward - C @ pred_mean  # tilde y_k
        innovation_cov = C @ pred_cov @ C.t() + obs_cov  # S_k
        kalman_gain = pred_cov @ C.t() @ innovation_cov.inverse()  # K_k

        # a posteriori estimate
        self._mean = pred_mean + kalman_gain @ innovation
        self._cov = pred_cov - kalman_gain @ C @ pred_cov
        # fix to be symmetric
        self._cov = (self._cov + self._cov.t()) * 0.5
