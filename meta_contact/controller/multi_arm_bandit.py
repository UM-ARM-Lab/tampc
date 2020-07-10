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


if __name__ == "__main__":
    from arm_pytorch_utilities import rand

    rand.seed(0)

    num_arms = 7
    obs_noise = torch.ones(1) * 1
    process_noise_scaling = 0.1
    num_costs = 3
    cost_weights = torch.rand((num_arms, num_costs))
    # each arm is a row of the cost weight; normalize so it sums to 1
    cost_weights /= cost_weights.sum(dim=1).view(num_arms, 1)
    # give special meaning to the first few arms (they are 1-hot)
    cost_weights[:num_costs, :num_costs] = torch.eye(num_costs)

    print("cost weights")
    print(cost_weights)


    def _calculate_mab_process_noise():
        P = torch.eye(num_arms)
        for i in range(num_arms):
            for j in range(i + 1, num_arms):
                sim = torch.cosine_similarity(cost_weights[i], cost_weights[j], dim=0)
                P[i, j] = P[j, i] = sim
        return P


    process_noise = _calculate_mab_process_noise()
    print("process noise")
    print(process_noise)

    mab = KFMANDB(torch.zeros(num_arms), torch.eye(num_arms))
    print(mab._mean)
    print(mab._cov)
    mab.update_arms(0, 0.5, transition_cov=process_noise * process_noise_scaling, obs_cov=obs_noise)
    print(mab._mean)
    print(mab._cov)
    mab.update_arms(3, 0.2, transition_cov=process_noise * process_noise_scaling, obs_cov=obs_noise)
    print(mab._mean)
    print(mab._cov)
