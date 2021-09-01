import numpy as np
import torch
from cottun import detection, tracking
from tampc.controller import controller


class RetrievalController(controller.Controller):

    def __init__(self, contact_detector: detection.ContactDetector, nu, dynamics, cost_to_go,
                 contact_set: tracking.ContactSetHard, u_min, u_max, num_samples=100,
                 walk_length=3):
        super().__init__()
        self.contact_detector = contact_detector
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = dynamics
        self.cost = cost_to_go
        self.num_samples = num_samples

        self.max_walk_length = walk_length
        self.remaining_random_actions = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set

    def command(self, obs, info=None):
        d = self.dynamics.device
        dtype = self.dynamics.dtype

        self.x_history.append(obs)

        if self.contact_detector.in_contact():
            self.remaining_random_actions = self.max_walk_length
            self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                    self.x_history[-1] - self.x_history[-2],
                                    self.contact_detector, torch.tensor(info['reaction']), info=info)

        if self.remaining_random_actions > 0:
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
            self.remaining_random_actions -= 1
        else:
            # take greedy action if not in contact
            state = torch.from_numpy(obs).to(device=d, dtype=dtype).repeat(self.num_samples, 1)
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=(self.num_samples, self.nu))
            u = torch.from_numpy(u).to(device=d, dtype=dtype)

            next_state = self.dynamics(state, u)
            costs = self.cost(torch.from_numpy(self.goal).to(device=d, dtype=dtype), next_state)
            min_i = torch.argmin(costs)
            u = u[min_i].cpu().numpy()

        self.u_history.append(u)
        return u


class RetrievalPredeterminedController(controller.Controller):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet, controls):
        super().__init__()
        self.contact_detector = contact_detector
        self.controls = controls
        self.i = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set

    def command(self, obs, info=None):
        self.x_history.append(obs)

        if self.contact_detector.in_contact():
            self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                    self.x_history[-1] - self.x_history[-2],
                                    self.contact_detector, torch.tensor(info['reaction']), info=info)

        if self.i < len(self.controls):
            u = self.controls[self.i]
            self.i += 1
        else:
            u = [0 for _ in range(len(self.controls[0]))]

        self.u_history.append(u)
        return u
