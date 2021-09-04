import os

import numpy as np
import pybullet as p
import torch
from arm_pytorch_utilities import rand
from arm_pytorch_utilities.math_utils import angular_diff

from cottun import detection, tracking
from tampc import cfg
from tampc.controller import controller
from tampc.env.pybullet_env import closest_point_on_surface, ContactInfo


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


def rot_2d_mat_to_angle(T):
    """T: bx3x3 homogenous transforms or bx2x2 rotation matrices"""
    return torch.atan2(T[:, 1, 0], T[:, 0, 0])


def sample_model_points(object_id, num_points=100, reject_too_close=0.002, force_z=None, seed=0, name=""):
    fullname = os.path.join(cfg.DATA_DIR, f'model_points_cache.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache:
            cache[name] = {}
        if seed in cache[name]:
            return cache[name][seed]
    else:
        cache = {name: {}}

    with rand.SavedRNG():
        rand.seed(seed)
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(object_id)
        z = orig_pos[2]
        # first reset to canonical location
        canonical_pos = [0, 0, z]
        p.resetBasePositionAndOrientation(object_id, canonical_pos, [0, 0, 0, 1])

        points = []
        sigma = 0.1
        while len(points) < num_points:
            tester_pos = np.r_[np.random.randn(2) * sigma, z]
            # sample an object at random points around this object and find closest point to it
            closest = closest_point_on_surface(object_id, tester_pos)
            pt = closest[ContactInfo.POS_A]
            if force_z is not None:
                pt = (pt[0], pt[1], force_z)
            if len(points) > 0:
                d = np.subtract(points, pt)
                d = np.linalg.norm(d, axis=1)
                if np.any(d < reject_too_close):
                    continue
            points.append(pt)

    p.resetBasePositionAndOrientation(object_id, orig_pos, orig_orientation)

    points = torch.tensor(points)

    cache[name][seed] = points
    torch.save(cache, fullname)

    return points


def pose_error(target_pose, guess_pose):
    # mirrored, so being off by 180 degrees is fine
    yaw_error = min(abs(angular_diff(target_pose[-1], guess_pose[-1])),
                    abs(angular_diff(target_pose[-1] + np.pi, guess_pose[-1])))
    pos_error = np.linalg.norm(np.subtract(target_pose[:2], guess_pose[:2]))
    return pos_error, yaw_error