import abc
import os
import typing

import numpy as np
import pybullet as p
import torch
from arm_pytorch_utilities import rand
from arm_pytorch_utilities.math_utils import angular_diff
from arm_pytorch_utilities.controller import Controller
from pynput import keyboard

from stucco import detection, tracking
from stucco.cluster_baseline import process_labels_with_noise
from stucco.defines import NO_CONTACT_ID
from stucco import cfg
from stucco.env.env import InfoKeys
from stucco.env.pybullet_env import closest_point_on_surface, ContactInfo, state_action_color_pairs


class RetrievalController(Controller):

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


class RetrievalPredeterminedController(Controller):

    def __init__(self, controls, nu=None):
        super().__init__()
        self.controls = controls
        self.i = 0
        self.nu = nu or len(self.controls[0])

        self.x_history = []
        self.u_history = []

    def done(self):
        return self.i >= len(self.controls)

    @abc.abstractmethod
    def update(self, obs, info):
        pass

    def command(self, obs, info=None):
        self.x_history.append(obs)

        if len(self.x_history) > 1:
            self.update(obs, info)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = self.controls[self.i]
            self.i += 1

        self.u_history.append(u)
        return u


class OursRetrievalPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet, controls,
                 nu=None):
        super().__init__(controls, nu=nu)
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.contact_indices = []

    def update(self, obs, info):
        if self.contact_detector.in_contact():
            self.contact_indices.append(self.i)

        self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                self.x_history[-1] - self.x_history[-2],
                                self.contact_detector, torch.tensor(info['reaction']), info=info)


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


class TrackingMethod:
    """Common interface for each tracking method including ours and baselines"""

    @abc.abstractmethod
    def __iter__(self):
        """Iterating over this provides a set of contact points corresponding to an object"""

    @abc.abstractmethod
    def create_predetermined_controller(self, controls):
        """Return a predetermined controller that updates the method when querying for a command"""

    @abc.abstractmethod
    def visualize_contact_points(self, env):
        """Render the tracked contact points in the given environment"""

    @abc.abstractmethod
    def get_labelled_moved_points(self, labels):
        """Return the final position of the tracked points as well as their object label"""


class SoftTrackingIterator:
    def __init__(self, pts, to_iter):
        self.pts = pts
        self.to_iter = to_iter

    def __next__(self):
        indices = next(self.to_iter)
        return self.pts[indices]


class OurTrackingMethod(TrackingMethod):
    def __init__(self, env):
        self.env = env
        self.ctrl = None

    @property
    @abc.abstractmethod
    def contact_set(self) -> tracking.ContactSet:
        """Return some contact set"""

    def visualize_contact_points(self, env):
        env.visualize_contact_set(self.contact_set)

    def create_predetermined_controller(self, controls):
        self.ctrl = OursRetrievalPredeterminedController(self.env.contact_detector, self.contact_set, controls)
        return self.ctrl


class OurSoftTrackingMethod(OurTrackingMethod):
    def __init__(self, env, contact_params, pt_to_config):
        self.contact_params = contact_params
        self._contact_set = tracking.ContactSetSoft(pt_to_config, self.contact_params)
        super(OurSoftTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetSoft:
        return self._contact_set

    def __iter__(self):
        pts = self.contact_set.get_posterior_points()
        to_iter = self.contact_set.get_hard_assignment(self.contact_set.p.hard_assignment_threshold)
        return SoftTrackingIterator(pts, iter(to_iter))

    def get_labelled_moved_points(self, labels):
        groups = self.contact_set.get_hard_assignment(self.contact_params.hard_assignment_threshold)
        contact_indices = torch.tensor(self.ctrl.contact_indices, device=groups[0].device)
        # self.ctrl.contact_indices = []

        for group_id, group in enumerate(groups):
            labels[contact_indices[group].cpu().numpy()] = group_id + 1

        contact_pts = self.contact_set.get_posterior_points()
        return labels, contact_pts


class HardTrackingIterator:
    def __init__(self, contact_objs):
        self.contact_objs = contact_objs

    def __next__(self):
        object: tracking.ContactObject = next(self.contact_objs)
        return object.points


class OurHardTrackingMethod(OurTrackingMethod):
    def __init__(self, env, contact_params):
        self.contact_params = contact_params
        self._contact_set = tracking.ContactSetHard(self.contact_params,
                                                    contact_object_factory=self.create_contact_object)
        super(OurHardTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetHard:
        return self._contact_set

    def __iter__(self):
        return HardTrackingIterator(iter(self.contact_set))

    def create_contact_object(self):
        return tracking.ContactUKF(None, self.contact_params)


class SklearnPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, online_method, contact_detector: detection.ContactDetector, controls, nu=None):
        super().__init__(controls, nu=nu)
        self.online_method = online_method
        self.contact_detector = contact_detector
        self.in_contact = []

    def update(self, obs, info, visualizer=None):
        contact_point = self.contact_detector.get_last_contact_location(visualizer=visualizer)
        if contact_point is not None:
            self.in_contact.append(True)
            contact_point = contact_point.cpu().numpy()
            dobj = info[InfoKeys.DEE_IN_CONTACT]
            self.online_method.update(contact_point - dobj, self.u_history[-1], dobj)
        else:
            self.in_contact.append(False)


class SklearnTrackingMethod(TrackingMethod):
    def __init__(self, env, online_class, method, inertia_ratio=0.5, **kwargs):
        self.env = env
        self.online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        self.ctrl: typing.Optional[SklearnPredeterminedController] = None

    def __iter__(self):
        moved_pt_labels = process_labels_with_noise(self.online_method.final_labels())
        moved_pts = self.online_method.moved_data()
        # labels[valid] = moved_pt_labels
        groups = []
        for i, obj_id in enumerate(np.unique(moved_pt_labels)):
            if obj_id == NO_CONTACT_ID:
                continue

            indices = moved_pt_labels == obj_id
            groups.append(moved_pts[indices])

        return iter(groups)

    def create_predetermined_controller(self, controls):
        self.ctrl = SklearnPredeterminedController(self.online_method, self.env.contact_detector, controls, nu=2)
        return self.ctrl

    def visualize_contact_points(self, env):
        # valid = self.ctrl.in_contact
        # labels = np.ones(len(valid)) * NO_CONTACT_ID
        moved_pt_labels = process_labels_with_noise(self.online_method.final_labels())
        moved_pts = self.online_method.moved_data()
        # labels[valid] = moved_pt_labels
        for i, obj_id in enumerate(np.unique(moved_pt_labels)):
            if obj_id == NO_CONTACT_ID:
                continue

            indices = moved_pt_labels == obj_id
            color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
            base_name = str(i)
            self.env.visualize_state_actions(base_name, moved_pts[indices], None, color, u_color, 0)

    def get_labelled_moved_points(self, labels):
        labels[1:][self.ctrl.in_contact] = process_labels_with_noise(self.online_method.final_labels())
        moved_pts = self.online_method.moved_data()
        return labels, moved_pts


class KeyboardDirPressed():
    def __init__(self):
        self._dir = [0, 0]
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.calibrate = False
        self.esc = False

    @property
    def dir(self):
        return self._dir

    def on_press(self, key):
        if key == keyboard.Key.down:
            self.dir[1] = -1
        elif key == keyboard.Key.left:
            self.dir[0] = -1
        elif key == keyboard.Key.up:
            self.dir[1] = 1
        elif key == keyboard.Key.right:
            self.dir[0] = 1
        elif key == keyboard.Key.shift:
            self.calibrate = True
        elif key == keyboard.Key.esc:
            self.esc = True

    def on_release(self, key):
        if key in [keyboard.Key.down, keyboard.Key.up]:
            self.dir[1] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.dir[0] = 0
        elif key == keyboard.Key.shift:
            self.calibrate = False


class KeyboardController(Controller):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet, nu=2):
        super().__init__()
        self.pushed = KeyboardDirPressed()
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.nu = nu

        self.x_history = []
        self.u_history = []

    def done(self):
        return self.pushed.esc

    @abc.abstractmethod
    def update(self, obs, info):
        # obs == self.x_history[-1]
        self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                self.x_history[-1] - self.x_history[-2],
                                self.contact_detector, torch.tensor(info['reaction']), info=info)

    def command(self, obs, info=None):
        self.x_history.append(obs)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = tuple(self.pushed.dir)

        if len(self.x_history) > 1 and self.u_history[-1] != (0, 0):
            self.update(obs, info)

        self.u_history.append(u)
        return u
