import torch
import math
import abc
import typing
from collections import deque

import pytorch_kinematics.transforms as tf
from arm_pytorch_utilities import math_utils

point = torch.tensor
points = torch.tensor
normals = torch.tensor


class ContactDetector:
    """Detect and isolate contacts given some form of generalized momentum residual measurements, see
    https://ieeexplore.ieee.org/document/7759743 (Localizing external contact using proprioceptive sensors).

    We additionally assume access to force torque sensors at the end effector, which is our residual."""

    def __init__(self, residual_precision, residual_threshold, num_sample_points=100,
                 max_friction_cone_angle=50 * math.pi / 180, window_size=5, dtype=torch.float, device='cpu'):
        """

        :param residual_precision: sigma_meas^-1 matrix that scales the different residual dimensions based on their
        expected precision
        :param residual_threshold: contact threshold for residual^T sigma_meas^-1 residual to count as being in contact
        :param max_friction_cone_angle: max angular difference (radian) from surface normal that a contact is possible
        note that 90 degrees is the weakest assumption that the force is from a push and not a pull
        """
        self.residual_precision = residual_precision
        self.residual_threshold = residual_threshold
        self.num_sample_points = num_sample_points
        self.max_friction_cone_angle = max_friction_cone_angle
        self.observation_history = deque(maxlen=500)
        self._window_size = window_size
        self.dtype = dtype
        self.device = device

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

    def observe_residual(self, ee_force_torque, pose=None):
        """Returns whether this residual implies we are currently in contact and track its location if given pose"""
        epsilon = ee_force_torque.T @ self.residual_precision @ ee_force_torque
        in_contact = epsilon > self.residual_threshold

        self.observation_history.append((in_contact, ee_force_torque, pose))

        return in_contact

    def __len__(self):
        return len(self.observation_history)

    def clear(self):
        self.observation_history.clear()

    @abc.abstractmethod
    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        """Return contact point in link frame that most likely explains the observed residual"""

    def in_contact(self):
        """Whether our last observed residual indicates that we are currently in contact"""
        if len(self.observation_history) == 0:
            return False
        for i in range(self._window_size):
            in_contact, ee_force_torque, prev_pose = self.observation_history[-1 - i]
            if in_contact:
                return True
        return False

    def get_last_contact_location(self, pose=None, **kwargs):
        """Get last contact point given the current end effector pose"""
        if len(self.observation_history) == 0:
            return None

        # allow for being in contact anytime within the latest window
        start_i = -1
        for i in range(0, min(len(self.observation_history), self._window_size)):
            in_contact, ee_force_torque, prev_pose = self.observation_history[-1 - i]
            if in_contact:
                if pose is None:
                    pose = prev_pose
                break
            start_i -= 1
        else:
            return None

        # use history of points to handle jitter
        ee_ft = [ee_force_torque]
        pp = [prev_pose]
        for i in range(2, self._window_size):
            from_end_i = start_i - i + 1
            if -from_end_i > len(self.observation_history):
                break
            in_contact, ee_force_torque, prev_pose = self.observation_history[from_end_i]
            # look back until we leave contact
            if not in_contact:
                break
            ee_ft.append(ee_force_torque)
            pp.append(prev_pose)
        ee_ft = torch.tensor(ee_ft, dtype=self.dtype, device=self.device)
        # assume we don't move that much in a short amount of time and we can just use the latest pose
        pp = tuple(torch.tensor(p, dtype=self.dtype, device=self.device) for p in pp[0])
        # pos = torch.tensor([p[0] for p in pp], dtype=self.dtype)
        # orientation = torch.tensor([p[1] for p in pp], dtype=self.dtype)
        # pp = (pos, orientation)

        # in link frame
        last_contact_point = self.isolate_contact(ee_ft, pp, **kwargs)
        if last_contact_point is None:
            return None

        x = tf.Translate(*pose[0], dtype=self.dtype, device=self.device)
        r = tf.Rotate(pose[1], dtype=self.dtype, device=self.device)
        link_to_current_tf = x.compose(r)
        return link_to_current_tf.transform_points(last_contact_point.view(1, -1))[0]

    @abc.abstractmethod
    def get_jacobian(self, locations, q=None):
        """Get J^T in the equation: wrench at end effector = J^T * wrench at contact point.
        In general the Jacobian is dependent on configuration.

        Locations are specified wrt the end effector frame."""

    @abc.abstractmethod
    def sample_robot_surface_points(self, pose, visualizer=None) -> typing.Tuple[points, points, normals]:
        """Get points on the surface of the robot that could be possible contact locations
        pose[0] and pose[1] are the position and orientation (quaternion) of the end effector, respectively.
        Also return the correspnoding surface normals for each of the points.
        """


class ContactDetectorPlanar(ContactDetector):
    """Contact detector for planar environments without rotation"""

    def get_jacobian(self, locations, q=None):
        """For planar robots, this kind of Jacobian is configuration independent"""
        return torch.stack(
            [torch.tensor([[1., 0.], [0., 1.], [-loc[1], loc[0]]], device=self.device, dtype=self.dtype) for loc in
             locations])

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        # 2D
        link_frame_pts, pts, normals = self.sample_robot_surface_points(pose, visualizer=visualizer)
        F_c = ee_force_torque[:, :2]
        T_ee = ee_force_torque[:, -1]

        while True:
            # reject points where force is sufficiently away from surface normal
            from_normal = math_utils.angle_between(normals[:, :2], -F_c)
            valid = from_normal < self.max_friction_cone_angle
            # validity has to hold across all experienced forces
            valid = valid.all(dim=1)
            # remove a single element of the constraint if we don't have any satisfying them
            if valid.any():
                break
            else:
                F_c = F_c[:-1]
                T_ee = T_ee[:-1]

        # no valid point
        if F_c.numel() == 0:
            return None

        pts = pts[valid]
        link_frame_pts = link_frame_pts[valid]

        # NOTE: if our system is able to rotate, would have to transform points by rotation too
        # get relative to end effector origin
        rel_pts = pts - pose[0]
        J = self.get_jacobian(rel_pts, q=q)
        # J_{r_c}^T F_c
        expected_residual = J @ F_c.transpose(-1, -2)

        # the below is the case for full residual; however we can shortcut since we only need to compare torque
        # error = ee_force_torque - expected_residual
        # combined_error = linalg.batch_quadratic_product(error, self.residual_precision)

        error = expected_residual[:, -1] - T_ee
        # don't have to worry about normalization since it's just the torque dimension
        combined_error = error.abs().sum(dim=1)

        min_err_i = torch.argmin(combined_error)

        if visualizer is not None:
            # also draw some other likely points
            likely_pt_index = torch.argsort(combined_error)
            for i in reversed(range(1, min(10, len(pts)))):
                pt = pts[likely_pt_index[i]]
                visualizer.draw_point(f'likely.{i}', pt, height=pt[2] + 0.001,
                                      color=(0, 1 - 0.8 * i / 10, 0))
            visualizer.draw_point(f'most likely contact', pts[min_err_i], color=(0, 1, 0), scale=2)
            visualizer.draw_2d_line('reaction', pts[min_err_i], ee_force_torque.mean(dim=0)[:3], color=(0, 0.2, 1.0),
                                    scale=0.2)

        return link_frame_pts[min_err_i]
