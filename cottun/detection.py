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
                 max_friction_cone_angle=80 * math.pi / 180):
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

    def observe_residual(self, ee_force_torque, pose=None):
        """Returns whether this residual implies we are currently in contact and track its location if given pose"""
        epsilon = ee_force_torque.T @ self.residual_precision @ ee_force_torque
        in_contact = epsilon > self.residual_threshold

        self.observation_history.append((in_contact, ee_force_torque, pose))

        return in_contact

    @abc.abstractmethod
    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        """Return contact point in link frame that most likely explains the observed residual"""
        # TODO if single pass evaluation doesn't work (e.g. from points being too sparse), try iteratively resampling

    def get_last_contact_location(self, pose, **kwargs):
        """Get last contact point given the current end effector pose"""
        if len(self.observation_history) == 0:
            return None
        in_contact, ee_force_torque, prev_pose = self.observation_history[-1]
        if not in_contact:
            return None

        # in link frame
        last_contact_point = self.isolate_contact(ee_force_torque, prev_pose, **kwargs)

        x = tf.Translate(*pose[0])
        r = tf.Rotate(pose[1])
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
    def get_jacobian(self, locations, q=None):
        """For planar robots, this kind of Jacobian is configuration independent"""
        return torch.stack([torch.tensor([[1., 0.], [0., 1.], [-loc[1], loc[0]]]) for loc in locations])

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        # 2D
        link_frame_pts, pts, normals = self.sample_robot_surface_points(pose, visualizer=visualizer)
        dtype = pts.dtype
        F_c = torch.tensor(ee_force_torque[:2], dtype=dtype)
        T_ee = torch.tensor(ee_force_torque[-1], dtype=dtype)

        # reject points where force is sufficiently away from surface normal
        from_normal = math_utils.angle_between(normals[:, :2], -F_c.view(1, -1))
        valid = from_normal < self.max_friction_cone_angle
        valid = valid.view(-1)
        pts = pts[valid]
        link_frame_pts = link_frame_pts[valid]

        # get relative to end effector origin
        rel_pts = pts - torch.tensor(pose[0], dtype=dtype)
        J = self.get_jacobian(rel_pts, q=q)
        # J_{r_c}^T F_c
        expected_residual = J @ F_c

        # the below is the case for full residual; however we can shortcut since we only need to compare torque
        # error = ee_force_torque - expected_residual
        # combined_error = linalg.batch_quadratic_product(error, self.residual_precision)

        error = expected_residual[:, -1] - T_ee
        # don't have to worry about normalization since it's just the torque dimension
        combined_error = error.abs()

        min_err_i = torch.argmin(combined_error)

        if visualizer is not None:
            visualizer.draw_point(f'most likely contact', pts[min_err_i], color=(0, 1, 0))
            # also draw some other likely points
            likely_pt_index = torch.argsort(combined_error)
            for i in range(1, 6):
                pt = pts[likely_pt_index[i]]
                visualizer.draw_point(f'likely{i}', pt, height=pt[2] + 0.001, color=(0, 1 - i / 8, 0))

        return link_frame_pts[min_err_i]
