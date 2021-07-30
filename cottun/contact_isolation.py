import numpy as np
import torch
import abc
import pybullet as p
import os

from tampc import cfg
from tampc.env.pybullet_env import ContactInfo
import pytorch_kinematics.transforms as tf


class ContactDetectorPlanar:
    """Detect and isolate contacts given some form of generalized momentum residual measurements, see
    https://ieeexplore.ieee.org/document/7759743 (Localizing external contact using proprioceptive sensors).

    We additionally assume access to force torque sensors at the end effector, which is our residual."""

    def __init__(self, residual_precision, residual_threshold, num_sample_points=100):
        """

        :param residual_precision: sigma_meas^-1 matrix that scales the different residual dimensions based on their
        expected precision
        :param residual_threshold: contact threshold for residual^T sigma_meas^-1 residual to count as being in contact
        """
        self.residual_precision = residual_precision
        self.residual_threshold = residual_threshold
        self.num_sample_points = num_sample_points

    def observe_residual(self, ee_force_torque):
        """Returns whether this residual implies we are currently in contact"""
        epsilon = ee_force_torque.T @ self.residual_precision @ ee_force_torque
        return epsilon > self.residual_threshold

    def get_jacobian(self, locations, q=None):
        """Get J^T in the equation: wrench at end effector = J^T * wrench at contact point.
        This kind of Jacobian is configuration independent,
        but in general the Jacobian is dependent on configuration.

        Locations are specified wrt the end effector frame."""
        return [np.array([[1., 0.], [0., 1.], [-loc[1], loc[0]]]) for loc in locations]

    @abc.abstractmethod
    def sample_robot_surface_points(self, pose, visualizer=None):
        """Get points on the surface of the robot that could be possible contact locations
        pose[0] and pose[1] are the position and orientation (quaternion) of the end effector, respectively.
        """


class ContactDetectorPlanarPybulletGripper(ContactDetectorPlanar):
    """Leverage pybullet to sample points on the robot"""

    def __init__(self, robot_id, base_orientation, default_joint_config, *args, sample_pt_min_separation=0.005,
                 **kwargs):
        self.robot_id = robot_id
        self._base_orientation = base_orientation
        self._default_joint_config = default_joint_config

        self._sample_pt_min_separation = sample_pt_min_separation
        self._cached_points = None
        super().__init__(*args, **kwargs)

    def _init_sample_surface_points_in_canonical_pose(self, visualizer=None):
        evenly_sample = True
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(self.robot_id)
        z = orig_pos[2]

        # first reset to canonical location
        canonical_pos = [0, 0, z]
        p.resetBasePositionAndOrientation(self.robot_id, canonical_pos, self._base_orientation)
        for i, joint_value in enumerate(self._default_joint_config):
            p.resetJointState(self.robot_id, i, joint_value)
        self._cached_points = []

        # initialize tester
        tester_id = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=True,
                               basePosition=canonical_pos, globalScaling=0.001)

        if evenly_sample:
            r = 0.115
            # sample evenly in terms of angles, but leave out the section in between the fingers
            leave_out = 0.01
            angles = np.linspace(leave_out, np.pi * 2 - leave_out, self.num_sample_points)
            for angle in angles:
                pt = [np.cos(angle) * r, np.sin(angle) * r, z]
                p.resetBasePositionAndOrientation(tester_id, pt, [0, 0, 0, 1])
                p.performCollisionDetection()
                pts_on_surface = p.getClosestPoints(self.robot_id, tester_id, 100, linkIndexB=-1)
                pts_on_surface = sorted(pts_on_surface, key=lambda c: c[ContactInfo.DISTANCE])
                min_pt = pts_on_surface[0][ContactInfo.POS_A]
                min_pt_at_z = [min_pt[0], min_pt[1], z]
                if len(self._cached_points) > 0:
                    d = np.subtract(self._cached_points, min_pt_at_z)
                    d = np.linalg.norm(d, axis=1)
                    if np.any(d < self._sample_pt_min_separation):
                        continue
                self._cached_points.append(min_pt_at_z)
        else:
            # randomly sample
            sigma = 0.2
            while len(self._cached_points) < self.num_sample_points:
                pt = np.r_[np.random.randn(2) * sigma, z]
                # sample an object at random points around this object and find closest point to it
                p.resetBasePositionAndOrientation(tester_id, pt, [0, 0, 0, 1])
                p.performCollisionDetection()
                pts_on_surface = p.getClosestPoints(self.robot_id, tester_id, 100, linkIndexB=-1)
                # don't want penetration since that could lead to closest being along z instead of x-y
                pts_on_surface = [pt for pt in pts_on_surface if pt[ContactInfo.DISTANCE] > 0]
                if not len(pts_on_surface):
                    continue
                pts_on_surface = sorted(pts_on_surface, key=lambda c: c[ContactInfo.DISTANCE])
                min_pt = pts_on_surface[0][ContactInfo.POS_A]
                min_pt_at_z = [min_pt[0], min_pt[1], z]
                if len(self._cached_points) > 0:
                    d = np.subtract(self._cached_points, min_pt_at_z)
                    d = np.linalg.norm(d, axis=1)
                    if np.any(d < self._sample_pt_min_separation):
                        continue
                self._cached_points.append(min_pt_at_z)

        if visualizer is not None:
            for i, min_pt_at_z in enumerate(self._cached_points):
                t = i / len(self._cached_points)
                visualizer.draw_point(f'c{t}', min_pt_at_z, color=(t, t, 1 - t))

        # convert points back to link frame
        x = tf.Translate(*canonical_pos)
        r = tf.Rotate(self._base_orientation)
        trans = x.compose(r)
        self._cached_points = trans.inverse().transform_points(torch.tensor(self._cached_points))

        # p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        # if visualizer is not None:
        #     for i, min_pt_at_z in enumerate(self._cached_points):
        #         t = i / len(self._cached_points)
        #         visualizer.draw_point(f'c{t}', min_pt_at_z, color=(t, t, 1 - t))

        p.resetBasePositionAndOrientation(self.robot_id, orig_pos, orig_orientation)
        p.removeBody(tester_id)

    def sample_robot_surface_points(self, pose, visualizer=None):
        if self._cached_points is None:
            self._init_sample_surface_points_in_canonical_pose()

        x = tf.Translate(*pose[0])
        r = tf.Rotate(pose[1])
        link_to_current_tf = x.compose(r)
        pts = link_to_current_tf.transform_points(self._cached_points)
        if visualizer is not None:
            for i, pt in enumerate(pts):
                visualizer.draw_point(f't{i}', pt, color=(1, 0, 0), height=pt[2])

        return pts
