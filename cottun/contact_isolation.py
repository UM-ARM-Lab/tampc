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
