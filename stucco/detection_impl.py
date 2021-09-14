import os
import logging

import numpy as np
import pybullet as p
import torch
from stucco.detection import ContactDetectorPlanar
from pytorch_kinematics import transforms as tf
from stucco import cfg
from stucco.env.pybullet_env import ContactInfo, closest_point_on_surface

logger = logging.getLogger(__name__)


class ContactDetectorPlanarPybulletGripper(ContactDetectorPlanar):
    """Leverage pybullet to sample points on the robot;
    if the sampled robot points and normals are cached then pybullet information can be omitted."""

    def __init__(self, name, *args, sample_pt_min_separation=0.005, robot_id=None,
                 canonical_orientation=None, default_joint_config=None, visualizer=None, **kwargs):
        self.name = name
        self.robot_id = robot_id
        self._canonical_orientation = canonical_orientation
        self._default_joint_config = default_joint_config

        self._sample_pt_min_separation = sample_pt_min_separation
        super().__init__(*args, **kwargs)

        self._cached_points = None
        self._cached_normals = None
        self._init_sample_surface_points_in_canonical_pose(visualizer)

    def _init_sample_surface_points_in_canonical_pose(self, visualizer=None):
        # load if possible; otherwise would require a running pybullet instance
        fullname = os.path.join(cfg.DATA_DIR, f'detection_{self.name}_cache.pkl')
        if os.path.exists(fullname):
            self._cached_points, self._cached_normals = torch.load(fullname)
            logger.info("cached robot points and normals loaded from %s", fullname)
            return

        evenly_sample = True
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(self.robot_id)
        z = orig_pos[2]

        # first reset to canonical location
        canonical_pos = [0, 0, z]
        p.resetBasePositionAndOrientation(self.robot_id, canonical_pos, self._canonical_orientation)
        for i, joint_value in enumerate(self._default_joint_config):
            p.resetJointState(self.robot_id, i, joint_value)
        self._cached_points = []
        self._cached_normals = []

        if evenly_sample:
            r = 0.115
            # sample evenly in terms of angles, but leave out the section in between the fingers
            leave_out = 0.01
            angles = np.linspace(leave_out, np.pi * 2 - leave_out, self.num_sample_points)
            for angle in angles:
                pt = [np.cos(angle) * r, np.sin(angle) * r, z]
                min_pt = closest_point_on_surface(self.robot_id, pt)
                min_pt_at_z = [min_pt[ContactInfo.POS_A][0], min_pt[ContactInfo.POS_A][1], z]
                if len(self._cached_points) > 0:
                    d = np.subtract(self._cached_points, min_pt_at_z)
                    d = np.linalg.norm(d, axis=1)
                    if np.any(d < self._sample_pt_min_separation):
                        continue
                self._cached_points.append(min_pt_at_z)
                normal = min_pt[ContactInfo.POS_B + 1]
                self._cached_normals.append([-normal[0], -normal[1], 0])
        else:
            # randomly sample
            sigma = 0.2
            while len(self._cached_points) < self.num_sample_points:
                pt = np.r_[np.random.randn(2) * sigma, z]
                min_pt = closest_point_on_surface(self.robot_id, pt)
                if min_pt[ContactInfo.DISTANCE] < 0:
                    continue
                min_pt_at_z = [min_pt[ContactInfo.POS_A][0], min_pt[ContactInfo.POS_A][1], z]
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
        x = tf.Translate(*canonical_pos, device=self.device, dtype=self.dtype)
        r = tf.Rotate(self._canonical_orientation, device=self.device, dtype=self.dtype)
        trans = x.compose(r).inverse()
        self._cached_points = trans.transform_points(torch.tensor(self._cached_points))
        self._cached_normals = trans.transform_normals(torch.tensor(self._cached_normals))

        torch.save((self._cached_points, self._cached_normals), fullname)
        logger.info("robot points and normals saved to %s", fullname)

        # p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        # if visualizer is not None:
        #     for i, min_pt_at_z in enumerate(self._cached_points):
        #         t = i / len(self._cached_points)
        #         visualizer.draw_point(f'c{t}', min_pt_at_z, color=(t, t, 1 - t))

        p.resetBasePositionAndOrientation(self.robot_id, orig_pos, orig_orientation)

    def sample_robot_surface_points(self, pose, visualizer=None):
        if self._cached_points is None:
            self._init_sample_surface_points_in_canonical_pose(visualizer)
        if self._cached_points.dtype != self.dtype or self._cached_points.device != self.device:
            self._cached_points = self._cached_points.to(device=self.device, dtype=self.dtype)
            self._cached_normals = self._cached_normals.to(device=self.device, dtype=self.dtype)

        x = tf.Translate(*pose[0], device=self.device, dtype=self.dtype)
        r = tf.Rotate(pose[1], device=self.device, dtype=self.dtype)
        link_to_current_tf = x.compose(r)
        pts = link_to_current_tf.transform_points(self._cached_points)
        normals = link_to_current_tf.transform_normals(self._cached_normals)
        if visualizer is not None:
            for i, pt in enumerate(pts):
                visualizer.draw_point(f't.{i}', pt, color=(1, 0, 0), height=pt[2])
                visualizer.draw_2d_line(f'n.{i}', pt, normals[i], color=(0.5, 0, 0), size=2., scale=0.1)

        return self._cached_points, pts, normals
