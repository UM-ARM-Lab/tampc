import os

import numpy as np
import pybullet as p
import torch
from cottun.detection import ContactDetectorPlanar
from pytorch_kinematics import transforms as tf
from tampc import cfg
from tampc.env.pybullet_env import ContactInfo


class ContactDetectorPlanarPybulletGripper(ContactDetectorPlanar):
    """Leverage pybullet to sample points on the robot"""

    def __init__(self, robot_id, base_orientation, default_joint_config, *args, sample_pt_min_separation=0.005,
                 **kwargs):
        self.robot_id = robot_id
        self._base_orientation = base_orientation
        self._default_joint_config = default_joint_config

        self._sample_pt_min_separation = sample_pt_min_separation
        super().__init__(*args, **kwargs)

        self._cached_points = None
        self._cached_normals = None
        self._init_sample_surface_points_in_canonical_pose()

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
        self._cached_normals = []

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
                normal = pts_on_surface[0][ContactInfo.POS_B + 1]
                self._cached_normals.append([-normal[0], -normal[1], 0])
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
        trans = x.compose(r).inverse()
        self._cached_points = trans.transform_points(torch.tensor(self._cached_points))
        self._cached_normals = trans.transform_normals(torch.tensor(self._cached_normals))

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
            normals = link_to_current_tf.transform_normals(self._cached_normals)
            for i, pt in enumerate(pts):
                visualizer.draw_point(f't{i}', pt, color=(1, 0, 0), height=pt[2])
                visualizer.draw_2d_line(f'n{i}', pt, normals[i], color=(0.5, 0, 0), size=2., scale=0.1)

        return self._cached_points, pts
