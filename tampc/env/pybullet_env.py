import abc
import logging
import pybullet as p
import random
import time
import numpy as np
import enum

from datetime import datetime

from arm_pytorch_utilities import math_utils

import pybullet_data
from tampc.env.env import Mode, Env

logger = logging.getLogger(__name__)


class PybulletEnv(Env):
    def __init__(self, mode=Mode.DIRECT, log_video=False, default_debug_height=0, camera_dist=1.5):
        self.log_video = log_video
        self.mode = mode
        self.realtime = False
        self.sim_step_s = 1. / 240.
        self.randseed = None

        # quadratic cost
        self.Q = self.state_cost()
        self.R = self.control_cost()

        self._configure_physics_engine()
        self._dd = DebugDrawer(default_debug_height, camera_dist)

    def set_camera_position(self, camera_pos, yaw=0, pitch=-89):
        self._dd.set_camera_position(camera_pos, yaw, pitch)

    def _configure_physics_engine(self):
        mode_dict = {Mode.GUI: p.GUI, Mode.DIRECT: p.DIRECT}

        # if the mode we gave is in the dict then use it, otherwise use the given mode value as is
        mode = mode_dict.get(self.mode) or self.mode

        self.physics_client = p.connect(mode)  # p.GUI for GUI or p.DIRECT for non-graphical version

        # disable useless menus on the left and right
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if self.log_video:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                "{}_{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), self.randseed))

        # use data provided by PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        if self.realtime:
            p.setRealTimeSimulation(True)
        else:
            p.setRealTimeSimulation(False)
            p.setTimeStep(self.sim_step_s)

    def seed(self, randseed=None):
        random.seed(time.time())
        if randseed is None:
            randseed = random.randint(0, 1000000)
        logger.info('random seed: %d', randseed)
        self.randseed = randseed
        random.seed(randseed)
        # potentially also randomize the starting configuration

    def close(self):
        p.disconnect(self.physics_client)

    def draw_user_text(self, text, location_index=1, left_offset=1.0, xy=None):
        if xy:
            self._dd.draw_screen_text('user_{}'.format(xy), text, xy)
        else:
            if location_index is 0:
                raise RuntimeError("Can't use same location index (0) as cost")
            self._dd.draw_text('user{}_{}'.format(location_index, left_offset), text, location_index, left_offset)

    @abc.abstractmethod
    def _draw_action(self, action, old_state=None, debug=0):
        pass

    @abc.abstractmethod
    def visualize_goal_set(self, states):
        pass

    @abc.abstractmethod
    def visualize_trap_set(self, states):
        pass

    @abc.abstractmethod
    def visualize_contact_set(self, contact_set):
        pass

    @abc.abstractmethod
    def visualize_rollouts(self, states):
        pass

    @abc.abstractmethod
    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""

    @staticmethod
    def _make_robot_translucent(robot_id, alpha=0.4):
        visual_data = p.getVisualShapeData(robot_id)
        for link in visual_data:
            link_id = link[1]
            rgba = list(link[7])
            rgba[3] = alpha
            p.changeVisualShape(robot_id, link_id, rgbaColor=rgba)


class ContactInfo(enum.IntEnum):
    """Semantics for indices of a contact info from getContactPoints"""
    LINK_A = 3
    LINK_B = 4
    POS_A = 5
    NORMAL_DIR_B = 7
    NORMAL_MAG = 9
    LATERAL1_MAG = 10
    LATERAL1_DIR = 11
    LATERAL2_MAG = 12
    LATERAL2_DIR = 13


def get_total_contact_force(contact, flip=True):
    force_sign = -1 if flip else 1
    force = force_sign * contact[ContactInfo.NORMAL_MAG]
    dv = [force * v for v in contact[ContactInfo.NORMAL_DIR_B]]
    fyd, fxd = get_lateral_friction_forces(contact, flip)
    f_all = [sum(i) for i in zip(dv, fyd, fxd)]
    return f_all


def get_lateral_friction_forces(contact, flip=True):
    force_sign = -1 if flip else 1
    fy = force_sign * contact[ContactInfo.LATERAL1_MAG]
    fyd = [fy * v for v in contact[ContactInfo.LATERAL1_DIR]]
    fx = force_sign * contact[ContactInfo.LATERAL2_MAG]
    fxd = [fx * v for v in contact[ContactInfo.LATERAL2_DIR]]
    return fyd, fxd


class DebugDrawer:
    def __init__(self, default_height, camera_height):
        self._debug_ids = {}
        self._camera_pos = None
        self._camera_height = camera_height
        self._default_height = default_height
        self._3dmode = False
        self._inv_camera_tsf = None
        self.set_camera_position([0, 0])

    def set_camera_position(self, camera_pos, yaw=0, pitch=-89):
        self._camera_pos = camera_pos
        p.resetDebugVisualizerCamera(cameraDistance=self._camera_height, cameraYaw=yaw, cameraPitch=pitch,
                                     cameraTargetPosition=[camera_pos[0], camera_pos[1], 0])
        # wait for reset
        for _ in range(1000):
            p.stepSimulation()
        # cache the inverse camera transform for efficiency
        info = p.getDebugVisualizerCamera()
        if info[0] == 0 and info[1] == 0:
            logger.warning("Setting empty camera; check that we are not in GUI mode")
        else:
            view_matrix = np.array(info[2]).reshape(4, 4).T
            self._inv_camera_tsf = np.linalg.inv(view_matrix)

    def toggle_3d(self, using_3d):
        self._3dmode = using_3d

    def _process_point_height(self, point, height):
        if height is None:
            if self._3dmode:
                height = point[2]
            else:
                height = self._default_height
        return height

    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, height=None):
        if name not in self._debug_ids:
            self._debug_ids[name] = [-1, -1]
        uids = self._debug_ids[name]

        # ignore 3rd dimension if it exists to plot everything at the same height
        height = self._process_point_height(point, height)

        location = (point[0], point[1], height)
        uids[0] = p.addUserDebugLine(np.add(location, [length, 0, 0]), np.add(location, [-length, 0, 0]), color, 2,
                                     replaceItemUniqueId=uids[0])
        uids[1] = p.addUserDebugLine(np.add(location, [0, length, 0]), np.add(location, [0, -length, 0]), color, 2,
                                     replaceItemUniqueId=uids[1])

    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=None):
        height = self._process_point_height(pose, height)
        if name not in self._debug_ids:
            self._debug_ids[name] = [-1, -1]
        uids = self._debug_ids[name]

        location = (pose[0], pose[1], height)
        side_lines = math_utils.rotate_wrt_origin((0, length * 0.2), pose[2])
        pointer = math_utils.rotate_wrt_origin((length, 0), pose[2])
        uids[0] = p.addUserDebugLine(np.add(location, [side_lines[0], side_lines[1], 0]),
                                     np.add(location, [-side_lines[0], -side_lines[1], 0]),
                                     color, 2, replaceItemUniqueId=uids[0])
        uids[1] = p.addUserDebugLine(np.add(location, [0, 0, 0]),
                                     np.add(location, [pointer[0], pointer[1], 0]),
                                     color, 2, replaceItemUniqueId=uids[1])

    def clear_visualization_after(self, prefix, index):
        name = "{}{}".format(prefix, index)
        while name in self._debug_ids:
            uids = self._debug_ids.pop(name)
            if type(uids) is int:
                uids = [uids]
            for id in uids:
                p.removeUserDebugItem(id)
            index += 1
            name = "{}{}".format(prefix, index)

    def draw_2d_line(self, name, start, diff, color=(0, 0, 0), size=2., scale=0.4):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        self._debug_ids[name] = p.addUserDebugLine(start, np.add(start, [diff[0] * scale, diff[1] * scale,
                                                                         diff[2] * scale if len(diff) is 3 else 0]),
                                                   color, lineWidth=size, replaceItemUniqueId=uid)

    def draw_contact_point(self, name, contact, flip=True):
        start = contact[ContactInfo.POS_A]
        f_all = get_total_contact_force(contact, flip)
        # combined normal vector (adding lateral friction)
        f_size = np.linalg.norm(f_all)
        self.draw_2d_line("{} xy".format(name), start, f_all, size=f_size, scale=0.03, color=(1, 1, 0))
        # _draw_contact_friction(line_unique_ids, contact, flip)
        return f_size

    def draw_contact_friction(self, name, contact, flip=True, height=None):
        start = list(contact[ContactInfo.POS_A])
        start[2] = self._process_point_height(start, height)
        # friction along y
        scale = 0.1
        c = (1, 0.4, 0.7)
        fyd, fxd = get_lateral_friction_forces(contact, flip)
        self.draw_2d_line('{}y'.format(name), start, fyd, size=np.linalg.norm(fyd), scale=scale, color=c)
        self.draw_2d_line('{}x'.format(name), start, fxd, size=np.linalg.norm(fxd), scale=scale, color=c)

    def draw_transition(self, prev_block, new_block, height=None):
        name = 't'
        if name not in self._debug_ids:
            self._debug_ids[name] = []

        self._debug_ids[name].append(
            p.addUserDebugLine([prev_block[0], prev_block[1], self._process_point_height(prev_block, height)],
                               (new_block[0], new_block[1], self._process_point_height(new_block, height)),
                               [0, 0, 1], 2))

    def clear_transitions(self):
        name = 't'
        if name in self._debug_ids:
            for line in self._debug_ids[name]:
                p.removeUserDebugItem(line)
            self._debug_ids[name] = []

    def draw_text(self, name, text, location_index, left_offset=1., offset_in_z=False):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        z = 0.1
        move_down = location_index * 0.15
        if offset_in_z:
            move_down = 0
            z += location_index * 0.1

        height_scale = self._camera_height * 0.7
        self._debug_ids[name] = p.addUserDebugText(str(text),
                                                   [self._camera_pos[0] + left_offset * height_scale,
                                                    self._camera_pos[1] + (1 - move_down) * height_scale, z],
                                                   textColorRGB=[0.5, 0.1, 0.1],
                                                   textSize=2,
                                                   replaceItemUniqueId=uid)

    def draw_screen_text(self, name, text, camera_frame_pos):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        # convert from camera frame to world frame
        pos_in = np.r_[camera_frame_pos, 1]
        world_frame_pos = self._inv_camera_tsf @ pos_in

        self._debug_ids[name] = p.addUserDebugText(str(text),
                                                   world_frame_pos[:3],
                                                   textColorRGB=[0.5, 0.1, 0.1],
                                                   textSize=2,
                                                   replaceItemUniqueId=uid)
