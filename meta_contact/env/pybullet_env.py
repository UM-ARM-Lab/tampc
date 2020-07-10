import abc
import functools
import logging
import pybullet as p
import random
import time
import numpy as np
import torch
import typing
import enum

from datetime import datetime

from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities import load_data as load_utils, math_utils
from arm_pytorch_utilities import array_utils
from meta_contact import cfg

import pybullet_data

logger = logging.getLogger(__name__)


class PybulletLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, **kwargs):
        self.info_desc = {}
        super().__init__(file_cfg, *args, **kwargs)

    @staticmethod
    @abc.abstractmethod
    def _info_names():
        return []

    def _apply_masks(self, d, x, y):
        """Handle common logic regardless of x and y"""
        info_index_offset = 0
        info = []
        for name in self._info_names():
            if name in d:
                info.append(d[name][1:])
                dim = info[-1].shape[1]
                self.info_desc[name] = slice(info_index_offset, info_index_offset + dim)
                info_index_offset += dim

        mask = d['mask']
        # add information about env/groups of data (different simulation runs are contiguous blocks)
        groups = array_utils.discrete_array_to_value_ranges(mask)
        envs = np.zeros(mask.shape[0])
        current_env = 0
        for v, start, end in groups:
            if v == 0:
                continue
            envs[start:end + 1] = current_env
            current_env += 1
        # throw away first element as always
        envs = envs[1:]
        info.append(envs)
        self.info_desc['envs'] = slice(info_index_offset, info_index_offset + 1)
        info = np.column_stack(info)

        u = d['U'][:-1]
        # potentially many trajectories, get rid of buffer state in between

        x = x[:-1]
        xu = np.column_stack((x, u))

        # pack expanded pxu into input if config allows (has to be done before masks)
        # otherwise would use cross-file data)
        if self.config.expanded_input:
            # move y down 1 row (first element can't be used)
            # (xu, pxu)
            xu = np.column_stack((xu[1:], xu[:-1]))
            y = y[1:]
            info = info[1:]

            mask = mask[1:-1]
        else:
            mask = mask[:-1]

        mask = mask.reshape(-1) != 0

        xu = xu[mask]
        info = info[mask]
        y = y[mask]

        self.config.load_data_info(x, u, y, xu)
        return xu, y, info


class Mode:
    DIRECT = 0
    GUI = 1


class PybulletEnv:
    @property
    @abc.abstractmethod
    def nx(self):
        return 0

    @property
    @abc.abstractmethod
    def nu(self):
        return 0

    @staticmethod
    @abc.abstractmethod
    def state_names():
        """Get list of names, one for each state corresponding to the index"""
        return []

    @staticmethod
    @abc.abstractmethod
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        return np.array([])

    @staticmethod
    @abc.abstractmethod
    def control_names():
        return []

    @staticmethod
    @abc.abstractmethod
    def get_control_bounds():
        """Get lower and upper bounds for control"""
        return np.array([]), np.array([])

    @classmethod
    @abc.abstractmethod
    def state_cost(cls):
        return np.diag([])

    @classmethod
    @abc.abstractmethod
    def control_cost(cls):
        return np.diag([])

    def __init__(self, mode=Mode.DIRECT, log_video=False, default_debug_height=0, camera_dist=1.5):
        self.log_video = log_video
        self.mode = mode
        self.realtime = False
        self.sim_step_s = 1. / 240.
        self.randseed = None
        self.camera_dist = camera_dist

        # quadratic cost
        self.Q = self.state_cost()
        self.R = self.control_cost()

        self._dd = DebugDrawer(default_debug_height, camera_dist)

        self._configure_physics_engine()

    def set_camera_position(self, camera_pos):
        self._dd._camera_pos = camera_pos
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_dist, cameraYaw=0, cameraPitch=-89,
                                     cameraTargetPosition=[camera_pos[0], camera_pos[1], 0])

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

    def verify_dims(self):
        u_min, u_max = self.get_control_bounds()
        assert u_min.shape[0] == u_max.shape[0]
        assert u_min.shape[0] == self.nu
        assert len(self.state_names()) == self.nx
        assert len(self.control_names()) == self.nu
        assert self.Q.shape[0] == self.nx
        assert self.R.shape[0] == self.nu

    def draw_user_text(self, text, location_index=1, left_offset=1.0):
        if location_index is 0:
            raise RuntimeError("Can't use same location index (0) as cost")
        self._dd.draw_text('user{}_{}'.format(location_index, left_offset), text, location_index, left_offset)

    def reset(self):
        """reset robot to init configuration"""
        pass

    @abc.abstractmethod
    def step(self, action):
        state = np.array(self.nx)
        cost, done = self.evaluate_cost(state, action)
        info = None
        return state, -cost, done, info

    @abc.abstractmethod
    def evaluate_cost(self, state, action=None):
        cost = 0
        done = False
        return cost, done

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
    def visualize_rollouts(self, states):
        pass

    @abc.abstractmethod
    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""


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


def handle_data_format_for_state_diff(state_diff):
    @functools.wraps(state_diff)
    def data_format_handler(state, other_state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(other_state.shape) == 1:
            other_state = other_state.reshape(1, -1)
        diff = state_diff(state, other_state)
        if torch.is_tensor(state):
            diff = torch.cat(diff, dim=1)
        else:
            diff = np.column_stack(diff)
        return diff

    return data_format_handler


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
        self._camera_pos = [0, 0]
        self._camera_height = camera_height
        self._default_height = default_height

    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, height=None):
        if height is None:
            height = self._default_height
        if name not in self._debug_ids:
            self._debug_ids[name] = [-1, -1]
        uids = self._debug_ids[name]

        # use default height if point is 2D, otherwise use point's z coordinate
        if point.shape[0] == 3:
            height = point[2]

        location = (point[0], point[1], height)
        uids[0] = p.addUserDebugLine(np.add(location, [length, 0, 0]), np.add(location, [-length, 0, 0]), color, 2,
                                     replaceItemUniqueId=uids[0])
        uids[1] = p.addUserDebugLine(np.add(location, [0, length, 0]), np.add(location, [0, -length, 0]), color, 2,
                                     replaceItemUniqueId=uids[1])

    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=None):
        if height is None:
            height = self._default_height
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

    def clear_point_after(self, prefix, index):
        self.clear_2d_poses_after(prefix, index)

    def clear_2d_poses_after(self, prefix, index):
        name = "{}{}".format(prefix, index)
        while name in self._debug_ids:
            uids = self._debug_ids.pop(name)
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
        if height is None:
            height = self._default_height
        start = list(contact[ContactInfo.POS_A])
        start[2] = height
        # friction along y
        scale = 0.1
        c = (1, 0.4, 0.7)
        fyd, fxd = get_lateral_friction_forces(contact, flip)
        self.draw_2d_line('{}y'.format(name), start, fyd, size=np.linalg.norm(fyd), scale=scale, color=c)
        self.draw_2d_line('{}x'.format(name), start, fxd, size=np.linalg.norm(fxd), scale=scale, color=c)

    def draw_transition(self, prev_block, new_block, height=None):
        if height is None:
            height = self._default_height
        name = 't'
        if name not in self._debug_ids:
            self._debug_ids[name] = []

        self._debug_ids[name].append(
            p.addUserDebugLine([prev_block[0], prev_block[1], height],
                               (new_block[0], new_block[1], height),
                               [0, 0, 1], 2))

    def clear_transitions(self):
        name = 't'
        if name in self._debug_ids:
            for line in self._debug_ids[name]:
                p.removeUserDebugItem(line)
            self._debug_ids[name] = []

    def draw_text(self, name, text, location_index, left_offset=1.):
        if name not in self._debug_ids:
            self._debug_ids[name] = -1
        uid = self._debug_ids[name]

        move_down = location_index * 0.15
        height_scale = self._camera_height * 0.7
        self._debug_ids[name] = p.addUserDebugText(str(text),
                                                   [self._camera_pos[0] + left_offset * height_scale,
                                                    self._camera_pos[1] + (1 - move_down) * height_scale, 0.1],
                                                   textColorRGB=[0.5, 0.1, 0.1],
                                                   textSize=2,
                                                   replaceItemUniqueId=uid)


class PybulletEnvDataSource(datasource.FileDataSource):
    def __init__(self, env, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = self._default_data_dir()
        loader_class = self._loader_map(type(env))
        if not loader_class:
            raise RuntimeError("Unrecognized data source for env {}".format(env))
        loader = loader_class()
        super().__init__(loader, data_dir, **kwargs)

    @staticmethod
    @abc.abstractmethod
    def _default_data_dir():
        return ""

    @staticmethod
    @abc.abstractmethod
    def _loader_map(env_type) -> typing.Union[typing.Callable, None]:
        return None

    def get_info_cols(self, info, name):
        """Get the info columns corresponding to this name"""
        return info[:, self.loader.info_desc[name]]

    def get_info_desc(self):
        """Get description of returned info columns in name: col slice format"""
        return self.loader.info_desc
