import abc
import functools
import logging
import pybullet as p
import random
import time
import numpy as np
import torch

from datetime import datetime

from arm_pytorch_utilities import load_data as load_utils
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
    def __init__(self, mode=Mode.DIRECT, log_video=False):
        self.log_video = log_video
        self.mode = mode
        self.realtime = False
        self.sim_step_s = 1. / 240.
        self.randseed = None
        self._configure_physics_engine()

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

    @staticmethod
    @abc.abstractmethod
    def state_names():
        """Get list of names, one for each state corresponding to the index"""
        return []


class ContactInfo:
    """Semantics for indices of a contact info from getContactPoints"""
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