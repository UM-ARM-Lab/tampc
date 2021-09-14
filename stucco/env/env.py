import abc
import functools
import typing

import numpy as np
import torch
from arm_pytorch_utilities import load_data as load_utils, array_utils, tensor_utils
from arm_pytorch_utilities.make_data import datasource
from stucco import cfg
from stucco.detection import ContactDetector


class InfoKeys:
    OBJ_POSES = "object_poses"
    DEE_IN_CONTACT = "dee in contact"
    CONTACT_ID = "contact_id"
    # highgest frequency feedback of reaction force and torque at end effector
    HIGH_FREQ_REACTION_F = "r"
    HIGH_FREQ_REACTION_T = "t"
    HIGH_FREQ_EE_POSE = "p"  # position cat with unit quaternion orientation
    HIGH_FREQ_CONTACT_POINT = "c"


class TrajectoryLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, ignore_masks=False, **kwargs):
        self.info_desc = {}
        self.ignore_masks = ignore_masks
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

        # might want to ignore masks if we need all data points
        if not self.ignore_masks:
            xu = xu[mask]
            info = info[mask]
            y = y[mask]

        self.config.load_data_info(x, u, y, xu)
        return xu, y, info


class Visualizer:
    """Common interface for drawing environment elements"""

    @abc.abstractmethod
    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, length_ratio=1, rot=0, height=None, label=None,
                   scale=2):
        pass

    @abc.abstractmethod
    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=None):
        pass

    @abc.abstractmethod
    def draw_2d_line(self, name, start, diff, color=(0, 0, 0), size=2., scale=0.4):
        pass

    @abc.abstractmethod
    def clear_visualizations(self, names=None):
        pass


class Env:
    @property
    @abc.abstractmethod
    def nx(self):
        """Dimensionality of state space"""
        return 0

    @property
    @abc.abstractmethod
    def nu(self):
        """Dimensionality of action space"""
        return 0

    @staticmethod
    @abc.abstractmethod
    def state_names():
        """Get list of names, one for each state corresponding to the index"""
        return []

    @classmethod
    @abc.abstractmethod
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        return np.array([])

    @classmethod
    @abc.abstractmethod
    def state_distance(cls, state_difference):
        """Get a measure of distance in the state space"""
        return 0

    @classmethod
    def state_distance_two_arg(cls, state, other_state):
        return cls.state_distance(cls.state_difference(state, other_state))

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
    def control_similarity(cls, u1, u2):
        """Get similarity between 0 - 1 of two controls"""

    @classmethod
    @abc.abstractmethod
    def state_cost(cls):
        """Assuming cost function is xQx + uRu, return Q"""
        return np.diag([])

    @classmethod
    @abc.abstractmethod
    def control_cost(cls):
        """Assuming cost function is xQx + uRu, return R"""
        return np.diag([])

    @property
    @abc.abstractmethod
    def vis(self) -> Visualizer:
        """Return the visualizer used to render elements for this environment"""

    @abc.abstractmethod
    def create_contact_detector(self, residual_threshold, residual_precision) -> ContactDetector:
        """Create a contact detector for detecting and isolating contact"""

    @property
    @abc.abstractmethod
    def contact_detector(self) -> ContactDetector:
        """Get the contact detector"""

    def verify_dims(self):
        u_min, u_max = self.get_control_bounds()
        assert u_min.shape[0] == u_max.shape[0]
        assert u_min.shape[0] == self.nu
        assert len(self.state_names()) == self.nx
        assert len(self.control_names()) == self.nu
        assert self.state_cost().shape[0] == self.nx
        assert self.control_cost().shape[0] == self.nu

    def reset(self):
        """reset robot to init configuration"""
        pass

    @abc.abstractmethod
    def step(self, action):
        """Take an action step, returning new state, reward, done, and additional info in the style of gym envs"""
        state = np.array(self.nx)
        cost, done = self.evaluate_cost(state, action)
        info = None
        return state, -cost, done, info

    @abc.abstractmethod
    def evaluate_cost(self, state, action=None):
        cost = 0
        done = False
        return cost, done


class Mode:
    DIRECT = 0
    GUI = 1


def handle_data_format_for_state_diff(state_diff):
    @functools.wraps(state_diff)
    def data_format_handler(cls, state, other_state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(other_state.shape) == 1:
            other_state = other_state.reshape(1, -1)
        diff = state_diff(cls, state, other_state)
        if type(diff) is tuple:
            if torch.is_tensor(state):
                diff = torch.cat(diff, dim=1)
            else:
                diff = np.column_stack(diff)
        return diff

    return data_format_handler


class EnvDataSource(datasource.FileDataSource):
    def __init__(self, env: Env, data_dir=None, loader_args=None, **kwargs):
        if data_dir is None:
            data_dir = self._default_data_dir()
        if loader_args is None:
            loader_args = {}
        loader_class = self._loader_map(type(env))
        if not loader_class:
            raise RuntimeError("Unrecognized data source for env {}".format(env))
        loader = loader_class(**loader_args)
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
        assert isinstance(self.loader, TrajectoryLoader)
        return info[:, self.loader.info_desc[name]]

    def get_info_desc(self):
        """Get description of returned info columns in name: col slice format"""
        assert isinstance(self.loader, TrajectoryLoader)
        return self.loader.info_desc


class PlanarPointToConfig:
    """Measure distance from a point to the robot surface for the robot in some configuration"""

    def __init__(self, d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len):
        self.d_cache = d_cache
        self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
        self.cache_y_len = cache_y_len
        self.cache_resolution = cache_resolution
        # for debugging, view a = self.d_cache.reshape(len(x), len(y))

    def __call__(self, configs, pts):
        if not torch.is_tensor(self.d_cache):
            self.d_cache = tensor_utils.ensure_tensor(pts.device, pts.dtype, self.d_cache)

        M = configs.shape[0]
        N = pts.shape[0]
        # take advantage of the fact our system has no rotation to just translate points; otherwise would need full tsf
        query_pts = pts.repeat(M, 1, 1).transpose(0, 1)
        # needed transpose to allow broadcasting
        query_pts -= configs
        # flatten
        query_pts = query_pts.transpose(0, 1).view(M * N, -1)
        d = torch.ones(M * N, dtype=pts.dtype, device=pts.device)
        # eliminate points outside bounding box
        valid = (self.min_x <= query_pts[:, 0]) & (query_pts[:, 0] <= self.max_x) & \
                (self.min_y <= query_pts[:, 1]) & (query_pts[:, 1] <= self.max_y)
        # convert coordinates to indices
        query = query_pts[valid]
        x_id = torch.round((query[:, 0] - self.min_x) / self.cache_resolution).to(dtype=torch.long)
        y_id = torch.round((query[:, 1] - self.min_y) / self.cache_resolution).to(dtype=torch.long)
        idx = x_id * self.cache_y_len + y_id
        d[valid] = self.d_cache[idx]

        return d.view(M, N)
