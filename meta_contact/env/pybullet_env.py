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
import matplotlib.pyplot as plt

from datetime import datetime

from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities import load_data as load_utils, math_utils
from arm_pytorch_utilities import array_utils
from meta_contact import cfg
from arm_pytorch_utilities import simulation
from meta_contact.controller import controller, online_controller
from meta_contact import cost as control_cost

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

    def __init__(self, mode=Mode.DIRECT, log_video=False, default_debug_height=0):
        self.log_video = log_video
        self.mode = mode
        self.realtime = False
        self.sim_step_s = 1. / 240.
        self.randseed = None

        # quadratic cost
        self.Q = self.state_cost()
        self.R = self.control_cost()

        self._dd = DebugDrawer(default_debug_height)

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
    def __init__(self, default_height):
        self._debug_ids = {}
        self._camera_pos = [0, 0]
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
        self._debug_ids[name] = p.addUserDebugText(str(text),
                                                   [self._camera_pos[0] + left_offset,
                                                    self._camera_pos[1] + 1 - move_down, 0.1],
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


class PybulletSim(simulation.Simulation):
    def __init__(self, env: PybulletEnv, controller, num_frames=1000, save_dir="base",
                 terminal_cost_multiplier=1, stop_when_done=True, visualize_rollouts=True,
                 visualize_action_sample=False,
                 **kwargs):

        super().__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        env.verify_dims()
        self.mode = env.mode
        self.stop_when_done = stop_when_done
        self.visualize_rollouts = visualize_rollouts
        self.visualize_action_sample = visualize_action_sample

        self.env = env
        self.ctrl = controller

        # keep track of last run's rewards
        self.terminal_cost_multiplier = terminal_cost_multiplier
        self.last_run_cost = []

        # plotting
        self.fig = None
        self.axes = None
        self.fu = None
        self.au = None
        self.fd = None
        self.ad = None

        self.fu_sample = None
        self.au_sample = None

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, self.env.nx))
        self.pred_traj = np.zeros_like(self.traj)
        self.u = np.zeros((self.num_frames, self.env.nu))
        self.reaction_force = np.zeros((self.num_frames, 2))
        self.wall_contact = np.zeros((self.num_frames,))
        self.model_error = np.zeros_like(self.traj)
        self.time = np.arange(0, self.num_frames * self.sim_step_s, self.sim_step_s)
        self.pred_cls = np.zeros_like(self.wall_contact)
        return simulation.ReturnMeaning.SUCCESS

    def _truncate_data(self, frame):
        self.traj, self.u, self.reaction_force, self.wall_contact, self.model_error, self.time, self.pred_cls = (
            data[:frame] for data
            in
            (self.traj, self.u,
             self.reaction_force,
             self.wall_contact,
             self.model_error,
             self.time, self.pred_cls))

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _run_experiment(self):
        self.last_run_cost = []
        obs = self._reset_sim()
        info = None

        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            self.env.draw_user_text("{}".format(simTime), 1)

            start = time.perf_counter()

            action = self.ctrl.command(obs, info)

            # visualizations before taking action
            if self._predicts_dynamics_cls():
                self.pred_cls[simTime] = self.ctrl.dynamics_class
                self.env.draw_user_text("dyn cls {}".format(self.ctrl.dynamics_class), 2)

                mode_text = "recovery" if self.ctrl.autonomous_recovery_mode else (
                    "local" if self.ctrl.using_local_model_for_nonnominal_dynamics else "")
                self.env.draw_user_text(mode_text, 3)
                if self.ctrl.recovery_cost and isinstance(self.ctrl.recovery_cost, control_cost.CostQRSet):
                    # plot goal set
                    self.env.visualize_goal_set(self.ctrl.recovery_cost.goal_set)

                # TODO change
                for i in range(4):
                    break
                    # dynamics_class_pred = self.ctrl.dynamics_class_prediction[i]
                    # nom_count = (dynamics_class_pred == DynamicsClass.NOMINAL).sum()
                    # text = "nom: {:.2f}".format(nom_count.float() / len(dynamics_class_pred))
                    # self.env.draw_user_text("t={} {}".format(i, text), 4 + i)

            if self.visualize_action_sample and isinstance(self.ctrl, controller.MPPI_MPC):
                self._plot_action_sample(self.ctrl.mpc.perturbed_action)
            if self.visualize_rollouts:
                self.env.visualize_rollouts(self.ctrl.get_rollouts(obs))

            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.debug("cost %-5.2f took %.3fs done %d action %-12s obs %s", cost, time.perf_counter() - start, done,
                         np.round(action, 2), np.round(obs, 3))

            self.last_run_cost.append(cost)
            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs
            # reaction force felt as we apply this action, as observed at the start of the next time step
            self.reaction_force[simTime + 1, :] = info['reaction']
            self.wall_contact[simTime + 1] = info['wall_contact']
            if self._predicts_state():
                self.pred_traj[simTime + 1, :] = self.ctrl.predicted_next_state
                # model error from the previous prediction step (can only evaluate it at the current step)
                self.model_error[simTime, :] = self.ctrl.prediction_error(obs)
                self.env.visualize_prediction_error(self.ctrl.predicted_next_state.reshape(-1))

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                self._truncate_data(simTime + 2)
                break

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        assert len(self.last_run_cost) == self.u.shape[0]

        return simulation.ReturnMeaning.SUCCESS

    def _export_data_dict(self):
        # output (1 step prediction; only need block state)
        X = self.traj
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        # need to also throw out first step if predicting reaction force since there's no previous state
        mask[0] = 0
        mask[-1] = 0
        u_norm = np.linalg.norm(self.u, axis=1)
        # shift by 1 since the control at t-1 affects the model error at t
        u_norm = np.roll(u_norm, 1).reshape(-1, 1)
        scaled_model_error = np.divide(self.model_error, u_norm, out=np.zeros_like(self.model_error), where=u_norm != 0)
        return {'X': X, 'U': self.u, 'reaction': self.reaction_force, 'model error': self.model_error,
                'scaled model error': scaled_model_error, 'wall contact': self.wall_contact.reshape(-1, 1),
                'mask': mask.reshape(-1, 1), 'predicted dynamics_class': self.pred_cls.reshape(-1, 1)}

    def _start_plot_action_sample(self):
        self.fu_sample, self.au_sample = plt.subplots(self.env.nu, 1)
        u_min, u_max = self.env.get_control_bounds()
        u_names = self.env.control_names()
        for i, name in enumerate(u_names):
            self.au_sample[i].set_xbound(u_min[i], u_max[i])
            self.au_sample[i].set_xlabel(name)
        plt.ion()
        plt.show()

    def _plot_action_sample(self, action):

        if self.fu_sample is None:
            self._start_plot_action_sample()
            plt.pause(0.0001)

        # for now just consider the sample over first step
        u = action[:, 0, :].cpu().numpy()
        for i in range(self.env.nu):
            self.au_sample[i].clear()
            self.au_sample[i].hist(u[:, i])
        plt.pause(0.0001)

    def start_plot_runs(self):
        axis_name = self.env.state_names()
        state_dim = self.traj.shape[1]
        assert state_dim == len(axis_name)
        ctrl_dim = self.u.shape[1]

        self.fig, self.axes = plt.subplots(state_dim, 1, sharex='all')
        self.fu, self.au = plt.subplots(ctrl_dim, 1, sharex='all')
        if self._predicts_state():
            self.fd, self.ad = plt.subplots(state_dim, 1, sharex='all')
        # plot of other info
        self.fo, self.ao = plt.subplots(3, 1, sharex='all')
        self.ao[0].set_ylabel('reaction magnitude')
        self.ao[1].set_ylabel('wall contacts')
        self.ao[2].set_ylabel('predicted dynamics_class')

        for i in range(state_dim):
            self.axes[i].set_ylabel(axis_name[i])
            if self._predicts_state():
                self.ad[i].set_ylabel('d' + axis_name[i])
        for i in range(ctrl_dim):
            self.au[i].set_ylabel('$u_{}$'.format(i))

        self.fig.tight_layout()
        self.fu.tight_layout()
        self.fo.tight_layout()

        plt.ion()
        plt.show()

    def _plot_data(self):
        if self.fig is None:
            self.start_plot_runs()
            plt.pause(0.0001)

        t = np.arange(1, self.pred_traj.shape[0])
        for i in range(self.traj.shape[1]):
            self.axes[i].plot(self.traj[:, i], label='true')
            if self._predicts_state():
                self.axes[i].scatter(t, self.pred_traj[1:, i], marker='*', color='k', label='predicted')
                self.ad[i].plot(self.model_error[:, i])
        self.axes[0].legend()

        mag = np.linalg.norm(self.reaction_force, axis=1)
        self.ao[0].plot(mag)
        self.ao[1].plot(self.wall_contact)
        self.ao[2].plot(self.pred_cls)

        self.fig.canvas.draw()
        for i in range(self.u.shape[1]):
            self.au[i].plot(self.u[:, i])
        plt.pause(0.0001)

    def _reset_sim(self):
        return self.env.reset()
