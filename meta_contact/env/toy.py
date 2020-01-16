import abc

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from arm_pytorch_utilities import load_data as load_utils
from arm_pytorch_utilities.make_data import datasource
from hybrid_sysid import simulation
from meta_contact import cfg
from meta_contact.env import myenv
from sklearn.preprocessing import PolynomialFeatures


class ToyLoader(load_utils.DataLoader):
    def __init__(self, *args, file_cfg=cfg, **kwargs):
        super().__init__(file_cfg, *args, **kwargs)

    def _process_file_raw_data(self, d):
        x = d['X']
        u = d['U'][:-1]
        cc = d['label'][1:]

        if self.config.predict_difference:
            y = x[1:] - x[:-1]
        else:
            y = x[1:]

        x = x[:-1]
        xu = np.column_stack((x, u))

        # potentially many trajectories, get rid of buffer state in between
        mask = d['mask']
        # pack expanded pxu into input if config allows (has to be done before masks); otherwise would use cross-file data)
        if self.config.expanded_input:
            # move y down 1 row (first element can't be used)
            # (xu, pxu)
            xu = np.column_stack((xu[1:], xu[:-1]))
            y = y[1:]
            cc = cc[1:]

            mask = mask[1:-1]
        else:
            mask = mask[:-1]

        mask = mask.reshape(-1) != 0

        xu = xu[mask]
        cc = cc[mask]
        y = y[mask]

        self.config.load_data_info(x, u, y, xu)

        return xu, y, cc


class ToyDataSource(datasource.FileDataSource):
    def __init__(self, data_dir='linear', **kwargs):
        super().__init__(ToyLoader, data_dir, **kwargs)


class ToyEnv:
    """Simple, fully kinematic 2D env that can be visualized"""
    nx = 2
    nu = 2
    ny = 2

    def __init__(self, init_state, goal, xlim=(-3, 3), ylim=(-3, 3), mode=myenv.Mode.GUI,
                 process_noise=(0.01, 0.01), max_move_step=0.01,
                 hide_disp=False, keep_within_bounds=True):
        """Simulate for T time steps with a controller"""
        # set at initialization time without a way to change it
        self.init_state, self.goal, self.last_state, self.state = None, None, None, None
        self.noise = process_noise

        self.xlim = xlim
        self.ylim = ylim

        self.mode = mode
        self.hide_disp = hide_disp
        self.keep_within_bounds = keep_within_bounds

        self.max_move_step = max_move_step

        # quadratic cost
        self.Q = np.eye(self.nx)
        self.R = np.eye(self.nu)

        self.f = None
        self.ax = None
        self.goal_marker = None
        self.set_task_config(goal, init_state)
        if self.mode == myenv.Mode.GUI:
            self.start_visualization()

    def set_task_config(self, goal=None, init_state=None):
        """Change task configuration"""
        if goal is not None:
            self.goal = np.array(goal)
            # remove old goal marker to render only the current one
            if self.goal_marker:
                self.goal_marker.remove()
                self.goal_marker = None
        if init_state is not None:
            self.init_state = np.array(init_state)
            self.last_state = None
            self.state = init_state.copy()

    def start_visualization(self):
        self.f = plt.figure()
        self.ax = self.f.add_subplot(111)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')

        self._draw_background()

        # if hiding display, create figures but don't show to speed up computation
        if self.hide_disp:
            plt.ioff()
        else:
            plt.ion()
            plt.show()
        self.render()

    def stop_visualization(self):
        plt.close(self.f)

    def reset(self, full=False):
        # full reset goes back to real initial conditions
        if full:
            if self.mode == myenv.Mode.GUI:
                self.goal_marker = None
                plt.close(self.f)
                self.start_visualization()
        self.state = self.init_state
        self.last_state = None
        return np.copy(self.state)

    def _evaluate_cost(self, action):
        diff = self.state - self.goal
        cost = diff.T.dot(self.Q).dot(diff)
        done = cost < 0.01
        cost += action.T.dot(self.R).dot(action)
        return cost, done

    def step(self, action, add_noise=True):
        self.last_state = self.state.copy()
        self.state = self.true_dynamics(self.state, action)
        # reduce to 1 dim
        self.state = self.state.reshape(-1)
        if add_noise:
            self.state += np.random.randn(self.nu) * self.noise

        if self.keep_within_bounds:
            bounds = tuple(zip(self.xlim, self.ylim))
            self.state = np.clip(self.state, bounds[0], bounds[1])

        cost, done = self._evaluate_cost(action)

        return np.copy(self.state), -cost, done, self.state_label(self.state)

    def render(self):
        if self.f is None:
            return

        if self.state is not None:
            self.ax.scatter(self.state[0], self.state[1], c='r', s=2)

        # draw an arrow from the last state to current to indicate motion
        if self.last_state is not None:
            diff = self.state - self.last_state
            nd = np.linalg.norm(diff) / 5
            self.ax.arrow(self.last_state[0], self.last_state[1], diff[0] * 0.8, diff[1] * 0.8,
                          head_length=nd, head_width=nd)

        if self.goal is not None and self.goal_marker is None:
            self.goal_marker = self.ax.scatter(self.goal[0], self.goal[1], c='g')

        self.f.canvas.draw()
        plt.pause(0.00001)

    @abc.abstractmethod
    def true_dynamics(self, x, u):
        """Actual x' = f(x,u) model of the environment; preferrably deals with batch data"""

    @abc.abstractmethod
    def state_label(self, x):
        """Give a label (integer or float) for the given state"""

    @abc.abstractmethod
    def _draw_background(self):
        """Draw background of this env to self.ax"""


class WaterWorld(ToyEnv):
    """Simple, fully kinematics world with water and air where water takes more effort to navigate"""

    def __init__(self, *args, **kwargs):
        self.A1 = np.eye(self.nx)
        self.A2 = np.eye(self.nx)
        # self.A2[1,0] = 0.01
        self.B1 = np.eye(self.nu) * 0.5
        self.B2 = np.eye(self.nu) * 0.7
        self.B2[1, 0] = 0.3
        self.B2[0, 1] = -0.2
        super().__init__(*args, **kwargs)

    def _draw_background(self):
        self.ax.fill([self.xlim[0], self.xlim[0], self.xlim[1], self.xlim[1]], [0, self.ylim[0], self.ylim[0], 0],
                     facecolor="none", hatch="X",
                     edgecolor="b")

    def true_dynamics(self, x, u):
        """Actual x' = f(x,u) model of the environment; only deals with non-batch input for now"""
        # TODO batch this (assuming each step keeps you inside the same mode)
        total_step = np.linalg.norm(u)
        move_step = u / total_step * self.max_move_step
        # cache the result of action
        u1 = self.B1 @ move_step
        u2 = self.B2 @ move_step
        while total_step > 0:
            if self.state_label(x):
                x = self.A2 @ x + u2
            else:
                x = self.A1 @ x + u1
            total_step -= self.max_move_step
        return x

    def state_label(self, x):
        if len(x.shape) == 2:
            return x[:, 1] < 0
        return x[1] < 0


def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


class ComplexWaterWorld(WaterWorld):
    def __init__(self, *args, r=1, **kwargs):
        self.order = 2
        self.poly = PolynomialFeatures(self.order, include_bias=False)
        # create input sample to fit (tells sklearn what input sizes to expect)
        u = np.random.rand(self.nu).reshape(1, -1)
        self.poly.fit(u)
        self.rr = 1.8 * r
        self.r = r
        super(ComplexWaterWorld, self).__init__(*args, **kwargs)

    def _draw_background(self):
        iv = make_circle(self.r)
        ov = make_circle(self.rr)
        codes = np.ones(
            len(iv), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO

        # Concatenate the inside and outside subpaths together
        inside = 1
        outside = -1
        vertices = np.concatenate((ov[::outside],
                                   iv[::inside]))
        all_codes = np.concatenate((codes, codes))
        path = mpath.Path(vertices, all_codes)
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor='b', hatch='X')
        self.ax.add_patch(patch)

    def feature(self, x):
        oned = len(x.shape) == 1
        if oned:
            x = x.reshape(1, -1)
        xx = self.poly.transform(x)
        # x y x^2 xy y^2
        r = np.sqrt(xx[:, 2] + xx[:, 4])
        return r

    def state_label(self, x):
        oned = len(x.shape) == 1
        r = self.feature(x)
        res = np.logical_or(r < self.r, r > self.rr)
        if oned:
            return res[0]
        return res


class PolynomialWorld(ToyEnv):
    def __init__(self, *args, **kwargs):
        self.order = 2
        self.poly = PolynomialFeatures(self.order, include_bias=False)
        # create input sample to fit (tells sklearn what input sizes to expect)
        u = np.random.rand(self.nu).reshape(1, -1)
        self.poly.fit(u)
        # self.true_params = np.array([3, 0, 0.6, 0, -0.8])
        # TODO try a non-convex polynomial to see if that makes a difference
        self.true_params = np.array([0, 0, 0.1, 0, 0.1])
        super().__init__(*args, **kwargs)

    def _draw_background(self):
        delta = 0.2
        x = np.arange(self.xlim[0], self.xlim[1] + 0.01, delta)
        y = np.arange(self.ylim[0], self.ylim[1] + 0.01, delta)
        X, Y = np.meshgrid(x, y)
        XY = np.c_[X.ravel(), Y.ravel()]
        Z = self.feature(XY)
        Z = Z.reshape(X.shape)
        CS = self.ax.contourf(X, Y, Z, cmap='plasma')
        CBI = self.f.colorbar(CS)
        CBI.ax.set_ylabel('control scale')

    def true_dynamics(self, x, u):
        # don't interpolate in discrete time to prevent complicating the dynamics
        f = self.feature(x)
        # simply scale control
        dx = u * f.reshape(-1, 1)
        return x + dx

    def feature(self, x):
        oned = len(x.shape) == 1
        if oned:
            x = x.reshape(1, -1)
        xx = self.poly.transform(x)
        # x y x^2 xy y^2
        r = xx @ self.true_params
        return r

    def state_label(self, x):
        return self.feature(x)


class ToySim(simulation.Simulation):
    def __init__(self, env: ToyEnv, controller, num_frames=100, save_dir='linear', **kwargs):
        super(ToySim, self).__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.mode = env.mode

        self.env = env
        self.ctrl = controller

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        self.ctrl.set_goal(self.env.goal)
        return simulation.ReturnMeaning.SUCCESS

    def _init_data(self):
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.num_frames, self.env.nx))
        self.u = np.zeros((self.num_frames, self.env.nu))
        self.labels = np.zeros((self.num_frames,))
        return simulation.ReturnMeaning.SUCCESS

    def _run_experiment(self):
        obs = self._reset_sim()
        for simTime in range(self.num_frames - 1):
            self.traj[simTime, :] = obs
            action = self.ctrl.command(obs)
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            if self.mode == myenv.Mode.GUI:
                self.env.render()

            self.u[simTime, :] = action
            self.traj[simTime + 1, :] = obs
            self.labels[simTime] = info

        return simulation.ReturnMeaning.SUCCESS

    def _export_data_dict(self):
        X = self.traj
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        mask[-1] = 0
        return {'X': X, 'U': self.u, 'label': self.labels.reshape(-1, 1), 'mask': mask.reshape(-1, 1)}

    def _reset_sim(self):
        return self.env.reset()
