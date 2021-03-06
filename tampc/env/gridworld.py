import logging
import torch
import rospy

import numpy as np
from tampc import cfg
from tampc.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource, Env

# drawer imports
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# runner imports
from arm_pytorch_utilities import simulation
from arm_pytorch_utilities.math_utils import rotate_wrt_origin
from tampc.controller import controller, online_controller
import time
import copy

logger = logging.getLogger(__name__)

DIR = "grid"


class GridLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if self.config.predict_difference:
            y = GridEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)
        return xu, y, cc


class GridEnv(Env):
    """Simple 2D grid environment (with continuous state and action space) for testing claims on traps"""
    nu = 1
    nx = 2
    NOMINAL_DYNAMICS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

    @staticmethod
    def state_names():
        return ['x', 'y']

    @staticmethod
    @handle_data_format_for_state_diff
    def state_difference(state, other_state):
        """Get state - other_state in state space"""
        return state - other_state

    @classmethod
    def state_cost(cls):
        # TODO use Manhattan instead of straight line distance
        return np.diag([1, 1])

    @staticmethod
    def state_distance(state_difference):
        return state_difference.norm(dim=1)

    @staticmethod
    def control_names():
        return ['dir']

    @classmethod
    def get_control_bounds(cls):
        u_min = np.array([0])
        u_max = np.array([len(cls.NOMINAL_DYNAMICS) - 0.0001])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def control_similarity(u1, u2):
        return (torch.floor(u1) == torch.floor(u2)).view(-1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, environment_level=0, goal=(0, 2), init=(5, 5), check_boundaries=True):
        self.level = environment_level
        self.check_boundaries = check_boundaries

        # initial config
        self.goal = None
        self.init = None
        self.size = None
        self.walls = set()
        self.cw_dynamics = set()

        self.set_task_config(goal, init)
        self._setup_experiment()
        self.state = copy.deepcopy(self.init)

    def set_task_config(self, goal=None, init=None):
        if goal is not None:
            if len(goal) != 2:
                raise RuntimeError("Expected goal to be (x, y), instead got {}".format(goal))
            self.goal = np.array(goal)
        if init is not None:
            if len(init) != 2:
                raise RuntimeError("Expected init to be (x, y), instead got {}".format(init))
            self.init = np.array(init)

    def _setup_experiment(self):
        if self.level == 0:
            self.size = (12, 12)
        elif self.level == 1 or self.level == 4:
            self.size = (12, 12)
            for y in range(3, 9):
                self.walls.add((5, y))
            self.walls.update([(4, 3), (6, 3), (4, 8), (6, 8)])
            if self.level == 4:
                for y in range(2):
                    for x in range(12):
                        self.cw_dynamics.add((x, y))
        elif self.level == 2:
            self.size = (12, 12)
            for y in range(4, 7):
                self.walls.add((5, y))
            for x in range(2, 6):
                self.walls.add((x, 4))
            for x in range(3, 6):
                self.walls.add((x, 6))
        elif self.level == 3:
            self.size = (12, 12)
            for y in range(2, 10):
                self.walls.add((5, y))
            self.walls.update([(4, 2), (6, 2), (4, 9), (6, 9)])
            for y in range(3, 9):
                for x in range(6, 9):
                    self.cw_dynamics.add((x, y))

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        manhattan_dist = np.sum(np.abs(state - self.goal))
        done = manhattan_dist == 0
        return manhattan_dist, done

    # --- control (commonly overridden)
    def _inside_boundaries(self, new_state):
        return (new_state[0] >= 0) and (new_state[0] < self.size[0]) and (new_state[1] >= 0) and (
                new_state[1] < self.size[1])

    def _take_step(self, state, action):
        dx, dy = self.unpack_action(state, action)

        # only move if we'll stay within map and not move into a wall
        new_state = (state[0] + dx, state[1] + dy)
        if (not self.check_boundaries) or (self._inside_boundaries(new_state) and new_state not in self.walls):
            state = new_state
        return state

    def unpack_action(self, state, action):
        action = int(action)  # floors
        dyn = self.NOMINAL_DYNAMICS
        if state in self.cw_dynamics:
            dyn = self.NOMINAL_DYNAMICS[1:] + self.NOMINAL_DYNAMICS[:1]
        dx, dy = dyn[action]
        return dx, dy

    def step(self, action):
        action = np.clip(action, *self.get_control_bounds())

        self.state = np.array(self._take_step(tuple(self.state), action))
        cost, done = self.evaluate_cost(self.state, action)

        return np.copy(self.state), -cost, done, None

    def reset(self):
        self.state = np.copy(self.init)
        return np.copy(self.state), None

    def _compute_trap_difficulty_state(self, state, ctrl):
        visited = set()
        visited.add(state)
        current = [state]
        d = 1
        c, _ = self.evaluate_cost(np.array(state))
        if c == 0:
            return 0, []
        # BFS on the current node until we find a node with lower cost
        while len(current):
            next_depth = []
            for s in current:
                # look at all neighbours
                for action in range(len(self.NOMINAL_DYNAMICS)):
                    new_s = self._take_step(s, action)
                    # ignore if we've already visited it
                    if new_s in visited:
                        continue
                    ccc, _ = self.evaluate_cost(np.array(new_s))
                    if ccc < c:
                        if d > 1:
                            return d, self.compute_trap_basin(state)
                        else:
                            return 0, []
                    next_depth.append(new_s)
                    visited.add(new_s)
            d += 1
            current = next_depth

    def compute_true_trap_difficulty(self, ctrl):
        """Given the current environment, compute: X -> R which maps each state to its trap difficulty"""
        trap_difficulty = {}
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                state = (x, y)
                if state in self.walls:
                    continue
                trap_difficulty[state] = self._compute_trap_difficulty_state(state, ctrl)

        return trap_difficulty

    def compute_trap_basin(self, trap_state):
        """Given the state, compute the basin of attraction around it"""
        visited = set()
        visited.add(trap_state)
        basin = set()
        basin.add(trap_state)
        c, _ = self.evaluate_cost(np.array(trap_state))
        current = [trap_state]
        # BFS on the current node to expand all nodes that will go to trap_state greedily
        while len(current):
            next_depth = []
            for s in current:
                # look at all neighbours
                for action in range(len(self.NOMINAL_DYNAMICS)):
                    new_s = self._take_step(s, action)
                    # ignore if we've already visited it
                    if new_s in visited:
                        continue
                    # look at their neighbours
                    their_neighbours = []
                    for their_action in range(len(self.NOMINAL_DYNAMICS)):
                        their_next_s = self._take_step(new_s, their_action)
                        # check if their minimum cost neighbour is in the basin
                        ccc, _ = self.evaluate_cost(np.array(their_next_s))
                        their_neighbours.append((ccc, their_next_s))
                    their_neighbours = sorted(their_neighbours)
                    min_cost_state = their_neighbours[0][1]
                    if min_cost_state in basin:
                        next_depth.append(new_s)
                        basin.add(new_s)
                    visited.add(new_s)
            current = next_depth

        return basin


class GridContinuousEnv(GridEnv):
    """Continuous action version of grid world"""
    nu = 2

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1])

    @staticmethod
    def state_distance(state_difference):
        if torch.is_tensor(state_difference):
            return state_difference.norm(dim=1)
        else:
            return np.linalg.norm(state_difference, axis=1)

    @staticmethod
    def control_names():
        return ['d$x$', 'd$y$']

    @classmethod
    def get_control_bounds(cls):
        u_min = np.array([-1, -1])
        u_max = np.array([1, 1])
        return u_min, u_max

    @staticmethod
    @handle_data_format_for_state_diff
    def control_similarity(u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def evaluate_cost(self, state, action=None):
        dist = np.linalg.norm(self.state_difference(state, self.goal))
        done = dist < self.dist_for_done
        return dist, done

    def __init__(self, mini_step_magnitude=0.05, dist_for_done=0.1, **kwargs):
        self.dist_for_done = dist_for_done
        self.mini_step_magnitude = mini_step_magnitude
        super(GridContinuousEnv, self).__init__(**kwargs)

    # --- control (commonly overridden)
    def unpack_action(self, state, action):
        return action

    def _take_step(self, state, action):
        # simulate taking steps in mini_steps
        step_size = np.linalg.norm(action)
        if step_size == 0:
            return state
        dmini = action / step_size * self.mini_step_magnitude
        stepped = 0

        while stepped < step_size:
            # last step, scale action down to the remaining step
            if stepped + self.mini_step_magnitude >= step_size:
                remaining_size = step_size - stepped
                dmini = action / step_size * remaining_size

            tile = self.get_tile(state)
            dxy = dmini
            if tile in self.cw_dynamics:
                dxy = np.array(rotate_wrt_origin(dxy, -np.pi / 2))

            new_state = state + dxy
            # can't move anymore (no sliding) so just return current state
            if self.check_boundaries and (
                    not self._inside_boundaries(new_state) or self.get_tile(new_state) in self.walls):
                return state
            state = new_state
            stepped += self.mini_step_magnitude

        return state

    # def _compute_trap_difficulty_state(self, state):
    #     """Compute trap difficulty with a definition of trap that is no movement induced by an optimal controller
    #     that knows the true local dynamics with a horizon of 1"""
    #     # optimal controller under euclidean distance would take straight line
    #     action = self.goal - state
    #     c, _ = self.evaluate_cost(state)
    #     # if it knew local dynamics
    #     if self.get_tile(state) in self.cw_dynamics:
    #         action = np.array(rotate_wrt_origin(action, np.pi / 2))
    #
    #     step_size = np.linalg.norm(action)
    #     if step_size > 1:
    #         action /= step_size
    #
    #     # if it also knew wall dynamics
    #     if self.get_tile(state + action) in self.walls:
    #         for dim in range(2):
    #             new_action = np.copy(action)
    #             new_action[dim] = 0
    #             if self.get_tile(state + new_action) not in self.walls:
    #                 action = new_action
    #                 break
    #
    #     step_size = np.linalg.norm(action)
    #     new_s = self._take_step(state, action)
    #     new_c, _ = self.evaluate_cost(new_s)
    #     blocked = self.state_distance(self.state_difference(state, new_s)) < 0.1
    #     decreased_cost = new_c + step_size * 0.05 < c
    #     # return 1 if blocked else 0
    #     return 1 if not decreased_cost else 0

    # def _compute_trap_difficulty_state(self, state, ctrl):
    #     """Compute trap difficulty with a definition of trap that is no movement induced by an optimal controller
    #     that knows the true local dynamics with a horizon of 1"""
    #     # optimal controller under euclidean distance would take straight line
    #     action = self.goal - state
    #     c, _ = self.evaluate_cost(state)
    #     d = 1
    #     max_d = 20
    #
    #     while d < max_d:
    #         # if it knew local non-nominal dynamics
    #         if self.get_tile(state) in self.cw_dynamics:
    #             action = np.array(rotate_wrt_origin(action, np.pi / 2))
    #         step_size = np.linalg.norm(action)
    #         if step_size > 1:
    #             action /= step_size
    #         # if it also knew wall dynamics
    #         if self.get_tile(state + action) in self.walls:
    #             for dim in range(2):
    #                 new_action = np.copy(action)
    #                 new_action[dim] = 0
    #                 if self.get_tile(state + new_action) not in self.walls:
    #                     action = new_action
    #                     break
    #         step_size = np.linalg.norm(action)
    #
    #         new_s = self._take_step(state, action)
    #         new_c, _ = self.evaluate_cost(new_s)
    #         if new_c + step_size * 0.05 < c:
    #             return d if d > 1 else 0
    #
    #         blocked = self.state_distance(self.state_difference(state, new_s)) < 0.1 * step_size
    #         if blocked:
    #             return 1000
    #         d += 1
    #         state = new_s
    #     return 1000

    def _compute_trap_difficulty_state(self, state, ctrl):
        """Applying the controller leads to entering a state cycle (visiting past state)"""
        states = [state]
        while True:
            action = ctrl.command(state, None)
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()

            state = self._take_step(state, action)
            # got to goal without getting stuck
            cost, done = self.evaluate_cost(state)
            if done:
                return 0, []
            # otherwise check if we are in a cycle (visited before)
            # TODO detect cycle instead of just the same previous state - would have to assume controller is stateless
            for i, prev_state in reversed(list(enumerate(states))):
                dist = self.state_distance(self.state_difference(state, prev_state))
                if dist < 0.001:
                    return 1, states[i:]
            states.append(state)

    def compute_true_trap_difficulty(self, ctrl):
        """Given the current environment, compute: X -> R which maps each state to its trap difficulty"""
        trap_difficulty = {}
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                state = (x, y)
                if self.get_tile(state) in self.walls:
                    continue
                trap_difficulty[state] = self._compute_trap_difficulty_state(np.array(state), ctrl)

        return trap_difficulty

    @staticmethod
    def get_tile(state):
        return round(state[0]), round(state[1])


# TODO move shared parts of this out of this function
class DebugRvizDrawer:
    BASE_SCALE = 0.5
    BASE_Z = 0.1

    def __init__(self, action_scale=0.5, max_nominal_model_error=20):
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=0)
        self.action_scale = action_scale * self.BASE_SCALE
        self.max_nom_model_error = max_nominal_model_error

    def make_marker(self, scale=BASE_SCALE, marker_type=Marker.POINTS):
        marker = Marker()
        marker.header.frame_id = "victor_root"
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        return marker

    def draw_board(self, env: GridEnv):
        max_pos = env.size
        z = self.BASE_Z

        # draw all nodes
        marker = self.make_marker(scale=self.BASE_SCALE * 2)
        marker.ns = "nodes"
        marker.id = 0
        marker.color.a = 1
        marker.color.g = 0.8
        marker.color.r = 0.8
        marker.color.b = 0.8
        for x in range(max_pos[0]):
            for y in range(max_pos[1]):
                marker.points.append(Point(x=x, y=y, z=z))
        self.marker_pub.publish(marker)

        # draw all walls
        marker = self.make_marker(scale=self.BASE_SCALE * 2)
        marker.ns = "walls"
        marker.id = 0
        marker.color.a = 1
        marker.color.r = 0.3
        marker.color.g = 0.3
        marker.color.b = 0.3
        for x, y in env.walls:
            marker.points.append(Point(x=x, y=y, z=z + 0.01))
        self.marker_pub.publish(marker)

        # draw all cw dynamics (non-nominal)
        marker = self.make_marker(scale=self.BASE_SCALE * 2)
        marker.ns = "cw_dynamics"
        marker.id = 0
        marker.color.a = 0.5
        marker.color.b = 0.5
        for x, y in env.cw_dynamics:
            marker.points.append(Point(x=x, y=y, z=z + 0.001))
        self.marker_pub.publish(marker)

    def draw_trap_difficulty(self, traps, max_difficulty=5):
        id = 0
        for (x, y), (difficulty, trap_states) in traps.items():
            if difficulty > 0:
                id += 1
                marker = self.make_marker(scale=self.BASE_SCALE * 1.5)
                marker.ns = "computed_trap_basin"
                marker.id = id
                marker.points.append(Point(x=x, y=y, z=self.BASE_Z + 0.005))
                marker.colors.append(ColorRGBA(r=1, g=0, b=0, a=difficulty / max_difficulty))
                self.marker_pub.publish(marker)

                marker = self.make_marker(scale=self.BASE_SCALE * 1)
                marker.ns = "computed_trap_state"
                marker.id = id
                for x, y in trap_states:
                    marker.points.append(Point(x=x, y=y, z=self.BASE_Z + 0.005))
                    marker.colors.append(ColorRGBA(r=1, g=0, b=1, a=difficulty / max_difficulty))
                self.marker_pub.publish(marker)
                rospy.sleep(0.01)

    def draw_state(self, state, time_step, nominal_model_error=0, action=None):
        marker = self.make_marker(scale=self.BASE_SCALE)
        marker.ns = "state_trajectory"
        marker.id = time_step
        z = state[2] if len(state) > 2 else self.BASE_Z + 0.01

        p = Point()
        p.x = state[0]
        p.y = state[1]
        p.z = z
        c = ColorRGBA()
        c.a = 1
        c.r = 0
        c.g = 1.0 * max(0, self.max_nom_model_error - nominal_model_error) / self.max_nom_model_error
        c.b = 0
        marker.colors.append(c)
        marker.points.append(p)
        self.marker_pub.publish(marker)
        if action is not None:
            action_marker = self.make_marker(marker_type=Marker.LINE_LIST, scale=self.action_scale)
            action_marker.ns = "action"
            action_marker.id = 0
            action_marker.points.append(p)
            p = Point(x=state[0] + action[0], y=state[1] + action[1], z=z)
            action_marker.points.append(p)

            c = ColorRGBA()
            c.a = 1
            c.r = 1
            c.g = 0
            c.b = 0
            action_marker.colors.append(c)
            action_marker.colors.append(c)
            self.marker_pub.publish(action_marker)

    def draw_goal(self, goal):
        marker = self.make_marker(scale=self.BASE_SCALE)
        marker.ns = "goal"
        marker.id = 0
        p = Point()
        p.x = goal[0]
        p.y = goal[1]
        p.z = goal[2] if len(goal) > 2 else self.BASE_Z + 0.01
        c = ColorRGBA()
        c.a = 1
        c.r = 1
        c.g = 0.8
        c.b = 0
        marker.colors.append(c)
        marker.points.append(p)
        self.marker_pub.publish(marker)

    def draw_rollouts(self, rollouts):
        if rollouts is None:
            return
        marker = self.make_marker()
        marker.ns = "rollouts"
        marker.id = 0
        # assume states is iterable, so could be a bunch of row vectors
        T = len(rollouts)
        for t in range(T):
            cc = (t + 1) / (T + 1)
            p = Point()
            p.x = rollouts[t][0]
            p.y = rollouts[t][1]
            p.z = self.BASE_Z + 0.01
            c = ColorRGBA()
            c.a = 1
            c.r = 0
            c.g = cc
            c.b = cc
            marker.colors.append(c)
            marker.points.append(p)
        self.marker_pub.publish(marker)

    def draw_trap_set(self, trap_set):
        if trap_set is None:
            return
        state_marker = self.make_marker(scale=self.BASE_SCALE * 1)
        state_marker.ns = "trap_state"
        state_marker.id = 0

        action_marker = self.make_marker(scale=self.BASE_SCALE * 0.3, marker_type=Marker.LINE_LIST)
        action_marker.ns = "trap_action"
        action_marker.id = 0

        z = self.BASE_Z + 0.007 + 0.01

        T = len(trap_set)
        for t in range(T):
            state, action = trap_set[t]

            p = Point()
            p.x = state[0]
            p.y = state[1]
            p.z = z
            state_marker.points.append(p)
            action_marker.points.append(p)
            # TODO unpack action from class to direction to draw it
            p = Point()
            p.x = state[0] + action[0]
            p.y = state[1] + action[1]
            p.z = z
            action_marker.points.append(p)

            cc = (t + 1) / (T + 1)
            c = ColorRGBA()
            c.a = 1
            c.r = 1
            c.g = 0
            c.b = cc
            state_marker.colors.append(c)
            action_marker.colors.append(c)
            action_marker.colors.append(c)

        self.marker_pub.publish(state_marker)
        self.marker_pub.publish(action_marker)

    def clear_markers(self, ns, delete_all=True):
        marker = self.make_marker()
        marker.ns = ns
        marker.action = Marker.DELETEALL if delete_all else Marker.DELETE
        self.marker_pub.publish(marker)

    def clear_all_markers(self):
        # clear board
        self.clear_markers("nodes", delete_all=False)
        self.clear_markers("walls", delete_all=False)
        self.clear_markers("cw_dynamics", delete_all=False)

        self.clear_markers("computed_trap_basin", delete_all=True)
        self.clear_markers("computed_trap_state", delete_all=True)
        self.clear_markers("state_trajectory", delete_all=True)
        self.clear_markers("trap_state", delete_all=False)
        self.clear_markers("trap_action", delete_all=False)
        self.clear_markers("action", delete_all=False)
        self.clear_markers("rollouts", delete_all=False)

    def draw_text(self, label, text, offset, left_offset=0):
        marker = self.make_marker(marker_type=Marker.TEXT_VIEW_FACING, scale=self.BASE_SCALE * 2)
        marker.ns = label
        marker.id = 0
        marker.text = text

        marker.pose.position.y = -1 - offset * self.BASE_SCALE * 2
        marker.pose.position.x = 9 + left_offset * 5
        marker.pose.position.z = self.BASE_Z
        marker.pose.orientation.w = 1

        marker.color.a = 1
        marker.color.r = 0.8
        marker.color.g = 0.3

        self.marker_pub.publish(marker)


class ExperimentRunner(simulation.Simulation):
    def __init__(self, env: GridEnv, ctrl: controller.Controller, num_frames=500, save_dir=DIR,
                 terminal_cost_multiplier=1, stop_when_done=True, pause_s_between_steps=0,
                 **kwargs):

        super().__init__(save_dir=save_dir, num_frames=num_frames, config=cfg, **kwargs)
        self.stop_when_done = stop_when_done
        self.pause_s_between_steps = pause_s_between_steps

        self.env = env
        self.ctrl = ctrl
        self.dd = DebugRvizDrawer()

        # keep track of last run's rewards
        self.terminal_cost_multiplier = terminal_cost_multiplier
        self.last_run_cost = []

    def _configure_physics_engine(self):
        return simulation.ReturnMeaning.SUCCESS

    def _predicts_state(self):
        return isinstance(self.ctrl, controller.ControllerWithModelPrediction)

    def _predicts_dynamics_cls(self):
        return isinstance(self.ctrl, online_controller.OnlineMPC)

    def _has_recovery_policy(self):
        return isinstance(self.ctrl, online_controller.OnlineMPPI)

    def clear_markers(self):
        self.dd.clear_all_markers()

    def _run_experiment(self):
        self.last_run_cost = []
        obs, info = self._reset_sim()
        self.dd.draw_board(self.env)
        self.dd.draw_state(obs, -1)
        if self.ctrl.goal is not None:
            self.dd.draw_goal(self.ctrl.goal)
        traj = [obs]
        u = []
        infos = [info]
        pred_cls = []
        pred_traj = []
        model_error = []
        model_error_normalized = []

        for simTime in range(self.num_frames - 1):
            self.dd.draw_text("step", "{}".format(simTime), 1)
            start = time.perf_counter()

            action = self.ctrl.command(obs, info)

            # visualization before taking action
            model_pred_error = 0
            if isinstance(self.ctrl, online_controller.OnlineMPPI):
                pred_cls.append(self.ctrl.dynamics_class)
                self.dd.draw_text("dynamics class", "dyn cls {}".format(self.ctrl.dynamics_class), 2)

                mode_text = "recovery" if self.ctrl.autonomous_recovery_mode else (
                    "local" if self.ctrl.using_local_model_for_nonnominal_dynamics else "nominal")
                self.dd.draw_text("control mode", mode_text, 3)
                if self.ctrl.trap_set is not None:
                    self.dd.draw_trap_set(self.ctrl.trap_set)

                # print current state; the model prediction error is last time step's prediction about the current state
                if self.ctrl.diff_predicted is not None:
                    model_pred_error = self.ctrl.diff_predicted.norm().item()
                model_error_normalized.append(model_pred_error)

                rollouts = self.ctrl.get_rollouts(obs)
                self.dd.draw_rollouts(rollouts)

            self.dd.draw_state(obs, simTime, model_pred_error, action=self.env.unpack_action(tuple(obs), np.clip(action,
                                                                                                                 *self.env.get_control_bounds())))
            # sanitize action
            if torch.is_tensor(action):
                action = action.cpu()
            action = np.array(action).flatten()
            obs, rew, done, info = self.env.step(action)
            cost = -rew
            logger.info("%d cost %-5.2f took %.3fs done %d action %-12s obs %s", simTime, cost,
                        time.perf_counter() - start, done,
                        np.round(action, 2), np.round(obs, 3))

            self.last_run_cost.append(cost)
            u.append(action)
            traj.append(obs)
            infos.append(info)

            if self._predicts_state():
                pred_traj.append(self.ctrl.predicted_next_state)
                # model error from the previous prediction step (can only evaluate it at the current step)
                model_error.append(self.ctrl.prediction_error(obs))

            if done and self.stop_when_done:
                logger.debug("done and stopping at step %d", simTime)
                break

            if self.pause_s_between_steps:
                rospy.sleep(self.pause_s_between_steps)

        self.traj = np.stack(traj)
        if len(pred_traj):
            self.pred_traj = np.stack(pred_traj)
            self.pred_cls = np.stack(pred_cls)
        else:
            self.pred_traj = np.array([])
            self.pred_cls = np.array([])
        u.append(np.zeros(self.env.nu))
        self.u = np.stack(u)
        # make same length as state trajectory by appending 0 action
        self.info = np.stack(infos)
        if len(model_error):
            self.model_error = np.stack(model_error)
            self.model_error_normalized = np.stack(model_error_normalized)
        else:
            self.model_error = np.array([])
            self.model_error_normalized = np.array([])

        terminal_cost, done = self.env.evaluate_cost(self.traj[-1])
        self.last_run_cost.append(terminal_cost * self.terminal_cost_multiplier)

        assert len(self.last_run_cost) == self.u.shape[0]

        return simulation.ReturnMeaning.SUCCESS

    def _export_data_dict(self):
        # output (1 step prediction)
        # only save the full information rather than states to allow changing dynamics dimensions without recollecting
        X = self.traj
        # mark the end of the trajectory (the last time is not valid)
        mask = np.ones(X.shape[0], dtype=int)
        # need to also throw out first step if predicting reaction force since there's no previous state
        mask[0] = 0
        mask[-1] = 0
        return {'X': X, 'U': self.u, 'X_pred': self.pred_traj, 'model error': self.model_error,
                'model error normalized': self.model_error_normalized, 'mask': mask.reshape(-1, 1)}

    def _reset_sim(self):
        return self.env.reset()


class GridDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {GridEnv: GridLoader, GridContinuousEnv: GridLoader}
        return loader_map.get(env_type, None)
