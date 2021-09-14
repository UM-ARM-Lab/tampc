import time
import random
from typing import Dict
import argparse

import torch
import numpy as np
import pybullet as p

from arm_pytorch_utilities import controller, rand

from stucco.env_getters.arm import ArmGetter
from stucco import tracking
from stucco.env import arm
from stucco.env.arm import task_map, Levels


class SimpleCartesianDynamics:
    def __init__(self):
        self.dtype = torch.float32
        self.device = "cpu"

    def __call__(self, state, u):
        new_state = state.clone()
        new_state[:, :2] += u * arm.FloatingGripperEnv.MAX_PUSH_DIST
        return new_state


class GreedyControllerWithRandomWalkOnContact(controller.Controller):
    """Sample actions then take one that leads to lowest cost; take a random action after experiencing contact"""

    def __init__(self, env, dynamics, cost_to_go, contact_set: tracking.ContactSetHard, u_min, u_max, num_samples=100,
                 walk_length=3, plot_contact_set=False):
        super().__init__()
        self.env = env
        self.nu = env.nu
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = dynamics
        self.cost = cost_to_go
        self.num_samples = num_samples

        self.max_walk_length = walk_length
        self.remaining_random_actions = 0

        self.x_history = []
        self.u_history = []

        self.plot_contact_set = plot_contact_set
        if self.plot_contact_set:
            self.contact_set = contact_set
            self.ground_truth_contact_map: Dict[int, tracking.ContactObject] = {}

    def command(self, obs, info=None):
        d = self.dynamics.device
        dtype = self.dynamics.dtype

        self.x_history.append(obs)

        if self.env.contact_detector.in_contact():
            self.remaining_random_actions = self.max_walk_length
            contact_id = info['contact_id']
            if self.plot_contact_set:
                if contact_id not in self.ground_truth_contact_map and self.contact_set is not None:
                    self.ground_truth_contact_map[contact_id] = self.contact_set.contact_object_factory()
                    self.contact_set.append(self.ground_truth_contact_map[contact_id])
                self.ground_truth_contact_map[contact_id].add_transition(self.x_history[-2], self.u_history[-1],
                                                                         self.x_history[-1] - self.x_history[-2])

        if self.remaining_random_actions > 0:
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
            self.remaining_random_actions -= 1
        else:
            # take greedy action if not in contact
            state = torch.from_numpy(obs).to(device=d, dtype=dtype).repeat(self.num_samples, 1)
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=(self.num_samples, self.nu))
            u = torch.from_numpy(u).to(device=d, dtype=dtype)

            next_state = self.dynamics(state, u)
            costs = self.cost(torch.from_numpy(self.goal).to(device=d, dtype=dtype), next_state)
            min_i = torch.argmin(costs)
            u = u[min_i].cpu().numpy()

        self.u_history.append(u)
        return u


def collect_tracking(level, seed_offset=0, trials=50, trial_length=300, force_gui=True, plot_contact_set=False):
    env = ArmGetter.env(level=level, mode=p.GUI if force_gui else p.DIRECT)
    contact_params = ArmGetter.contact_parameters(env)

    def cost_to_go(state, goal):
        return env.state_distance_two_arg(state, goal)

    def create_contact_object():
        return tracking.ContactUKF(None, contact_params)

    ctrl = controller.Controller()
    save_dir = '{}{}'.format(ArmGetter.env_dir, level)
    sim = arm.ExperimentRunner(env, ctrl, num_frames=trial_length, plot=False, save=True,
                               stop_when_done=True, save_dir=save_dir)

    # randomly distribute data
    for offset in range(trials):
        u_min, u_max = env.get_control_bounds()

        # use mode p.GUI to see what the trials look like
        seed = rand.seed(seed_offset + offset)

        contact_set = tracking.ContactSetHard(contact_params, contact_object_factory=create_contact_object)
        ctrl = GreedyControllerWithRandomWalkOnContact(env, SimpleCartesianDynamics(), cost_to_go,
                                                       contact_set,
                                                       u_min,
                                                       u_max,
                                                       walk_length=6, plot_contact_set=plot_contact_set)
        # random position
        intersects_existing_objects = True
        while intersects_existing_objects:
            init = [random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)]
            init_state = np.array(init + [0, 0])
            goal = [random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)]

            # reject if init and goal is too close
            if np.linalg.norm(np.subtract(init, goal)) < 0.7:
                continue

            env.set_task_config(init=init, goal=goal)
            env.set_state(env.goal)

            # want both goal and start to be free from collision
            p.performCollisionDetection()
            goal_intersection = False
            for obj in env.movable + env.immovable:
                c = env.get_ee_contact_info(obj)
                if len(c):
                    goal_intersection = True
                    break
            if goal_intersection:
                continue

            env.set_state(init_state)
            ctrl.set_goal(env.goal)

            p.performCollisionDetection()
            for obj in env.movable + env.immovable:
                c = env.get_ee_contact_info(obj)
                if len(c):
                    break
            else:
                intersects_existing_objects = False

        sim.ctrl = ctrl
        env.draw_user_text(f"seed {seed}", xy=(0.5, 0.8, -1))
        sim.run(seed)
        env.clear_debug_trajectories()
        # reset so collision checks are on a valid scene for the next trial
        env.reset()

    env.close()
    # wait for it to fully close; otherwise could skip next run due to also closing that when it's created
    time.sleep(5.)


parser = argparse.ArgumentParser(description='Collect data to tune STUCCO and baselines')
parser.add_argument('--gui', action='store_true', help='force GUI for some commands that default to not having GUI')
# run parameters
parser.add_argument('--task', default=list(task_map.keys())[0], choices=task_map.keys(),
                    help='run parameter: what task to run')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    accepted_levels = [Levels.SELECT1, Levels.SELECT2, Levels.SELECT3, Levels.SELECT4]
    if level not in accepted_levels:
        raise RuntimeError(f"Task must be one of {accepted_levels}")
    for offset in [7]:
        collect_tracking(level, seed_offset=offset, trials=40, force_gui=args.gui)
