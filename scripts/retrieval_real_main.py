#! /usr/bin/env python
import colorama
import numpy as np
import logging

from tampc.env.arm import Levels
from tampc.util import EnvGetter

try:
    import rospy

    rospy.init_node("victor_retrieval", log_level=rospy.DEBUG)
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

import os
from datetime import datetime

from tampc import cfg
from tampc.env import arm_real
from cottun import tracking

ask_before_moving = True

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True


def no_function():
    raise RuntimeError("This function shouldn't be run!")


class RealRetrievalGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "arm_real"

    @classmethod
    def env(cls, level=Levels.NO_CLUTTER, **kwargs):
        level = Levels(level)
        env = arm_real.RealArmEnv(environment_level=level)
        return env

    @staticmethod
    def ds(env, data_dir, **kwargs):
        return None

    @staticmethod
    def pre_invariant_preprocessor(use_tsf):
        return None

    @staticmethod
    def controller_options(env):
        return None

    @staticmethod
    def contact_parameters(env: arm_real.RealArmEnv, **kwargs) -> tracking.ContactParameters:
        params = tracking.ContactParameters(state_to_pos=env.get_ee_pos_states,
                                            pos_to_state=no_function,
                                            control_similarity=no_function,
                                            state_to_reaction=no_function,
                                            max_pos_move_per_action=env.MAX_PUSH_DIST,
                                            length=0.02,
                                            hard_assignment_threshold=0.4,
                                            intersection_tolerance=0.002,
                                            weight_multiplier=0.1,
                                            ignore_below_weight=0.2)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params


def object_robot_penetration_score(pt_to_config, config, object_transform, model_pts):
    """Compute the penetration between object and robot for a given transform of the object"""
    # transform model points by object transform
    transformed_model_points = model_pts @ object_transform.transpose(-1, -2)
    d = pt_to_config(config.view(1, -1), transformed_model_points)
    d = min(d)
    return -d


# TODO sample model points of cheezit box

from pynput import keyboard


class KeyboardDirPressed():
    def __init__(self):
        self._dir = [0, 0]
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    @property
    def dir(self):
        return self._dir

    def on_press(self, key):
        if key == keyboard.Key.down:
            self.dir[1] = -1
        elif key == keyboard.Key.left:
            self.dir[0] = -1
        elif key == keyboard.Key.up:
            self.dir[1] = 1
        elif key == keyboard.Key.right:
            self.dir[0] = 1

    def on_release(self, key):
        if key in [keyboard.Key.down, keyboard.Key.up]:
            self.dir[1] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.dir[0] = 0


def main():
    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    colorama.init(autoreset=True)

    env = arm_real.RealArmEnv()
    contact_params = RealRetrievalGetter.contact_parameters(env)

    pt_to_config = arm_real.RealArmPointToConfig(env)
    contact_set = tracking.ContactSetSoft(pt_to_config, contact_params)

    # while True:
    #     env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)

    # confirm pt to config implementation
    # config = env.state
    # from arm_pytorch_utilities import rand
    # import torch
    # rand.seed(1)
    # pts = (torch.rand((10, 2)) - 0.5) * 0.3
    # pts += config
    # pts[:, 1] += 0.1
    # d = pt_to_config(torch.from_numpy(config).view(1, -1), pts)
    # d = d.view(-1)
    # for i, pt in enumerate(pts):
    #     env.vis.ros.draw_point(f'temp.{i}', pt, height=env.REST_POS[2], label=str(round(d[i].item(), 2)),
    #                            color=(1, 1, 1, 1))
    #     print(d[i])

    # test basic environment control
    pushed = KeyboardDirPressed()
    print("waiting for arrow keys to be pressed to command a movement")
    while True:
        env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)
        push = pushed.dir
        if push[0] != 0 or push[1] != 0:
            obs, _, done, info = env.step(push)
            print(f"pushed {push} state {obs}")
        rospy.sleep(0.1)

    # # Or with cartesian planning
    # myinput("Cartersian motion back to pose 3?")
    # victor.plan_to_position_cartesian(victor.right_arm_group, victor.right_tool_name, [0.9, -0.4, 0.9], step_size=0.01)
    # victor.plan_to_position_cartesian(victor.right_arm_group, victor.right_tool_name, [0.7, -0.4, 0.9], step_size=0.01)
    #
    # # Move hand straight works either with jacobian following
    # myinput("Follow jacobian to pose 2?")
    # victor.store_current_tool_orientations([victor.right_tool_name])
    # victor.follow_jacobian_to_position(victor.right_arm_group, [victor.right_tool_name], [[[0.7, -0.4, 0.6]]])
    # victor.follow_jacobian_to_position(victor.right_arm_group, [victor.right_tool_name], [[[0.8, -0.4, 1.0]]])
    # victor.follow_jacobian_to_position(victor.right_arm_group, [victor.right_tool_name], [[[1.1, -0.4, 0.9]]])
    # result = victor.follow_jacobian_to_position(group_name=victor.right_arm_group,
    #                                             tool_names=[victor.right_tool_name],
    #                                             preferred_tool_orientations=[quaternion_from_euler(np.pi, 0, 0)],
    #                                             points=[[[1.1, -0.2, 0.8]]])

    # victor.display_robot_traj(result.planning_result.plan, 'jacobian')


if __name__ == "__main__":
    main()
