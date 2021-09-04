#! /usr/bin/env python
import colorama
import numpy as np
import logging
import torch

# from window_recorder.recorder import WindowRecorder
from cottun.retrieval_controller import RetrievalPredeterminedController, sample_model_points
from tampc.env.real_env import VideoLogger
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
from cottun import tracking, icp

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
        self.calibrate = False

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
        elif key == keyboard.Key.shift:
            self.calibrate = True

    def on_release(self, key):
        if key in [keyboard.Key.down, keyboard.Key.up]:
            self.dir[1] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.dir[0] = 0
        elif key == keyboard.Key.shift:
            self.calibrate = False


def estimate_wrench_per_dir(env):
    pushed = KeyboardDirPressed()
    dir_to_wrench = {}
    print("waiting for arrow keys to be pressed to command a movement")
    with VideoLogger():
        while not rospy.is_shutdown():
            env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)
            push = tuple(pushed.dir)
            if push[0] != 0 or push[1] != 0:
                if push not in dir_to_wrench:
                    dir_to_wrench[push] = []

                env.static_wrench = None
                env._temp_wrenches = []
                obs, _, done, info = env.step(push)
                dir_to_wrench[push] += env._temp_wrenches

                print(f"pushed {push} state {obs}")
            rospy.sleep(0.1)

    for k, v in dir_to_wrench.items():
        print(f"{k} ({len(v)} points): {np.mean(v, axis=0)} {np.var(v, axis=0)}")
    print("code copy friendly print")
    for k, v in dir_to_wrench.items():
        print(f"{k}: {list(np.mean(v, axis=0))}")


def confirm_pt_to_config(env, pt_to_config):
    # confirm pt to config implementation
    config = env.state
    from arm_pytorch_utilities import rand
    import torch
    rand.seed(1)
    pts = (torch.rand((10, 2)) - 0.5) * 0.3
    pts += config
    pts[:, 1] += 0.1
    d = pt_to_config(torch.from_numpy(config).view(1, -1), pts)
    d = d.view(-1)
    for i, pt in enumerate(pts):
        env.vis.ros.draw_point(f'temp.{i}', pt, height=env.REST_POS[2], label=str(round(d[i].item(), 2)),
                               color=(1, 1, 1, 1))
        print(d[i])


def keyboard_control(env):
    pushed = KeyboardDirPressed()
    print("waiting for arrow keys to be pressed to command a movement")
    with VideoLogger():
        while not rospy.is_shutdown():
            env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)
            if pushed.calibrate:
                env.recalibrate_static_wrench()
            push = tuple(pushed.dir)
            if push[0] != 0 or push[1] != 0:
                obs, _, done, info = env.step(push)
                print(f"pushed {push} state {obs}")
            rospy.sleep(0.1)


class RealRetrievalPredeterminedController(RetrievalPredeterminedController):
    def done(self):
        return self.i >= len(self.controls)

    def command(self, obs, info=None, visualizer=None):
        self.x_history.append(obs)

        if self.i < len(self.controls):
            u = self.controls[self.i]
            self.i += 1
        else:
            u = [0 for _ in range(len(self.controls[0]))]

        # 3 elements in a control means to perform it but not calibrate (and ignore whether we think we're in contact or not)
        skip_update = (u is None) or (len(u) > 2) or (self.i < len(self.controls) and self.controls[self.i] is None)
        if not skip_update:
            prev_u = self.u_history[-1]
            if prev_u is not None:
                prev_u = torch.tensor(prev_u[:2])
            self.contact_set.update(self.x_history[-2], prev_u,
                                    self.x_history[-1] - self.x_history[-2],
                                    self.contact_detector, torch.tensor(info['reaction']), info=info,
                                    visualizer=visualizer)

        self.u_history.append(u)
        return u, skip_update


def predetermined_controls(env, controls, pt_to_config, contact_set):
    input("enter to start execution")
    ctrl = RealRetrievalPredeterminedController(env.contact_detector, contact_set, controls)
    obs, info = env._obs()
    dtype = torch.float32

    z = env._observe_ee(return_z=True)[-1]

    model_points = sample_model_points(None, num_points=50, force_z=z, seed=0, name="cheezit")
    mph = model_points.clone().to(dtype=dtype)
    # make homogeneous [x, y, 1]
    mph[:, -1] = 1

    best_tsf_guess = None

    with VideoLogger():
        while not ctrl.done():
            with env.motion_status_input_lock:
                u, skip_update = ctrl.command(obs, info, env.vis.ros)
            env.visualize_contact_set(contact_set)

            if env.contact_detector.in_contact() and not skip_update:
                pts = contact_set.get_posterior_points()
                to_iter = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
                dist_per_est_obj = []
                for c in to_iter:
                    this_pts = pts[c]
                    T, distances, _ = icp.icp_3(this_pts, model_points[:, :2], given_init_pose=best_tsf_guess, batch=30)
                    T = T.inverse()
                    penetration = [object_robot_penetration_score(pt_to_config, obs, T[b], mph) for b in
                                   range(T.shape[0])]
                    score = np.abs(penetration)
                    best_tsf_index = np.argmin(score)

                    # pick object with lowest variance in its translation estimate
                    translations = T[:, :2, 2]
                    best_tsf_distances = (translations.var(dim=0).sum()).item()

                    dist_per_est_obj.append(best_tsf_distances)
                    if best_distance is None or best_tsf_distances < best_distance:
                        best_distance = best_tsf_distances
                        best_tsf_guess = T[best_tsf_index].inverse()

                rospy.loginfo(f"err each obj {np.round(dist_per_est_obj, 4)}")
                best_T = best_tsf_guess.inverse()

                transformed_model_points = mph @ best_T.transpose(-1, -2)
                for i, pt in enumerate(transformed_model_points):
                    if i % 2 == 0:
                        continue
                    pt = [pt[0], pt[1], z]
                    env._dd.draw_point(f"tmptbest-{i}", pt, color=(0, 0, 1), length=0.008)

            if u is None:
                env.recalibrate_static_wrench()
                continue
            obs, _, done, info = env.step(u[:2])
            print(f"pushed {u} state {obs}")
        rospy.sleep(1)


def clear_all_markers(env):
    env.vis.ros.clear_markers("residualmag")
    env.vis.ros.clear_markers("t")
    env.vis.ros.clear_markers("n")
    env.vis.ros.clear_markers("likely")
    env.vis.ros.clear_markers("most likely contact")
    env.vis.ros.clear_markers("reaction")


def main():
    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    colorama.init(autoreset=True)

    env = arm_real.RealArmEnv()
    contact_params = RealRetrievalGetter.contact_parameters(env)
    clear_all_markers(env)

    pt_to_config = arm_real.RealArmPointToConfig(env)
    contact_set = tracking.ContactSetSoft(pt_to_config, contact_params)

    # while True:
    #     env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)

    # confirm_pt_to_config(env, pt_to_config)
    # estimate_wrench_per_dir(env)
    # keyboard_control(env)
    # test basic environment control

    # get 4 points on the front face
    ctrl = [[0, 0.5], None]
    ctrl += [[0, 0.8]] * 2
    ctrl += [[0.4, -1., None], [1.0, -0.5, None]]
    ctrl += [[0, 0.5], None]
    ctrl += [[0, 0.7]] * 2
    ctrl += [[0.4, -1., None], [1.0, -0.5, None]]
    ctrl += [[0, 0.5], None]
    ctrl += [[0, 0.7]] * 2
    ctrl += [[0.4, -1., None], [1.0, -0.5, None]]
    ctrl += [[0, 0.5], None]
    ctrl += [[0, 0.7]] * 2

    # move to the left side of front the cheezit box and poke again
    ctrl += [[-0.4, -0.6, None]]
    ctrl += [[-1.0, 0], None]
    ctrl += [[-1.0, 0]] * 4
    ctrl += [[0., -0.9, None]]
    ctrl += [[0.0, 0.3], None]
    ctrl += [[0.0, 0.7]] * 2

    ctrl += [[-1.0, 0], None]
    ctrl += [[-1.0, 0]] * 3
    ctrl += [None]
    ctrl += [[-1.0, 0]] * 3
    ctrl += [[0.0, 0.4], None]
    ctrl += [[0.0, 1.0]] * 3
    ctrl += [None]
    ctrl += [[0.0, 1.0]] * 3

    # move to the actual left side
    predetermined_controls(env, ctrl, pt_to_config, contact_set)


if __name__ == "__main__":
    main()
