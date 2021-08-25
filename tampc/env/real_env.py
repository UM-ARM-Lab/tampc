# utilities for real environments that are typically using ROS
import os.path
import time
from datetime import datetime

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from window_recorder.recorder import WindowRecorder

from arm_video_recorder.srv import TriggerVideoRecording
from tampc import cfg
from tampc.env.env import Visualizer
import logging

logger = logging.getLogger(__name__)


class VideoLogger:
    def __init__(self):
        self.wr = WindowRecorder(window_names=("RViz*", "RViz"), name_suffix="rviz", frame_rate=30.0,
                                 save_dir=cfg.VIDEO_DIR)

    def __enter__(self):
        logger.info("Start recording videos")
        srv_name = "video_recorder"
        rospy.wait_for_service(srv_name)
        self.srv_video = rospy.ServiceProxy(srv_name, TriggerVideoRecording)
        self.srv_video(os.path.join(cfg.VIDEO_DIR, '{}_robot.mp4'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),
                       True, 3600)
        self.wr.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stop recording videos")
        # stop logging video
        self.wr.__exit__()
        if self.srv_video is not None:
            self.srv_video('{}.mp4'.format(time.time()), False, 3600)


class DebugRvizDrawer(Visualizer):
    BASE_SCALE = 0.005

    def __init__(self, action_scale=0.1, max_nominal_model_error=20):
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=0)
        self.action_scale = action_scale
        self.max_nom_model_error = max_nominal_model_error
        # self.array_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=0)

    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, length_ratio=1, rot=0, height=None, label=None,
                   scale=2):
        ns, this_id = name.split('.')
        marker = self.make_marker()
        marker.ns = ns
        marker.id = int(this_id)
        z = height if height is not None else point[2]

        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = z
        c = ColorRGBA()
        c.a = 1
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        marker.colors.append(c)
        marker.points.append(p)
        self.marker_pub.publish(marker)
        return p

    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=None):
        pass

    def draw_2d_line(self, name, start, diff, color=(0, 0, 0), size=2., scale=0.4):
        pass

    def make_marker(self, scale=BASE_SCALE, marker_type=Marker.POINTS):
        marker = Marker()
        marker.header.frame_id = "victor_root"
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        return marker

    def draw_state(self, state, time_step, nominal_model_error=0, action=None, height=None):
        z = height if height is not None else state[2]
        p = self.draw_point("state_trajectory.{}".format(time_step), state, (
            0, 1.0 * max(0, self.max_nom_model_error - nominal_model_error) / self.max_nom_model_error, 0),
                            height=height)
        if action is not None:
            action_marker = self.make_marker(marker_type=Marker.LINE_LIST)
            action_marker.ns = "action"
            action_marker.id = 0
            action_marker.points.append(p)
            p = Point()
            p.x = state[0] + action[0] * self.action_scale
            p.y = state[1] + action[1] * self.action_scale
            p.z = z
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
        marker = self.make_marker(scale=self.BASE_SCALE * 2)
        marker.ns = "goal"
        marker.id = 0
        p = Point()
        p.x = goal[0]
        p.y = goal[1]
        p.z = goal[2]
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
            p.z = rollouts[t][2]
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
        state_marker = self.make_marker(scale=self.BASE_SCALE * 2)
        state_marker.ns = "trap_state"
        state_marker.id = 0

        action_marker = self.make_marker(marker_type=Marker.LINE_LIST)
        action_marker.ns = "trap_action"
        action_marker.id = 0

        T = len(trap_set)
        for t in range(T):
            action = None
            if len(trap_set[t]) == 2:
                state, action = trap_set[t]
            else:
                state = trap_set[t]

            p = Point()
            p.x = state[0]
            p.y = state[1]
            p.z = state[2]
            state_marker.points.append(p)
            if action is not None:
                action_marker.points.append(p)
                p = Point()
                p.x = state[0] + action[0] * self.action_scale
                p.y = state[1] + action[1] * self.action_scale
                p.z = state[2]
                action_marker.points.append(p)

            cc = (t + 1) / (T + 1)
            c = ColorRGBA()
            c.a = 1
            c.r = 1
            c.g = 0
            c.b = cc
            state_marker.colors.append(c)
            if action is not None:
                action_marker.colors.append(c)
                action_marker.colors.append(c)

        self.marker_pub.publish(state_marker)
        self.marker_pub.publish(action_marker)

    def clear_markers(self, ns, delete_all=True):
        marker = self.make_marker()
        marker.ns = ns
        marker.action = Marker.DELETEALL if delete_all else Marker.DELETE
        self.marker_pub.publish(marker)

    def draw_text(self, label, text, offset, left_offset=0, scale=5):
        marker = self.make_marker(marker_type=Marker.TEXT_VIEW_FACING, scale=self.BASE_SCALE * scale)
        marker.ns = label
        marker.id = 0
        marker.text = text

        marker.pose.position.x = 1.4 + offset * self.BASE_SCALE * 6
        marker.pose.position.y = 0.4 + left_offset * 0.5
        marker.pose.position.z = 1
        marker.pose.orientation.w = 1

        marker.color.a = 1
        marker.color.r = 0.8
        marker.color.g = 0.3

        self.marker_pub.publish(marker)
