#! /usr/bin/env python
import colorama
import numpy as np
import logging

try:
    import rospy

    rospy.init_node("victor_retrieval", log_level=rospy.DEBUG)
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

from arc_utilities import ros_init
from arm_robots.victor import Victor
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

from tampc.env.arm_real import ContactDetectorPlanarRealArm, RealArmEnv
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

ask_before_moving = True


def myinput(msg):
    global ask_before_moving
    if ask_before_moving:
        input(msg)


def main():
    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    colorama.init(autoreset=True)

    env = RealArmEnv()
    while True:
        loc = env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)
        if loc is not None:
            print(f"contact at {loc}")

    # rospy.sleep(1)
    # victor.open_left_gripper()
    # rospy.sleep(1)
    # victor.close_left_gripper()
    # rospy.sleep(1)
    # victor.open_right_gripper()
    # rospy.sleep(1)
    # victor.close_right_gripper()
    # rospy.sleep(1)
    #
    # print("press enter if prompted")
    #
    # # Plan to joint config
    # myinput("Plan to joint config?")
    # victor.plan_to_joint_config(victor.right_arm_group, [0.35, 1, 0.2, -1, 0.2, -1, 0])
    #
    # # Plan to joint config by a named group_state
    # myinput("Plan to joint config?")
    # victor.plan_to_joint_config('both_arms', 'zero')
    #
    # # Plan to pose
    # myinput("Plan to pose 1?")
    # victor.plan_to_pose(victor.right_arm_group, victor.right_tool_name, [0.6, -0.2, 1.0, 4, 1, 0])
    #
    # # Or you can use a geometry msgs Pose
    # myinput("Plan to pose 2?")
    # pose = Pose()
    # pose.position.x = 0.7
    # pose.position.y = -0.2
    # pose.position.z = 1.0
    # q = quaternion_from_euler(np.pi, 0, 0)
    # pose.orientation.x = q[0]
    # pose.orientation.y = q[1]
    # pose.orientation.z = q[2]
    # pose.orientation.w = q[3]
    # victor.plan_to_pose(victor.right_arm_group, victor.right_tool_name, pose)

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
