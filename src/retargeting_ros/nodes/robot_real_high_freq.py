#!/usr/bin/env python3
import threading
import time
from typing import List, Optional, Union

import numpy as np
from scipy.interpolate import CubicSpline
from retargeting_ros.messages import stamp_to_seconds, time_to_seconds

try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ModuleNotFoundError as _ROS_IMPORT_ERROR:
    rclpy = None
    ReentrantCallbackGroup = None
    Node = object
    JointState = None
    Float64MultiArray = None
else:
    _ROS_IMPORT_ERROR = None


def _require_ros_support() -> None:
    if _ROS_IMPORT_ERROR is not None:
        from retargeting_ros._optional import missing_ros_error

        raise missing_ros_error("RobotRealHighFreq", _ROS_IMPORT_ERROR)


class RobotRealHighFreq(Node):

    def __init__(self):
        _require_ros_support()
        super().__init__("robot_real_high_freq")
        # -------- hyper-parameters begin --------
        self.n_arm_joints = 7
        self.n_hand_joints = 16
        self.window_size = 4
        self.hf_commands_window_size = 5
        self.timer_period = 0.01  # seconds
        self.rviz_viz = False
        # -------- hyper-parameters end --------

        if self.rviz_viz:
            self.joint_names = [f"panda_joint{i+1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
            self.robot_joint_state_vis_pub = self.create_publisher(JointState, "visualize/robot_joint_states", 10)

        self.sent_hf_commands_window = []
        self.sent_hf_commands_stamp_window = []
        self.joint_pos_command_high_freq: Optional[np.ndarray] = None
        self.sent_num_command_high_freq: int = 0

        self.lock = threading.Lock()
        self.arm_joint_pos_command_pub = self.create_publisher(Float64MultiArray, "franka/joint_impedance_command", 10)
        self.hand_joint_pos_command_pub = self.create_publisher(JointState, "/cmd_leap", 10)
        self.joint_pos_command_sub = self.create_subscription(
            JointState,
            "robot/joint_pos_command_low_freq",
            self._robot_command_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.timer = self.create_timer(self.timer_period, self._timer_callback, callback_group=ReentrantCallbackGroup())
        self.get_logger().info("Ready ...")

    def _publish_arm_joint_pos_command(self, joint_pos):
        assert len(joint_pos) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        self.arm_joint_pos_command_pub.publish(msg)

    def _publish_hand_joint_pos_command(self, joint_pos):
        assert len(joint_pos) == self.n_hand_joints
        msg = JointState()
        msg.position = joint_pos.tolist()
        self.hand_joint_pos_command_pub.publish(msg)

    def _robot_command_callback(self, msg):
        self.get_logger().info(f"Received low-freqency joint command: {list(msg.position)}")

        current_time = time_to_seconds(self.get_clock().now())
        target_command = np.asarray(msg.position)
        target_reach_time = stamp_to_seconds(msg.header.stamp)

        if len(self.sent_hf_commands_window) >= 2:
            self.lock.acquire()
            xs = self.sent_hf_commands_stamp_window + [target_reach_time]
            ys = self.sent_hf_commands_window + [target_command]
            self.lock.release()
            spline_func = CubicSpline(xs, ys, bc_type="natural")
            x_new = np.arange(current_time, target_reach_time, self.timer_period)
            y_new = spline_func(x_new)
        else:
            y_new = np.asarray([target_command])

        self.lock.acquire()
        self.joint_pos_command_high_freq = y_new
        self.sent_num_command_high_freq = 0
        self.lock.release()

    def _timer_callback(self):
        self.lock.acquire()

        if (self.joint_pos_command_high_freq is not None) and self.sent_num_command_high_freq < len(
            self.joint_pos_command_high_freq
        ):
            command = self.joint_pos_command_high_freq[self.sent_num_command_high_freq]
            self._publish_arm_joint_pos_command(command[: self.n_arm_joints])
            self._publish_hand_joint_pos_command(command[self.n_arm_joints :])
            self.sent_num_command_high_freq += 1

            # maintain the window
            self.sent_hf_commands_window.append(command)
            self.sent_hf_commands_stamp_window.append(time_to_seconds(self.get_clock().now()))
            if len(self.sent_hf_commands_window) > self.hf_commands_window_size:
                self.sent_hf_commands_window.pop(0)
                self.sent_hf_commands_stamp_window.pop(0)

            if self.rviz_viz:
                msg = JointState()
                msg.name = self.joint_names
                msg.position = list(command)
                self.robot_joint_state_vis_pub.publish(msg)

            # self.get_logger().debug(f"Sending high-frequency command: {command}.")

            # print("HF command: ", command)
            # print("time.time(): ", time.time())

        self.lock.release()


def main(args=None):
    _require_ros_support()
    rclpy.init(args=args)

    robot = RobotRealHighFreq()

    rclpy.spin(robot)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
