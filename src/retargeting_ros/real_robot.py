#!/usr/bin/env python3
import os
import time
from typing import List

import numpy as np

try:
    from leap_hand.srv import LeapPosition, LeapPosVelEff
except ModuleNotFoundError as _LEAP_IMPORT_ERROR:
    LeapPosition = None
    LeapPosVelEff = None
else:
    _LEAP_IMPORT_ERROR = None
from retargeting_ros.messages import seconds_to_stamp

try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
except ModuleNotFoundError as _ROS_IMPORT_ERROR:
    rclpy = None
    ReentrantCallbackGroup = None
    Node = object
    JointState = None
    Float64MultiArray = None
    String = None
else:
    _ROS_IMPORT_ERROR = None


def _require_ros_support() -> None:
    if _ROS_IMPORT_ERROR is not None:
        from retargeting_ros._optional import missing_ros_error

        raise missing_ros_error("RobotReal", _ROS_IMPORT_ERROR)


def _require_leap_support() -> None:
    if _LEAP_IMPORT_ERROR is not None:
        from retargeting_ros._optional import missing_ros_error

        raise missing_ros_error("RobotReal Leap hand service client", _LEAP_IMPORT_ERROR)


def time_to_seconds(time) -> float:
    return time.nanoseconds / 1e9


class RobotReal:
    def __init__(self, node: Node, virtual_hardware=False, use_high_freq_interp=False):
        _require_ros_support()
        # hyper-parameters
        self.n_arm_joints = 7
        self.arm_joint_names = [f"panda_joint{i+1}" for i in range(self.n_arm_joints)]
        self.n_hand_joints = 16
        self.n_joints = self.n_arm_joints + self.n_hand_joints
        self.timestep = 0.05

        # variable
        self.curr_joint_pos = np.zeros((self.n_joints))
        self.target_joint_pos = self.curr_joint_pos.copy()
        self.one_step_time_record = time.time()

        self.node = node
        self.virtual_hardware = virtual_hardware
        self.use_high_freq_interp = use_high_freq_interp

        if not virtual_hardware:
            _require_leap_support()
            # state subscriber
            self.arm_joint_states_sub = self.node.create_subscription(
                JointState,
                "franka/low_freq_joint_states",
                self._arm_joint_states_callback,
                1,
                callback_group=ReentrantCallbackGroup(),
            )
            self.hand_joint_pos_client = self.node.create_client(LeapPosition, "/leap_position")

        # command publisher
        if self.use_high_freq_interp:
            print("Please launch the main_robot_real_high_freq.py node.")
            self.robot_joint_pos_command_pub = self.node.create_publisher(
                JointState, "robot/joint_pos_command_low_freq", 10
            )
        else:
            self.arm_joint_pos_command_pub = self.node.create_publisher(
                Float64MultiArray, "franka/joint_impedance_command", 10
            )
            self.hand_joint_pos_command_pub = self.node.create_publisher(JointState, "/cmd_leap", 10)

        self.record_video_command_pub = self.node.create_publisher(String, "/record_video_command", 10)

    def wait_for_initialization(self):
        self.node.get_logger().info("Waiting for initialization.")
        time.sleep(1.0)
        self.update_joint_pos()
        if not self.virtual_hardware:
            while np.all(self.curr_joint_pos[: self.n_arm_joints] == 0):  # wait for receiving the arm joint states
                self.update_joint_pos()
                time.sleep(0.01)

        # send multiple control command to high-freq-interpolation for padding the window
        for i in range(10):
            self.ctrl_joint_pos(self.get_joint_pos())
            self.step()

        self.target_joint_pos = self.curr_joint_pos.copy()
        self.node.get_logger().info("Robot initialization done.")

    def _arm_joint_states_callback(self, msg):
        indices = [msg.name.index(name) for name in self.arm_joint_names]
        self.curr_joint_pos[: self.n_arm_joints] = np.asarray(msg.position)[indices]

    def _update_hand_joint_pos(self):
        """
        The updated joint pos is the received msg - 180 degree.
        """
        req = LeapPosition.Request()
        future = self.hand_joint_pos_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        hand_joint_pos = np.asarray(future.result().position) - np.pi
        self.curr_joint_pos[self.n_arm_joints :] = hand_joint_pos

    def _publish_arm_joint_pos_command(self, joint_pos):
        assert len(joint_pos) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        # Publish the message to the topic
        self.arm_joint_pos_command_pub.publish(msg)
        # self.node.get_logger().info(f'Publishing: {msg.data}')

    def _publish_hand_joint_pos_command(self, joint_pos):
        """
        The sent msg is the input joint pos + 180 degree.
        """
        assert len(joint_pos) == self.n_hand_joints
        msg = JointState()

        leap_offset = np.pi if not self.virtual_hardware else 0.0
        msg.position = [(q + leap_offset) for q in joint_pos]  # add 180 degree
        self.hand_joint_pos_command_pub.publish(msg)

    def _publish_robot_joint_pos_command(self, joint_pos):
        """
        Publish command to high-frequency interpolator node.
        """
        assert len(joint_pos) == self.n_joints
        msg = JointState()

        # expect the robot to reach the goal waypoint in one timestep
        expected_time = (
            time_to_seconds(self.node.get_clock().now()) + self.timestep
        )  # TODO: is self.timestep appropriate?
        msg.header.stamp = seconds_to_stamp(expected_time)

        leap_offset = np.pi if not self.virtual_hardware else 0.0
        msg.position = [q for q in joint_pos[: self.n_arm_joints]] + [
            (q + leap_offset) for q in joint_pos[self.n_arm_joints :]
        ]  # real leap hand: add 180 degree

        self.robot_joint_pos_command_pub.publish(msg)

    def step(self, refresh=False):
        while (time.time() - self.one_step_time_record) < self.timestep:
            time.sleep(0.001)
        self.one_step_time_record = time.time()

    def ctrl_joint_pos(self, target_joint_pos):
        target_joint_pos = np.asarray(target_joint_pos).tolist()
        if self.use_high_freq_interp:
            self._publish_robot_joint_pos_command(target_joint_pos)
        else:
            arm_joint_pos = target_joint_pos[: self.n_arm_joints]
            hand_joint_pos = target_joint_pos[self.n_arm_joints :]
            self._publish_arm_joint_pos_command(arm_joint_pos)
            self._publish_hand_joint_pos_command(hand_joint_pos)
        self.target_joint_pos[:] = target_joint_pos

    def update_joint_pos(self):
        """
        Update self.curr_joint_pos
        """
        if not self.virtual_hardware:
            self._update_hand_joint_pos()
        else:
            self.curr_joint_pos = self.target_joint_pos.copy()

    def get_joint_pos(self, update=True):
        """
        Args:
            update: if False, it will return the current self.curr_joint_pos,
                    but it does not mean the self.curr_joint_pos is the latest received joint pos.
        """
        if update:
            self.update_joint_pos()
        return self.curr_joint_pos.copy()

    def get_target_joint_pos(self):
        return self.target_joint_pos.copy()

    def start_record_video(self, data_dir):
        msg = String()
        msg.data = str(data_dir)
        self.record_video_command_pub.publish(msg)

    def stop_record_video(self):
        msg = String()
        msg.data = "stop"
        self.record_video_command_pub.publish(msg)


def main():
    _require_ros_support()
    from threading import Thread

    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=None)
    node = Node("node_name")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot_real = RobotReal(node, virtual_hardware=False, use_high_freq_interp=True)
    robot_real.wait_for_initialization()

    # curr_joint_pos = robot_real.get_joint_pos()
    # target_joint_pos = curr_joint_pos.copy()
    # target_joint_pos[7:] = 0
    # robot_real.ctrl_joint_pos(target_joint_pos)

    while True:
        t1 = time.time()
        joint_pos = robot_real.get_joint_pos()
        print("current joint pos: ", joint_pos)
        print("get_joint_pos() time cost: ", time.time() - t1)
        time.sleep(0.2)

    # Cleanup on shutdown
    robot_real.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
