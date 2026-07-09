import sys
import time
from pathlib import Path
from threading import Thread

import numpy as np
import rclpy
from pynput import keyboard
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_adaptor import RobotAdaptor
from robot_control import RobotControl
from robot_pinocchio import RobotPinocchio
from utils.utils_keyboard import KeyboardListener


class RobotControlMain:
    def __init__(self):
        # --------- hyper-parameters ---------
        repo_root = Path(__file__).resolve().parents[4]
        urdf_file_name = str(repo_root / "assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf")
        actuated_joints_name = [f"panda_joint{i+1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
        self.joint_vel_max = np.array([0.1] * 7 + [0.5] * 16)
        self.use_hardware = True
        self.use_virtual_hardware = False
        self.use_high_freq_interp = True
        self.use_ros = self.use_hardware
        # -------------------------------------

        if self.use_ros:
            rclpy.init(args=None)
            self.node = Node("main_node")
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()
        else:
            self.node = None

        self.robot_model = RobotPinocchio(
            robot_file_path=urdf_file_name,
            robot_file_type="urdf",
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            actuated_joints_name=actuated_joints_name,
        )
        self.robot_control = RobotControl(
            robot_model=self.robot_model,
            robot_adaptor=self.robot_adaptor,
            use_hardware=self.use_hardware,
            use_virtual_hardware=self.use_virtual_hardware,
            use_high_freq_interp=self.use_high_freq_interp,
            node=self.node,
        )

        self.keyboard_listener = KeyboardListener()

    def main(self):
        self.robot_control.move_to_joint_pos(self.robot_control.init_joint_pos)

        self.keyboard_listener.start_keyboard_listening_thread()

        while True:
            pressed_keys = self.keyboard_listener.pressed_keys
            tcp_motion = self.robot_control.keys_to_tcp_motion(pressed_keys)
            if np.any(tcp_motion != 0):
                self.robot_control.cartesian_move(tcp_link_name="panda_link8", tcp_motion=tcp_motion)

            self.robot_control.step()

        # ------------ end ------------
        self.spin_thread.join()
        if self.use_ros:
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    robot_control = RobotControlMain()
    robot_control.main()
