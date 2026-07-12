#!/usr/bin/env python3
import numpy as np
import time
import cv2
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from robot_teleoperation import RobotTeleoperation
from robot_adaptor import RobotAdaptor
from robot_pinocchio import RobotPinocchio
from rviz_visualize import RvizVisualizer
from retargeting.config import (
    load_detection_source_config,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)


class RobotTeleoperationRos(Node, RobotTeleoperation):
    """
    RobotTeleoperation with ROS interface.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)
        repo_root = Path(__file__).resolve().parents[4]
        profile_config = load_retargeting_profile_config(
            repo_root / "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
        )
        detection_source_config = load_detection_source_config(repo_root / "configs/detection_sources/rgb.yaml")
        robot_config = load_robot_config(repo_root / profile_config.robot)
        retargeting_config = load_retargeting_config(repo_root / profile_config.method)
        solver_config = load_solver_config(repo_root / "configs/solvers/nlopt_slsqp.yaml")
        robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
        robot_adaptor = RobotAdaptor(robot_model, actuated_joints_name=list(robot_config.actuated_joints))
        RobotTeleoperation.__init__(
            self,
            robot_adaptor=robot_adaptor,
            robot_control=None,
            robot_config=robot_config,
            profile_config=profile_config,
            method_config=retargeting_config,
            detection_source_config=detection_source_config,
            solver_config=solver_config,
        )

        self.cv_bridge = CvBridge()
        self.camera_K = None
        self.rviz_visualizer = RvizVisualizer(node=self)

        self.get_logger().info("Waiting for image stream ...")
        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            1,
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.cam_info_callback,
            1,
        )

    def cam_info_callback(self, msg):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        self.process(color_img)

    def process(self, color_img):
        if self.camera_K is None:
            return

        observation, qpos, _ = self.retarget_input(color_img, camera_K=self.camera_K, show_detection=True)
        if observation is not None:
            # visualize the human hand in rviz
            self.rviz_visualizer.publish_hand_detection_results(
                observation.hand_kps_in_wrist, observation.wrist_pose_in_world, frame_id="world"
            )

            # visualize the robot hand in rviz
            joints_name = self.robot_model.joint_names
            qpos_dof = self.robot_adaptor.forward_qpos(qpos)
            self.rviz_visualizer.publish_robot_joint_states(
                joints_name=joints_name, joints_pos=qpos_dof
            )

        print(qpos)


def main():
    node_name = "robot_teleoperation"

    rclpy.init(args=None)
    teleoperation = RobotTeleoperationRos(node_name)
    rclpy.spin(teleoperation)

    # Cleanup on shutdown
    teleoperation.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
