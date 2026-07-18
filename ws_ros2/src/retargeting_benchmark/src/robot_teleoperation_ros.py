#!/usr/bin/env python3
"""ROS RGB callback adapter for the canonical teleoperation execution flow."""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from retargeting.config import (
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)
from retargeting.core import Retargeter
from retargeting.evaluation.robot_metrics import RobotBenchmark
from retargeting_ros.backends import RosCommandBackend
from robot_adaptor import RobotAdaptor
from robot_pinocchio import RobotPinocchio
from rviz_visualize import RvizVisualizer
from teleoperation.config import (
    load_detection_source_config,
    load_teleoperation_command_config,
    load_teleoperation_mode_config,
)
from teleoperation.flow import ExecutionFlow
from teleoperation.inputs.rgb.common import SingleHandDetector, decode_rgb_sample
from teleoperation.observation_mapping import StaticCalibrationMapper
from teleoperation.output import QposCommandLimiter, QposOutputFilter


class RobotTeleoperationRos(Node):
    """Decode ROS images and deliver normalized samples to one execution flow."""

    def __init__(self, node_name: str) -> None:
        """Construct ROS transport, retargeting components, and virtual backend.

        Args:
            node_name: ROS node name.

        Returns:
            None.
        """
        super().__init__(node_name)
        repo_root = Path(__file__).resolve().parents[4]
        profile_source = repo_root / "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
        profile_config = load_retargeting_profile_config(profile_source)
        detection_config = load_detection_source_config(repo_root / "configs/detection_sources/rgb.yaml")
        robot_config = load_robot_config(repo_root / profile_config.robot)
        method_config = load_retargeting_config(repo_root / profile_config.method)
        solver_config = load_solver_config(repo_root / "configs/solvers/nlopt_slsqp.yaml")
        mode_config = load_teleoperation_mode_config(repo_root / "configs/teleoperation_modes/simulation.yaml")
        command_config = load_teleoperation_command_config(profile_source, robot_config=robot_config)
        robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
        robot_adaptor = RobotAdaptor(robot_model, actuated_joints_name=list(robot_config.actuated_joints))
        retargeter = Retargeter(robot_adaptor, robot_config, profile_config, method_config, solver_config)
        output_filter = QposOutputFilter(retargeter.qpos_init, mode_config)
        mapper = StaticCalibrationMapper(detection_config, robot_config.human_hand_scale)
        evaluator = RobotBenchmark(robot_adaptor, robot_config.benchmark)
        self.robot_model = robot_model
        self.robot_adaptor = robot_adaptor
        self.rviz_visualizer = RvizVisualizer(node=self)

        def publish_robot(qpos: np.ndarray) -> np.ndarray:
            """Publish one virtual robot command period to RViz.

            Args:
                qpos: Retargeted positions in actuated-joint order.

            Returns:
                Same positions as the virtual measured state.
            """
            qpos_dof = self.robot_adaptor.forward_qpos(qpos)
            self.rviz_visualizer.publish_robot_joint_states(
                joints_name=self.robot_model.joint_names,
                joints_pos=qpos_dof,
            )
            return qpos

        backend = RosCommandBackend(
            initial_qpos=robot_config.initial_qpos,
            control_period=0.05,
            execute_callback=publish_robot,
        )
        joint_limits = robot_adaptor.backward_qpos(robot_model.joint_limits)
        command_policy = QposCommandLimiter(
            initial_qpos=backend.get_target_joint_pos(),
            max_joint_speed=np.asarray(command_config.max_joint_speed, dtype=float),
            command_hz=20.0,
            lower=joint_limits[:, 0],
            upper=joint_limits[:, 1],
        )
        self.flow = ExecutionFlow(
            input=None,
            observation_mapper=mapper,
            retargeter=retargeter,
            output_filter=output_filter,
            evaluator=evaluator,
            command_policy=command_policy,
            backend=backend,
            realtime=False,
        )
        self.detector = SingleHandDetector("Right")
        self.cv_bridge = CvBridge()
        self.camera_K: np.ndarray | None = None
        self.get_logger().info("Waiting for image stream ...")
        self.image_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 1)
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.cam_info_callback,
            1,
        )

    def cam_info_callback(self, msg: CameraInfo) -> None:
        """Capture camera intrinsics from the first ROS camera-info message.

        Args:
            msg: ROS camera information message.

        Returns:
            None.
        """
        if self.camera_K is None:
            self.camera_K = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def image_callback(self, msg: Image) -> None:
        """Convert one ROS image message and process it through the shared flow.

        Args:
            msg: ROS color image message.

        Returns:
            None.
        """
        try:
            color_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as error:
            self.get_logger().error(f"Could not convert image: {error}")
            return
        self.process(color_img)

    def process(self, color_img: np.ndarray) -> None:
        """Decode and execute one externally delivered RGB frame.

        Args:
            color_img: OpenCV image from the ROS callback.

        Returns:
            None.
        """
        if self.camera_K is None:
            return
        sample = decode_rgb_sample(self.detector, color_img, self.camera_K)
        result = self.flow.step(sample)
        frame = result.retargeted_frame
        if frame is not None:
            observation = frame.observation
            self.rviz_visualizer.publish_hand_detection_results(
                observation.keypoints_wrist,
                observation.wrist_pose_world,
                frame_id="world",
            )
            print(frame.retargeted_qpos)
        if sample.presentation is not None:
            rendered = self.detector.draw_skeleton_on_image(color_img, sample.presentation, style="default")
            cv2.imshow("detection result", rendered)
            cv2.waitKey(1)


def main() -> None:
    """Run the ROS callback adapter until shutdown.

    Args:
        None.

    Returns:
        None.
    """
    rclpy.init(args=None)
    teleoperation = RobotTeleoperationRos("robot_teleoperation")
    rclpy.spin(teleoperation)
    teleoperation.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
