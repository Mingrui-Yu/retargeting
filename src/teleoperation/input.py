"""Adapters that convert device-specific frames into canonical hand observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retargeting.config import DetectionSourceConfig
from retargeting.inputs import HandObservation
from retargeting.utils.utils_calc import posRotMat2Isometry3d, sciR

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


@dataclass(frozen=True)
class DetectedHand:
    """Canonical observation plus data needed only by input-side presentation."""

    observation: HandObservation
    keypoint_2d: np.ndarray | None = None


class HandObservationAdapter:
    """Normalize RGB or AVP detector output into robot-world hand observations."""

    def __init__(self, config: DetectionSourceConfig, human_hand_scale: float) -> None:
        """Create an input adapter for one detector source and target embodiment.

        Args:
            config: Detector selection, calibration, and relative-alignment settings.
            human_hand_scale: Scale converting detector hand geometry to target units.

        Returns:
            None.
        """
        config.validate()
        self.config = config
        self.human_hand_scale = float(human_hand_scale)
        self.robot_initial_wrist_pose: np.ndarray | None = None
        self.detection_initial_wrist_pose: np.ndarray | None = None
        if config.input_device == "rgb":
            from retargeting.inputs.rgb import SingleHandDetector

            self.detector = SingleHandDetector("Right")
        elif config.input_device == "avp":
            from retargeting.inputs.avp import AvpDetector

            self.detector = AvpDetector()
        else:
            raise NotImplementedError(f"Unsupported input_device: {config.input_device}")

    def set_robot_initial_wrist_pose(self, pose: np.ndarray) -> None:
        """Store the robot pose used by relative input alignment.

        Args:
            pose: Initial robot wrist pose in robot world coordinates.

        Returns:
            None.
        """
        self.robot_initial_wrist_pose = np.asarray(pose, dtype=float).copy()

    def set_detection_initial_wrist_pose(self, pose: np.ndarray) -> None:
        """Store the calibrated detector pose used by relative input alignment.

        Args:
            pose: Initial detected wrist pose in robot world coordinates.

        Returns:
            None.
        """
        self.detection_initial_wrist_pose = np.asarray(pose, dtype=float).copy()

    def pose_from_detection_world_to_robot_world(self, pose: np.ndarray) -> np.ndarray:
        """Transform a detector-world wrist pose into robot-world coordinates.

        Args:
            pose: Wrist pose in the detector world frame.

        Returns:
            Wrist pose in the robot world frame.
        """
        transform = posRotMat2Isometry3d(
            pos=[0, 0, 0],
            rot_mat=sciR.from_euler("xyz", self.config.rotation_euler_xyz_deg, degrees=True).as_matrix(),
        )
        result = transform @ np.asarray(pose, dtype=float)
        result[:3, 3] += np.asarray(self.config.translation)
        return result

    def detect(self, sensor_data: Any, camera_K: np.ndarray | None = None) -> DetectedHand | None:
        """Convert one raw device frame into a canonical hand observation.

        Args:
            sensor_data: Raw frame accepted by the configured detector.
            camera_K: Camera intrinsics required by RGB detection.

        Returns:
            Canonical detected hand, or None when no hand is detected.
        """
        if self.config.input_device == "rgb":
            if cv2 is None:
                raise ModuleNotFoundError("cv2 is required for RGB teleoperation, but it is not installed.")
            if camera_K is None:
                raise ValueError("RGB input requires camera_K.")
            _, keypoints, keypoint_2d, wrist_pose = self.detector.detect(sensor_data, camera_K)
            if keypoints is None:
                return None
            observation = HandObservation(
                keypoints_wrist=keypoints * self.human_hand_scale,
                wrist_pose_world=self.pose_from_detection_world_to_robot_world(wrist_pose),
                keypoint_2d=keypoint_2d,
                raw=sensor_data,
            )
            return DetectedHand(observation=observation, keypoint_2d=keypoint_2d)
        _, keypoints, _, wrist_pose = self.detector.detect(sensor_data)
        if keypoints is None:
            return None
        wrist_pose_world = self.pose_from_detection_world_to_robot_world(wrist_pose)
        if self.config.use_relative_wrist_alignment:
            if self.robot_initial_wrist_pose is None or self.detection_initial_wrist_pose is None:
                raise ValueError("Relative wrist alignment requires initial robot and detector wrist poses.")
            wrist_pose_world[:3, 3] += (
                self.robot_initial_wrist_pose[:3, 3] - self.detection_initial_wrist_pose[:3, 3]
            )
        return DetectedHand(
            observation=HandObservation(
                keypoints_wrist=keypoints * self.human_hand_scale,
                wrist_pose_world=wrist_pose_world,
                raw=sensor_data,
            )
        )

    def draw_detection(self, image: np.ndarray, detected_hand: DetectedHand) -> np.ndarray:
        """Render a detector overlay when the configured source supports it.

        Args:
            image: Raw RGB image used for the detection.
            detected_hand: Detection result containing 2D keypoints.

        Returns:
            Annotated RGB image.
        """
        if self.config.input_device != "rgb":
            return image
        return self.detector.draw_skeleton_on_image(image, detected_hand.keypoint_2d, style="default")
