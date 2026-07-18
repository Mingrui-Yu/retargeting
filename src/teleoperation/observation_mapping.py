"""Map sensor-normalized hand samples into the retargeting core frame."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mr_utils.utils_calc import posRotMat2Isometry3d, sciR
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.core.types import RetargetingHandObservation
from teleoperation.config import DetectionSourceConfig
from teleoperation.types import SensorHandSample


class HandObservationMapper(Protocol):
    """Robot-aware calibration boundary between sensor and core observations."""

    def initialize(self, sample: SensorHandSample, robot_qpos: np.ndarray) -> bool:
        """Try to initialize mapping state from one sample and robot state.

        Args:
            sample: Sensor-normalized hand sample.
            robot_qpos: Current measured robot positions in actuated-joint order.

        Returns:
            True when subsequent mapping can proceed.
        """
        ...

    def map(self, sample: SensorHandSample) -> RetargetingHandObservation | None:
        """Map one sample into the canonical robot-world observation.

        Args:
            sample: Sensor-normalized hand sample.

        Returns:
            Canonical observation, or None for a missing detection.
        """
        ...

    def reset(self) -> None:
        """Clear mapping state before a new execution cycle.

        Args:
            None.

        Returns:
            None.
        """
        ...


class StaticCalibrationMapper:
    """Apply a configured fixed sensor-world to robot-world calibration."""

    def __init__(self, config: DetectionSourceConfig, human_hand_scale: float) -> None:
        """Create a fixed calibration mapper.

        Args:
            config: Sensor calibration expressed by the existing detection source schema.
            human_hand_scale: Scale from normalized human geometry to target units.

        Returns:
            None.
        """
        config.validate()
        self.config = config
        self.human_hand_scale = float(human_hand_scale)
        self._sensor_to_robot = posRotMat2Isometry3d(
            pos=[0.0, 0.0, 0.0],
            rot_mat=sciR.from_euler("xyz", config.rotation_euler_xyz_deg, degrees=True).as_matrix(),
        )

    def initialize(self, sample: SensorHandSample, robot_qpos: np.ndarray) -> bool:
        """Confirm that the first sample has enough data for fixed calibration.

        Args:
            sample: Sensor-normalized hand sample.
            robot_qpos: Current robot state, unused by fixed calibration.

        Returns:
            True when the sample includes hand keypoints and a wrist pose.
        """
        del robot_qpos
        return sample.has_hand

    def pose_from_sensor_world_to_robot_world(self, pose: np.ndarray) -> np.ndarray:
        """Apply the configured sensor-world calibration to a wrist pose.

        Args:
            pose: Wrist pose in the sensor world frame.

        Returns:
            Wrist pose in the robot world frame.
        """
        result = self._sensor_to_robot @ np.asarray(pose, dtype=float)
        result[:3, 3] += np.asarray(self.config.translation, dtype=float)
        return result

    def map(self, sample: SensorHandSample) -> RetargetingHandObservation | None:
        """Apply fixed calibration and target-specific human-hand scaling.

        Args:
            sample: Sensor-normalized hand sample.

        Returns:
            Canonical robot-world observation, or None for missing hand data.
        """
        if not sample.has_hand:
            return None
        presentation = sample.presentation
        keypoint_2d = presentation if isinstance(presentation, np.ndarray) else None
        return RetargetingHandObservation(
            keypoints_wrist=np.asarray(sample.keypoints_wrist, dtype=float) * self.human_hand_scale,
            wrist_pose_world=self.pose_from_sensor_world_to_robot_world(sample.wrist_pose_sensor),
            timestamp=sample.timestamp,
            keypoint_2d=keypoint_2d,
            raw=sample.raw,
        )

    def reset(self) -> None:
        """Reset the stateless fixed calibration mapper.

        Args:
            None.

        Returns:
            None.
        """


class IdentityHandObservationMapper:
    """Treat a sample wrist pose as an already calibrated robot-world pose."""

    def __init__(self, human_hand_scale: float = 1.0) -> None:
        """Create an identity mapper with optional geometry scaling.

        Args:
            human_hand_scale: Scale applied to sensor-normalized keypoints.

        Returns:
            None.
        """
        self.human_hand_scale = float(human_hand_scale)

    def initialize(self, sample: SensorHandSample, robot_qpos: np.ndarray) -> bool:
        """Confirm that a callback sample contains a complete hand observation.

        Args:
            sample: Sensor-normalized hand sample.
            robot_qpos: Current robot state, unused by identity mapping.

        Returns:
            True when the sample contains both canonical hand fields.
        """
        del robot_qpos
        return sample.has_hand

    def map(self, sample: SensorHandSample) -> RetargetingHandObservation | None:
        """Copy a complete sample into the canonical core observation.

        Args:
            sample: Sensor-normalized hand sample.

        Returns:
            Canonical observation, or None for a missing hand sample.
        """
        if not sample.has_hand:
            return None
        return RetargetingHandObservation(
            keypoints_wrist=np.asarray(sample.keypoints_wrist, dtype=float) * self.human_hand_scale,
            wrist_pose_world=np.asarray(sample.wrist_pose_sensor, dtype=float),
            timestamp=sample.timestamp,
            keypoint_2d=sample.presentation if isinstance(sample.presentation, np.ndarray) else None,
            raw=sample.raw,
        )

    def reset(self) -> None:
        """Reset the stateless identity mapper.

        Args:
            None.

        Returns:
            None.
        """


class AvpRelativeWristMapper(StaticCalibrationMapper):
    """Align calibrated AVP wrist translation to the measured robot wrist."""

    def __init__(
        self,
        config: DetectionSourceConfig,
        human_hand_scale: float,
        robot_adaptor: RobotAdaptor,
        robot_model: Any,
        wrist_frame_name: str,
    ) -> None:
        """Create an AVP mapper tied to one target robot kinematics model.

        Args:
            config: AVP sensor-world calibration and relative-alignment flag.
            human_hand_scale: Scale from AVP hand geometry to target units.
            robot_adaptor: Actuated-to-model qpos adapter.
            robot_model: Kinematics model exposing ``get_frame_pose``.
            wrist_frame_name: Robot wrist frame used as the alignment origin.

        Returns:
            None.
        """
        super().__init__(config, human_hand_scale)
        if config.input_device != "avp":
            raise ValueError("AvpRelativeWristMapper requires an avp detection source config.")
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_model
        self.wrist_frame_name = str(wrist_frame_name)
        self._robot_initial_wrist_pose: np.ndarray | None = None
        self._sensor_initial_wrist_pose: np.ndarray | None = None

    def initialize(self, sample: SensorHandSample, robot_qpos: np.ndarray) -> bool:
        """Capture relative wrist origins using the current measured robot state.

        Args:
            sample: AVP sample selected as the sensor alignment origin.
            robot_qpos: Current measured actuated-joint positions.

        Returns:
            True when the sample contains a wrist pose and origins were captured.
        """
        if not sample.has_hand:
            return False
        model_qpos = self.robot_adaptor.forward_qpos(np.asarray(robot_qpos, dtype=float))
        self._robot_initial_wrist_pose = np.asarray(
            self.robot_model.get_frame_pose(self.wrist_frame_name, qpos=model_qpos),
            dtype=float,
        ).copy()
        self._sensor_initial_wrist_pose = self.pose_from_sensor_world_to_robot_world(sample.wrist_pose_sensor)
        return True

    def map(self, sample: SensorHandSample) -> RetargetingHandObservation | None:
        """Map AVP hand data using the captured relative wrist translation.

        Args:
            sample: Sensor-normalized AVP sample.

        Returns:
            Canonical robot-world observation, or None for missing hand data.
        """
        observation = super().map(sample)
        if observation is None or not self.config.use_relative_wrist_alignment:
            return observation
        if self._robot_initial_wrist_pose is None or self._sensor_initial_wrist_pose is None:
            raise RuntimeError("AVP relative wrist mapping requires successful initialization.")
        wrist_pose_world = np.asarray(observation.wrist_pose_world, dtype=float).copy()
        wrist_pose_world[:3, 3] += (
            self._robot_initial_wrist_pose[:3, 3] - self._sensor_initial_wrist_pose[:3, 3]
        )
        return RetargetingHandObservation(
            keypoints_wrist=observation.keypoints_wrist,
            wrist_pose_world=wrist_pose_world,
            timestamp=observation.timestamp,
            handedness=observation.handedness,
            keypoint_2d=observation.keypoint_2d,
            raw=observation.raw,
        )

    def reset(self) -> None:
        """Clear AVP and robot wrist origins before a new cycle.

        Args:
            None.

        Returns:
            None.
        """
        self._robot_initial_wrist_pose = None
        self._sensor_initial_wrist_pose = None
