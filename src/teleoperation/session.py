"""Application service that connects hand input, retargeting, and output filtering."""

from __future__ import annotations

from typing import Any

import numpy as np

from retargeting.config import (
    DetectionSourceConfig,
    RetargetingConfig,
    RetargetingProfileConfig,
    RobotConfig,
    SolverConfig,
    TeleoperationModeConfig,
)
from retargeting.core import Retargeter
from retargeting.inputs import HandObservation
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.evaluation.robot_metrics import RobotBenchmark
from mr_utils.utils_calc import transformPositions
from teleoperation.input import HandObservationAdapter
from teleoperation.output import QposOutputFilter

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


class TeleoperationSession:
    """Coordinate input adaptation, a pure retargeter, and execution filtering."""

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        robot_config: RobotConfig,
        profile_config: RetargetingProfileConfig,
        method_config: RetargetingConfig,
        detection_source_config: DetectionSourceConfig,
        teleoperation_mode_config: TeleoperationModeConfig,
        solver_config: SolverConfig | None = None,
        evaluate: bool = True,
    ) -> None:
        """Create a teleoperation composition around the retargeting core.

        Args:
            robot_adaptor: Kinematics adapter for the target robot.
            robot_config: Target robot embodiment configuration.
            profile_config: Robot-method objective and output-limit configuration.
            method_config: Retargeting method configuration.
            detection_source_config: Device input and calibration configuration.
            teleoperation_mode_config: Runtime output filtering configuration.
            solver_config: Optional numerical solver configuration.
            evaluate: Whether to compute benchmark diagnostics for each result.

        Returns:
            None.
        """
        if teleoperation_mode_config is None:
            raise ValueError("TeleoperationSession requires teleoperation_mode_config.")
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model
        self.robot_config = robot_config
        self.profile_config = profile_config
        self.retargeter = Retargeter(
            robot_adaptor=robot_adaptor,
            robot_config=robot_config,
            profile_config=profile_config,
            method_config=method_config,
            solver_config=solver_config,
        )
        self.input_adapter = HandObservationAdapter(detection_source_config, robot_config.human_hand_scale)
        self.output_filter = QposOutputFilter(self.retargeter.qpos_init, teleoperation_mode_config)
        self.evaluator = (
            RobotBenchmark(robot_adaptor=robot_adaptor, benchmark_config=robot_config.benchmark) if evaluate else None
        )

    @property
    def detector(self) -> Any:
        """Expose the device detector for transitional live-entrypoint compatibility.

        Args:
            None.

        Returns:
            Detector instance owned by the input adapter.
        """
        return self.input_adapter.detector

    @property
    def qpos_last(self) -> np.ndarray:
        """Return the previous execution command retained by the output layer.

        Args:
            None.

        Returns:
            Previous filtered qpos command.
        """
        return self.output_filter.previous_qpos

    def set_robot_init_wrist_pose(self, pose: np.ndarray) -> None:
        """Set the robot pose required by relative input alignment.

        Args:
            pose: Initial robot wrist pose in robot world coordinates.

        Returns:
            None.
        """
        self.input_adapter.set_robot_initial_wrist_pose(pose)

    def set_detection_source_init_wrist_pose(self, pose: np.ndarray) -> None:
        """Set the detector pose required by relative input alignment.

        Args:
            pose: Initial detected wrist pose in robot world coordinates.

        Returns:
            None.
        """
        self.input_adapter.set_detection_initial_wrist_pose(pose)

    def pose_from_detection_world_to_robot_world(self, pose: np.ndarray) -> np.ndarray:
        """Expose calibrated pose conversion for entrypoint setup code.

        Args:
            pose: Wrist pose in detector-world coordinates.

        Returns:
            Wrist pose in robot-world coordinates.
        """
        return self.input_adapter.pose_from_detection_world_to_robot_world(pose)

    def _evaluate(self, qpos: np.ndarray, observation: HandObservation) -> dict[str, float]:
        """Compute optional benchmark metrics outside the retargeting solver.

        Args:
            qpos: Raw optimized robot configuration.
            observation: Canonical observation used for this solve.

        Returns:
            Mapping of benchmark metrics, or an empty mapping when disabled.
        """
        if self.evaluator is None:
            return {}
        hand_kps_world = transformPositions(
            observation.keypoints_wrist, target_frame_pose_inv=observation.wrist_pose_world
        )
        return {
            "position_err": self.evaluator.position_error(qpos, hand_kps_world, 1),
            "orientation_err": self.evaluator.orientation_error(qpos, hand_kps_world, 1),
            "relative_position_err": self.evaluator.relative_position_error(qpos, hand_kps_world, 1),
            "relative_position_to_wrist_err": self.evaluator.relative_position_to_wrist_error(qpos, hand_kps_world, 1),
        }

    def retarget_observation(self, observation: HandObservation) -> tuple[np.ndarray, dict[str, float]]:
        """Retarget a canonical observation and apply execution output processing.

        Args:
            observation: Device-independent hand observation in robot world coordinates.

        Returns:
            Filtered qpos command and algorithm/evaluation diagnostics.
        """
        result = self.retargeter.solve(observation, previous_qpos=self.output_filter.previous_qpos)
        diagnostics = {**self._evaluate(result.qpos, observation), **result.diagnostics}
        qpos = self.output_filter.apply(result.qpos)
        # Preserve the original temporal behavior: the objective references the
        # previous execution command, not an unobservable pre-filter solution.
        self.retargeter.previous_qpos = qpos.copy()
        return qpos, diagnostics

    def hand_retarget(
        self, hand_kps_in_wrist: np.ndarray, wrist_pose_in_world: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Retarget already-normalized hand geometry for compatibility callers.

        Args:
            hand_kps_in_wrist: Hand keypoints in the detected wrist frame.
            wrist_pose_in_world: Wrist pose in robot world coordinates.

        Returns:
            Filtered qpos command and algorithm/evaluation diagnostics.
        """
        return self.retarget_observation(
            HandObservation(
                keypoints_wrist=np.asarray(hand_kps_in_wrist, dtype=float),
                wrist_pose_world=np.asarray(wrist_pose_in_world, dtype=float),
            )
        )

    def detect_observation(
        self, sensor_data: Any, camera_K: np.ndarray | None = None
    ) -> HandObservation | None:
        """Adapt one raw device frame into a canonical hand observation.

        Args:
            sensor_data: Raw frame for the configured input adapter.
            camera_K: Optional RGB camera intrinsics.

        Returns:
            Canonical observation, or None when no hand is detected.
        """
        detected = self.input_adapter.detect(sensor_data, camera_K=camera_K)
        return None if detected is None else detected.observation

    def retarget_input(
        self,
        sensor_data: Any,
        camera_K: np.ndarray | None = None,
        show_detection: bool = False,
    ) -> tuple[HandObservation | None, np.ndarray | None, dict[str, float] | None]:
        """Run input adaptation, core retargeting, and output filtering for one frame.

        Args:
            sensor_data: Raw frame for the configured input adapter.
            camera_K: Optional RGB camera intrinsics.
            show_detection: Whether to render the RGB detector overlay.

        Returns:
            Observation, filtered qpos command, and diagnostics; all are None when no hand is detected.
        """
        detected = self.input_adapter.detect(sensor_data, camera_K=camera_K)
        if detected is None:
            if show_detection and self.input_adapter.config.input_device == "rgb":
                if cv2 is None:
                    raise ModuleNotFoundError("cv2 is required for RGB teleoperation, but it is not installed.")
                cv2.imshow("detection result", sensor_data)
                cv2.waitKey(1)
            return None, None, None
        qpos, diagnostics = self.retarget_observation(detected.observation)
        if show_detection and self.input_adapter.config.input_device == "rgb":
            if cv2 is None:
                raise ModuleNotFoundError("cv2 is required for RGB teleoperation, but it is not installed.")
            cv2.imshow("detection result", self.input_adapter.draw_detection(sensor_data, detected))
            cv2.waitKey(1)
        return detected.observation, qpos, diagnostics
