"""Backend-neutral data contracts for sensor input and execution flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from retargeting.core.types import RetargetingHandObservation


def _immutable_array(value: np.ndarray | None) -> np.ndarray | None:
    """Copy an optional array and prevent mutation through a result object.

    Args:
        value: Optional array-like runtime value.

    Returns:
        Independent read-only float array, or None.
    """
    if value is None:
        return None
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


def _immutable_diagnostics(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy scalar or array diagnostics into a read-only mapping.

    Args:
        values: Diagnostic values produced by runtime components.

    Returns:
        Read-only mapping with normalized keys and detached array values.
    """
    copied: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            array = np.asarray(value).copy()
            array.setflags(write=False)
            copied[str(key)] = array
        elif np.isscalar(value):
            copied[str(key)] = float(value)
        else:
            copied[str(key)] = value
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SensorHandSample:
    """Sensor-normalized hand data before robot-specific calibration or alignment."""

    keypoints_wrist: np.ndarray | None
    wrist_pose_sensor: np.ndarray | None
    raw: Any
    source_index: int | None = None
    timestamp: float | None = None
    presentation: Any | None = None

    def __post_init__(self) -> None:
        """Detach normalized numeric data from mutable sensor buffers.

        Args:
            None.

        Returns:
            None.
        """
        object.__setattr__(self, "keypoints_wrist", _immutable_array(self.keypoints_wrist))
        object.__setattr__(self, "wrist_pose_sensor", _immutable_array(self.wrist_pose_sensor))

    @property
    def has_hand(self) -> bool:
        """Return whether the sample contains the fields required for mapping.

        Args:
            None.

        Returns:
            True when both wrist-frame keypoints and a sensor wrist pose exist.
        """
        return self.keypoints_wrist is not None and self.wrist_pose_sensor is not None


@dataclass(frozen=True)
class RetargetedFrameResult:
    """Backend-neutral result for one successfully retargeted source frame."""

    observation: RetargetingHandObservation
    retargeted_qpos: np.ndarray
    diagnostics: Mapping[str, Any]
    source_index: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Copy mutable result fields before publishing the frame.

        Args:
            None.

        Returns:
            None.
        """
        observation = RetargetingHandObservation(
            keypoints_wrist=_immutable_array(self.observation.keypoints_wrist),
            wrist_pose_world=_immutable_array(self.observation.wrist_pose_world),
            timestamp=self.observation.timestamp,
            handedness=self.observation.handedness,
            keypoint_2d=_immutable_array(self.observation.keypoint_2d),
            raw=self.observation.raw,
        )
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "retargeted_qpos", _immutable_array(self.retargeted_qpos))
        object.__setattr__(self, "diagnostics", _immutable_diagnostics(self.diagnostics))


class ExecutionStatus(str, Enum):
    """State reached after processing one execution source frame."""

    EXECUTED = "executed"
    HELD = "held"
    WAITING_FOR_MAPPING = "waiting_for_mapping"


@dataclass(frozen=True)
class ExecutionStepResult:
    """Robot execution outcome for one source frame."""

    status: ExecutionStatus
    retargeted_frame: RetargetedFrameResult | None
    command_qpos: np.ndarray | None
    actual_qpos: np.ndarray | None
    diagnostics: Mapping[str, Any]
    source_index: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Copy mutable backend state before exposing the execution result.

        Args:
            None.

        Returns:
            None.
        """
        object.__setattr__(self, "command_qpos", _immutable_array(self.command_qpos))
        object.__setattr__(self, "actual_qpos", _immutable_array(self.actual_qpos))
        object.__setattr__(self, "diagnostics", _immutable_diagnostics(self.diagnostics))


@dataclass(frozen=True)
class FlowSummary:
    """Aggregate state returned after a pull-based execution run finishes."""

    source_frames_processed: int
    retarget_frames_processed: int
    command_periods_advanced: int
    cycles_completed: int
    last_result: ExecutionStepResult | None

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Build application-facing summary diagnostics.

        Args:
            None.

        Returns:
            Last-step diagnostics augmented with aggregate flow counters.
        """
        values = {} if self.last_result is None else dict(self.last_result.diagnostics)
        values.update(
            {
                "source_frames_processed": float(self.source_frames_processed),
                "retarget_frames_processed": float(self.retarget_frames_processed),
                "command_periods_advanced_total": float(self.command_periods_advanced),
                "cycles_completed": float(self.cycles_completed),
            }
        )
        return values
