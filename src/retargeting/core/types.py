"""Data returned by the retargeting algorithm boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class RetargetingHandObservation:
    """Canonical hand motion consumed by device-independent retargeting."""

    keypoints_wrist: np.ndarray
    wrist_pose_world: np.ndarray
    timestamp: float | None = None
    handedness: Literal["left", "right"] = "right"
    keypoint_2d: np.ndarray | None = None
    raw: Any = None


@dataclass(frozen=True)
class RetargetingResult:
    """Result of solving one canonical human-hand observation."""

    qpos: np.ndarray
    diagnostics: Mapping[str, float]
