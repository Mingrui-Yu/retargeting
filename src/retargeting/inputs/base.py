from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class HandObservation:
    keypoints_wrist: np.ndarray
    wrist_pose_world: np.ndarray
    timestamp: float | None = None
    handedness: Literal["left", "right"] = "right"


class HandInput(Protocol):
    """Source of canonical human-hand observations."""

    def get_observation(self) -> HandObservation:
        ...
