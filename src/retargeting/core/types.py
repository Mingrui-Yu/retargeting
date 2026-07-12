"""Data returned by the retargeting algorithm boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RetargetingResult:
    """Result of solving one canonical human-hand observation."""

    qpos: np.ndarray
    diagnostics: Mapping[str, float]
