"""Computation-only helpers for retargeting canonical observation sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from retargeting.core.types import RetargetingHandObservation, RetargetingResult


class ObservationRetargeter(Protocol):
    """Solver boundary required by canonical sequence retargeting."""

    def solve(
        self,
        observation: RetargetingHandObservation,
        previous_qpos: np.ndarray | None = None,
    ) -> RetargetingResult:
        """Solve one canonical observation.

        Args:
            observation: Device-independent hand observation.
            previous_qpos: Optional temporal reference for the objective.

        Returns:
            Raw robot qpos and computation diagnostics.
        """
        ...


def retarget_observation_sequence(
    observations: Iterable[RetargetingHandObservation],
    retargeter: ObservationRetargeter,
    previous_qpos: np.ndarray | None = None,
) -> tuple[RetargetingResult, ...]:
    """Retarget canonical observations without input, file, or presentation I/O.

    Args:
        observations: Canonical hand observations in processing order.
        retargeter: Core solver implementing the observation boundary.
        previous_qpos: Optional temporal reference for the first observation.

    Returns:
        Raw qpos and pure computation diagnostics for each observation.
    """
    temporal_reference = None if previous_qpos is None else np.asarray(previous_qpos, dtype=float).copy()
    results: list[RetargetingResult] = []
    for observation in observations:
        result = retargeter.solve(observation, previous_qpos=temporal_reference)
        results.append(result)
        temporal_reference = np.asarray(result.qpos, dtype=float).copy()
    return tuple(results)
