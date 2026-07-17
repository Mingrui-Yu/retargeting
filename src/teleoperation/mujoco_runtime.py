"""Online composition of one retargeting frame and one MuJoCo command period."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from retargeting.backends.base import RobotBackend
from retargeting.inputs import HandObservation
from teleoperation.output import QposCommandLimiter
from teleoperation.session import TeleoperationSession


@dataclass(frozen=True)
class MujocoStepResult:
    """State and diagnostics produced by one online simulation frame."""

    observation: HandObservation | None
    requested_qpos: np.ndarray | None
    command_qpos: np.ndarray
    actual_qpos: np.ndarray
    diagnostics: dict[str, float]


class MujocoTeleoperationRuntime:
    """Retarget and execute each input frame immediately at a fixed command rate."""

    def __init__(
        self,
        session: TeleoperationSession,
        backend: RobotBackend,
        command_limiter: QposCommandLimiter,
        realtime: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create an online retargeting-to-MuJoCo runtime.

        Args:
            session: Input-independent retargeting session.
            backend: Robot backend advanced once for every processed input frame.
            command_limiter: Per-frame range and velocity command policy.
            realtime: Whether to wait for the remainder of each wall-clock period.
            clock: Monotonic clock dependency used for timing and tests.
            sleep: Wall-clock sleep dependency used for real-time pacing.

        Returns:
            None.
        """
        self.session = session
        self.backend = backend
        self.command_limiter = command_limiter
        self.realtime = bool(realtime)
        self._clock = clock
        self._sleep = sleep
        self.frame_count = 0
        expected_period = 1.0 / self.command_limiter.command_hz
        if not np.isclose(self.backend.control_period, expected_period, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Backend and command limiter periods differ: "
                f"backend={self.backend.control_period}, limiter={expected_period}."
            )

    @property
    def control_period(self) -> float:
        """Return the fixed simulated duration advanced for each frame.

        Args:
            None.

        Returns:
            Command period in seconds.
        """
        return self.backend.control_period

    def _advance(
        self,
        started_at: float,
        observation: HandObservation | None,
        requested_qpos: np.ndarray | None,
        command_qpos: np.ndarray,
        diagnostics: dict[str, float] | None = None,
    ) -> MujocoStepResult:
        """Apply one command, advance physics, and enforce optional pacing.

        Args:
            started_at: Wall-clock timestamp captured before frame processing.
            observation: Current canonical observation, or None for a hold frame.
            requested_qpos: Raw retargeting result, or None for a hold frame.
            command_qpos: Range- and velocity-limited execution command.
            diagnostics: Retargeting diagnostics collected before simulation.

        Returns:
            State and merged diagnostics for the completed online frame.
        """
        applied_qpos = self.backend.ctrl_joint_pos(command_qpos)
        self.backend.step()
        compute_elapsed = self._clock() - started_at
        overrun = max(0.0, compute_elapsed - self.control_period)
        sleep_duration = max(0.0, self.control_period - compute_elapsed) if self.realtime else 0.0
        if sleep_duration > 0:
            self._sleep(sleep_duration)
        wall_elapsed = self._clock() - started_at
        merged_diagnostics = dict(diagnostics or {})
        backend_diagnostics = getattr(self.backend, "get_diagnostics", lambda: {})()
        merged_diagnostics.update({str(key): float(value) for key, value in backend_diagnostics.items()})
        merged_diagnostics.update(
            {
                "runtime_compute_time": float(compute_elapsed),
                "runtime_wall_time": float(wall_elapsed),
                "runtime_overrun": float(overrun),
            }
        )
        self.frame_count += 1
        return MujocoStepResult(
            observation=observation,
            requested_qpos=None if requested_qpos is None else requested_qpos.copy(),
            command_qpos=np.asarray(applied_qpos, dtype=float).copy(),
            actual_qpos=self.backend.get_joint_pos(),
            diagnostics=merged_diagnostics,
        )

    def step_observation(self, observation: HandObservation) -> MujocoStepResult:
        """Retarget and execute one canonical observation immediately.

        Args:
            observation: Current human-hand observation.

        Returns:
            Completed online simulation frame.
        """
        started_at = self._clock()
        requested_qpos, diagnostics = self.session.retarget_observation(observation)
        command_qpos = self.command_limiter.apply(requested_qpos)
        return self._advance(started_at, observation, requested_qpos, command_qpos, diagnostics)

    def step_sensor_data(self, sensor_data: Any, camera_K: np.ndarray | None = None) -> MujocoStepResult:
        """Detect, retarget, and execute one raw sensor frame.

        Args:
            sensor_data: Raw frame accepted by the configured input adapter.
            camera_K: Optional camera intrinsics for RGB input.

        Returns:
            Completed simulation frame; missing detections hold the previous command.
        """
        started_at = self._clock()
        observation = self.session.detect_observation(sensor_data, camera_K=camera_K)
        if observation is None:
            return self._advance(
                started_at,
                observation=None,
                requested_qpos=None,
                command_qpos=self.backend.get_target_joint_pos(),
            )
        requested_qpos, diagnostics = self.session.retarget_observation(observation)
        command_qpos = self.command_limiter.apply(requested_qpos)
        return self._advance(started_at, observation, requested_qpos, command_qpos, diagnostics)

    def step_hold(self) -> MujocoStepResult:
        """Advance one command period while retaining the previous target.

        Args:
            None.

        Returns:
            Completed hold frame.
        """
        return self._advance(
            self._clock(),
            observation=None,
            requested_qpos=None,
            command_qpos=self.backend.get_target_joint_pos(),
        )
