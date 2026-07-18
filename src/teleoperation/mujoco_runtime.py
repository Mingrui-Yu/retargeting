"""Shared composition of retargeting frames and MuJoCo command periods."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from retargeting.core.types import HandObservation
from teleoperation.backends.base import RobotBackend
from teleoperation.output import QposCommandLimiter
from teleoperation.session import TeleoperationSession


AlignmentInitializer = Callable[[TeleoperationSession, Any, np.ndarray], bool]


@dataclass(frozen=True)
class MujocoStepResult:
    """State and diagnostics produced by one MuJoCo simulation frame."""

    observation: HandObservation | None
    requested_qpos: np.ndarray | None
    command_qpos: np.ndarray
    actual_qpos: np.ndarray
    diagnostics: dict[str, float]


class MujocoTeleoperationRuntime:
    """Retarget input frames and execute fixed-period MuJoCo commands."""

    def __init__(
        self,
        session: TeleoperationSession,
        backend: RobotBackend,
        command_limiter: QposCommandLimiter,
        realtime: bool = True,
        startup_move_frames: int = 0,
        post_command_step: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a retargeting-to-MuJoCo runtime shared by live and offline runners.

        Args:
            session: Input-independent retargeting session.
            backend: Robot backend advanced once for every execution command period.
            command_limiter: Actuator-range and startup interpolation policy.
            realtime: Whether to wait for the remainder of each wall-clock period.
            startup_move_frames: Valid retargeted frames that synchronously move to each target.
            post_command_step: Optional observer called after every backend command period.
            clock: Monotonic clock dependency used for timing and tests.
            sleep: Wall-clock sleep dependency used for real-time pacing.

        Returns:
            None.
        """
        self.session = session
        self.backend = backend
        self.command_limiter = command_limiter
        self.realtime = bool(realtime)
        if isinstance(startup_move_frames, bool) or not isinstance(startup_move_frames, int):
            raise ValueError("startup_move_frames must be a non-negative integer.")
        if startup_move_frames < 0:
            raise ValueError("startup_move_frames must be a non-negative integer.")
        self.startup_move_frames = startup_move_frames
        self._post_command_step = post_command_step
        self._clock = clock
        self._sleep = sleep
        self.frame_count = 0
        self.command_step_count = 0
        self.retarget_frame_count = 0
        expected_period = 1.0 / self.command_limiter.command_hz
        if not np.isclose(self.backend.control_period, expected_period, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Backend and command limiter periods differ: "
                f"backend={self.backend.control_period}, limiter={expected_period}."
            )

    @property
    def control_period(self) -> float:
        """Return the fixed simulated duration advanced for each command.

        Args:
            None.

        Returns:
            Command period in seconds.
        """
        return self.backend.control_period

    def set_post_command_step(self, callback: Callable[[], None] | None) -> None:
        """Set an optional observer called after each completed command period.

        Args:
            callback: Zero-argument observer, or None to disable observation.

        Returns:
            None.
        """
        self._post_command_step = callback

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset simulation and all temporal command state to one joint configuration.

        Args:
            qpos: Optional backend reset configuration; defaults to its configured initial qpos.

        Returns:
            None.
        """
        self.backend.reset(qpos)
        reset_qpos = self.backend.get_target_joint_pos()
        self.session.reset(reset_qpos)
        self.command_limiter.reset(reset_qpos)
        self.frame_count = 0
        self.command_step_count = 0
        self.retarget_frame_count = 0

    def _run_command_period(
        self,
        started_at: float,
        command_qpos: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], float, float]:
        """Apply one target, advance physics, notify observers, and pace it.

        Args:
            started_at: Wall-clock timestamp captured before this command period's work.
            command_qpos: Target applied during this command period.

        Returns:
            Applied target, final backend diagnostics, compute time, and overrun.
        """
        applied_qpos = self.backend.ctrl_joint_pos(command_qpos)
        self.backend.step()
        if self._post_command_step is not None:
            self._post_command_step()
        compute_elapsed = self._clock() - started_at
        overrun = max(0.0, compute_elapsed - self.control_period)
        sleep_duration = max(0.0, self.control_period - compute_elapsed) if self.realtime else 0.0
        if sleep_duration > 0:
            self._sleep(sleep_duration)
        self.command_step_count += 1
        backend_diagnostics = getattr(self.backend, "get_diagnostics", lambda: {})()
        diagnostics = {str(key): float(value) for key, value in backend_diagnostics.items()}
        return np.asarray(applied_qpos, dtype=float).copy(), diagnostics, compute_elapsed, overrun

    def _advance(
        self,
        started_at: float,
        observation: HandObservation | None,
        requested_qpos: np.ndarray | None,
        command_qpos: np.ndarray,
        diagnostics: dict[str, float] | None = None,
        startup_move_active: bool = False,
    ) -> MujocoStepResult:
        """Apply one or more commands for one source frame and build its result.

        Args:
            started_at: Wall-clock timestamp captured before frame processing.
            observation: Current canonical observation, or None for a hold frame.
            requested_qpos: Raw retargeting result, or None for a hold frame.
            command_qpos: Non-empty array of execution commands, one row per period.
            diagnostics: Retargeting diagnostics collected before simulation.
            startup_move_active: Whether the commands belong to startup interpolation.

        Returns:
            State and merged diagnostics for the completed source frame.
        """
        commands = np.asarray(command_qpos, dtype=float)
        if commands.ndim != 2 or commands.shape[0] == 0:
            raise ValueError("command_qpos must be a non-empty two-dimensional waypoint array.")
        total_compute_elapsed = 0.0
        total_overrun = 0.0
        backend_diagnostics: dict[str, float] = {}
        applied_qpos = commands[-1].copy()
        for command_index, command in enumerate(commands):
            command_started_at = started_at if command_index == 0 else self._clock()
            applied_qpos, backend_diagnostics, compute_elapsed, overrun = self._run_command_period(
                command_started_at,
                command,
            )
            total_compute_elapsed += compute_elapsed
            total_overrun += overrun
        wall_elapsed = self._clock() - started_at
        merged_diagnostics = dict(diagnostics or {})
        merged_diagnostics.update(backend_diagnostics)
        merged_diagnostics.update(
            {
                "runtime_compute_time": float(total_compute_elapsed),
                "runtime_wall_time": float(wall_elapsed),
                "runtime_overrun": float(total_overrun),
                "command_periods_advanced": float(len(commands)),
                "startup_move_active": float(startup_move_active),
            }
        )
        self.command_limiter.reset(applied_qpos)
        self.frame_count += 1
        return MujocoStepResult(
            observation=observation,
            requested_qpos=None if requested_qpos is None else requested_qpos.copy(),
            command_qpos=np.asarray(applied_qpos, dtype=float).copy(),
            actual_qpos=self.backend.get_joint_pos(),
            diagnostics=merged_diagnostics,
        )

    def move_to_joint_pos(
        self,
        requested_qpos: np.ndarray,
        *,
        started_at: float | None = None,
        observation: HandObservation | None = None,
        diagnostics: dict[str, float] | None = None,
    ) -> MujocoStepResult:
        """Synchronously interpolate actuator targets to one requested position.

        Args:
            requested_qpos: Desired final target before actuator-range handling.
            started_at: Optional frame start time that includes upstream processing.
            observation: Optional canonical observation associated with the target.
            diagnostics: Optional retargeting diagnostics for the source frame.

        Returns:
            Result after every planned command period has completed.
        """
        frame_started_at = self._clock() if started_at is None else started_at
        waypoints = self.command_limiter.plan_move(
            self.backend.get_target_joint_pos(),
            requested_qpos,
        )
        return self._advance(
            frame_started_at,
            observation,
            requested_qpos,
            waypoints,
            diagnostics,
            startup_move_active=True,
        )

    def _execute_requested_qpos(
        self,
        started_at: float,
        observation: HandObservation,
        requested_qpos: np.ndarray,
        diagnostics: dict[str, float],
    ) -> MujocoStepResult:
        """Select startup interpolation or direct execution for one valid frame.

        Args:
            started_at: Wall-clock timestamp captured before frame processing.
            observation: Current canonical observation.
            requested_qpos: Retargeted joint-position request.
            diagnostics: Retargeting diagnostics for the source frame.

        Returns:
            Completed result for the valid retargeted frame.
        """
        if self.retarget_frame_count < self.startup_move_frames:
            result = self.move_to_joint_pos(
                requested_qpos,
                started_at=started_at,
                observation=observation,
                diagnostics=diagnostics,
            )
        else:
            target = self.command_limiter.apply_range(requested_qpos)
            result = self._advance(
                started_at,
                observation,
                requested_qpos,
                target[None, :],
                diagnostics,
            )
        self.retarget_frame_count += 1
        return result

    def step_observation(self, observation: HandObservation) -> MujocoStepResult:
        """Retarget and execute one canonical observation immediately.

        Args:
            observation: Current human-hand observation.

        Returns:
            Completed simulation frame.
        """
        started_at = self._clock()
        requested_qpos, diagnostics = self.session.retarget_observation(observation)
        return self._execute_requested_qpos(started_at, observation, requested_qpos, diagnostics)

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
                command_qpos=self.backend.get_target_joint_pos()[None, :],
            )
        requested_qpos, diagnostics = self.session.retarget_observation(observation)
        return self._execute_requested_qpos(started_at, observation, requested_qpos, diagnostics)

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
            command_qpos=self.backend.get_target_joint_pos()[None, :],
        )


class AlignedMujocoTeleoperationDriver:
    """Align raw input before forwarding each frame to a MuJoCo runtime."""

    def __init__(
        self,
        runtime: MujocoTeleoperationRuntime,
        alignment_initializer: AlignmentInitializer,
    ) -> None:
        """Create a shared aligned-input driver.

        Args:
            runtime: Immediate retarget-command-step runtime.
            alignment_initializer: Input-specific session alignment operation.

        Returns:
            None.
        """
        self.runtime = runtime
        self.alignment_initializer = alignment_initializer
        self.alignment_initialized = False

    def reset(self) -> None:
        """Reset runtime execution and require alignment from a new source frame.

        Args:
            None.

        Returns:
            None.
        """
        self.alignment_initialized = False
        self.runtime.reset()

    def step(self, sensor_data: Any) -> MujocoStepResult:
        """Align, execute, or hold exactly one input frame.

        Args:
            sensor_data: Raw frame accepted by the configured input adapter.

        Returns:
            Runtime result after one complete source-frame operation.
        """
        if not self.alignment_initialized:
            self.alignment_initialized = self.alignment_initializer(
                self.runtime.session,
                sensor_data,
                self.runtime.backend.get_joint_pos(),
            )
            if not self.alignment_initialized:
                return self.runtime.step_hold()
        return self.runtime.step_sensor_data(sensor_data)
