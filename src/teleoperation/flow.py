"""Backend-neutral batch and execution flows for retargeting."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from mr_utils.utils_calc import transformPositions
from retargeting.core import Retargeter
from retargeting.core.types import RetargetingHandObservation
from retargeting.evaluation.robot_metrics import RobotBenchmark
from teleoperation.backends.base import BackendStepResult, RobotBackend
from teleoperation.inputs.base import HandInput
from teleoperation.observation_mapping import HandObservationMapper
from teleoperation.output import QposCommandLimiter, QposOutputFilter
from teleoperation.types import (
    ExecutionStatus,
    ExecutionStepResult,
    FlowSummary,
    RetargetedFrameResult,
    SensorHandSample,
)


def evaluate_raw_result(
    evaluator: RobotBenchmark | None,
    qpos: np.ndarray,
    observation: RetargetingHandObservation,
) -> dict[str, Any]:
    """Compute optional robot metrics from the unfiltered solver result.

    Args:
        evaluator: Optional pure robot benchmark implementation.
        qpos: Raw optimized robot configuration.
        observation: Canonical observation supplied to the solver.

    Returns:
        Existing benchmark diagnostic keys, or an empty mapping when disabled.
    """
    if evaluator is None:
        return {}
    keypoints_world = transformPositions(
        observation.keypoints_wrist,
        target_frame_pose_inv=observation.wrist_pose_world,
    )
    return {
        "position_err": evaluator.position_error(qpos, keypoints_world, 1),
        "orientation_err": evaluator.orientation_error(qpos, keypoints_world, 1),
        "relative_position_err": evaluator.relative_position_error(qpos, keypoints_world, 1),
        "relative_position_to_wrist_err": evaluator.relative_position_to_wrist_error(qpos, keypoints_world, 1),
    }


def retarget_frame(
    observation: RetargetingHandObservation,
    sample: SensorHandSample,
    retargeter: Retargeter,
    output_filter: QposOutputFilter,
    evaluator: RobotBenchmark | None,
) -> RetargetedFrameResult:
    """Run the shared solve, raw evaluation, filtering, and temporal-state path.

    Args:
        observation: Canonical robot-world hand observation.
        sample: Sensor sample carrying source metadata.
        retargeter: Pure retargeting solver with temporal objective state.
        output_filter: Execution-layer qpos output filter.
        evaluator: Optional pure robot benchmark evaluator.

    Returns:
        Backend-neutral filtered retargeting frame result.
    """
    solved = retargeter.solve(observation, previous_qpos=output_filter.previous_qpos)
    diagnostics = {**evaluate_raw_result(evaluator, solved.qpos, observation), **solved.diagnostics}
    filtered_qpos = output_filter.apply(solved.qpos)
    # The temporal objective intentionally follows the filtered output that is
    # visible to downstream execution and artifact consumers.
    retargeter.previous_qpos = np.asarray(filtered_qpos, dtype=float).copy()
    return RetargetedFrameResult(
        observation=observation,
        retargeted_qpos=filtered_qpos,
        diagnostics=diagnostics,
        source_index=sample.source_index,
        timestamp=sample.timestamp,
    )


class BatchRetargetFlow:
    """Retarget valid finite input frames without a robot backend or timing."""

    def __init__(
        self,
        *,
        input: HandInput,
        observation_mapper: HandObservationMapper,
        retargeter: Retargeter,
        output_filter: QposOutputFilter,
        initial_robot_qpos: np.ndarray,
        evaluator: RobotBenchmark | None = None,
        observers: Iterable[Callable[[RetargetedFrameResult], None]] = (),
    ) -> None:
        """Create a finite computation-only retargeting component graph.

        Args:
            input: Finite sensor-first hand input.
            observation_mapper: Sensor-to-core calibration and alignment strategy.
            retargeter: Pure canonical-observation solver.
            output_filter: Qpos output filter shared with execution semantics.
            initial_robot_qpos: Robot state used for alignment and temporal reset.
            evaluator: Optional raw-result robot evaluator.
            observers: Passive observers notified for valid retargeted frames.

        Returns:
            None.
        """
        self.input = input
        self.observation_mapper = observation_mapper
        self.retargeter = retargeter
        self.output_filter = output_filter
        self.initial_robot_qpos = np.asarray(initial_robot_qpos, dtype=float).copy()
        self.evaluator = evaluator
        self._observers = list(observers)
        self.mapping_initialized = False

    def reset(self) -> None:
        """Reset input, mapping, solver, and filtering state for a new batch.

        Args:
            None.

        Returns:
            None.
        """
        self.retargeter.reset(self.initial_robot_qpos)
        self.output_filter.reset(self.initial_robot_qpos)
        self.input.reset()
        self.observation_mapper.reset()
        self.mapping_initialized = False

    def run(self) -> list[RetargetedFrameResult]:
        """Retarget every valid selected frame and skip later missing detections.

        Args:
            None.

        Returns:
            Backend-neutral results for valid source frames only.
        """
        records: list[RetargetedFrameResult] = []
        self.input.open()
        try:
            while True:
                sample = self.input.read()
                if sample is None:
                    break
                if not self.mapping_initialized:
                    self.mapping_initialized = self.observation_mapper.initialize(sample, self.initial_robot_qpos)
                    if not self.mapping_initialized:
                        source = "unknown" if sample.source_index is None else str(sample.source_index)
                        raise ValueError(f"Unable to initialize hand observation mapping from source frame {source}.")
                observation = self.observation_mapper.map(sample)
                if observation is None:
                    continue
                record = retarget_frame(
                    observation,
                    sample,
                    self.retargeter,
                    self.output_filter,
                    self.evaluator,
                )
                records.append(record)
                for observer in self._observers:
                    observer(record)
        finally:
            self.input.close()
        return records


class ExecutionFlow:
    """Coordinate input, mapping, retargeting, commands, timing, and reset."""

    def __init__(
        self,
        *,
        input: HandInput | None,
        observation_mapper: HandObservationMapper,
        retargeter: Retargeter,
        output_filter: QposOutputFilter,
        command_policy: QposCommandLimiter,
        backend: RobotBackend,
        evaluator: RobotBenchmark | None = None,
        realtime: bool = True,
        startup_move_frames: int = 0,
        loop: bool = False,
        max_frames: int | None = None,
        command_observers: Iterable[Callable[[BackendStepResult], None]] = (),
        step_observers: Iterable[Callable[[ExecutionStepResult], None]] = (),
        reset_observers: Iterable[Callable[[np.ndarray], None]] = (),
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create one complete execution component graph.

        Args:
            input: Pull-based sensor input, or None for callback-only ``step`` use.
            observation_mapper: Sensor-to-core calibration and alignment strategy.
            retargeter: Pure canonical-observation solver.
            output_filter: Qpos filtering outside the retargeting core.
            command_policy: Actuator range and startup waypoint policy.
            backend: Atomic robot command-period backend.
            evaluator: Optional raw-result robot evaluator.
            realtime: Whether each command period waits for wall-clock pacing.
            startup_move_frames: Valid source frames using synchronous interpolation.
            loop: Whether finite input restarts with a complete flow reset.
            max_frames: Optional total source-frame limit for live or finite runs.
            command_observers: Passive observers notified after each backend period.
            step_observers: Passive observers notified after each immutable source result.
            reset_observers: Passive observers notified with reset backend target state.
            clock: Injected monotonic clock.
            sleep: Injected wall-clock sleep operation.

        Returns:
            None.
        """
        if isinstance(startup_move_frames, bool) or not isinstance(startup_move_frames, int):
            raise ValueError("startup_move_frames must be a non-negative integer.")
        if startup_move_frames < 0:
            raise ValueError("startup_move_frames must be a non-negative integer.")
        if max_frames is not None and (isinstance(max_frames, bool) or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer when provided.")
        expected_period = 1.0 / command_policy.command_hz
        if not np.isclose(backend.control_period, expected_period, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Backend and command policy periods differ: "
                f"backend={backend.control_period}, policy={expected_period}."
            )
        self.input = input
        self.observation_mapper = observation_mapper
        self.retargeter = retargeter
        self.output_filter = output_filter
        self.command_policy = command_policy
        self.backend = backend
        self.evaluator = evaluator
        self.realtime = bool(realtime)
        self.startup_move_frames = startup_move_frames
        self.loop = bool(loop)
        self.max_frames = max_frames
        self._command_observers = list(command_observers)
        self._step_observers = list(step_observers)
        self._reset_observers = list(reset_observers)
        self._clock = clock
        self._sleep = sleep
        self.mapping_initialized = False
        self.source_frame_count = 0
        self.retarget_frame_count = 0
        self.command_period_count = 0

    @property
    def control_period(self) -> float:
        """Return the backend command period in seconds.

        Args:
            None.

        Returns:
            Fixed command period in seconds.
        """
        return self.backend.control_period

    def add_command_observer(self, observer: Callable[[BackendStepResult], None]) -> None:
        """Register a passive observer for completed command periods.

        Args:
            observer: Callback receiving immutable backend state.

        Returns:
            None.
        """
        self._command_observers.append(observer)

    def add_step_observer(self, observer: Callable[[ExecutionStepResult], None]) -> None:
        """Register a passive observer for completed source-frame results.

        Args:
            observer: Callback receiving an immutable execution result.

        Returns:
            None.
        """
        self._step_observers.append(observer)

    def add_reset_observer(self, observer: Callable[[np.ndarray], None]) -> None:
        """Register a passive observer for synchronized reset state.

        Args:
            observer: Callback receiving the backend reset target.

        Returns:
            None.
        """
        self._reset_observers.append(observer)

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Synchronize every stateful component for a fresh source cycle.

        Args:
            qpos: Optional backend reset configuration.

        Returns:
            None.
        """
        self.backend.reset(qpos)
        reset_qpos = self.backend.get_target_joint_pos()
        self.retargeter.reset(reset_qpos)
        self.output_filter.reset(reset_qpos)
        self.command_policy.reset(reset_qpos)
        if self.input is not None:
            self.input.reset()
        self.observation_mapper.reset()
        self.mapping_initialized = False
        self.source_frame_count = 0
        self.retarget_frame_count = 0
        self.command_period_count = 0
        for observer in self._reset_observers:
            observer(np.asarray(reset_qpos, dtype=float).copy())

    def _execute_command_period(
        self,
        started_at: float,
        command_qpos: np.ndarray,
    ) -> tuple[BackendStepResult, float, float]:
        """Execute, observe, and pace one atomic backend command period.

        Args:
            started_at: Wall-clock time including upstream work for this period.
            command_qpos: Planned target for the current command period.

        Returns:
            Backend state, compute time, and timing overrun in seconds.
        """
        backend_result = self.backend.execute(command_qpos)
        for observer in self._command_observers:
            observer(backend_result)
        compute_elapsed = self._clock() - started_at
        overrun = max(0.0, compute_elapsed - self.control_period)
        sleep_duration = max(0.0, self.control_period - compute_elapsed) if self.realtime else 0.0
        if sleep_duration > 0.0:
            self._sleep(sleep_duration)
        self.command_period_count += 1
        return backend_result, compute_elapsed, overrun

    def _execute_waypoints(
        self,
        *,
        frame_started_at: float,
        waypoints: np.ndarray,
        status: ExecutionStatus,
        retargeted_frame: RetargetedFrameResult | None,
        startup_move_active: bool,
        sample: SensorHandSample,
    ) -> ExecutionStepResult:
        """Execute every waypoint for one source frame and merge diagnostics.

        Args:
            frame_started_at: Time captured before mapping and retargeting.
            waypoints: Non-empty command array with one row per backend period.
            status: Source-frame execution state.
            retargeted_frame: Successful retargeting result, or None for hold/wait.
            startup_move_active: Whether startup interpolation selected the waypoints.
            sample: Source sample carrying index and timestamp metadata.

        Returns:
            Immutable execution result after all command periods finish.
        """
        commands = np.asarray(waypoints, dtype=float)
        if commands.ndim != 2 or commands.shape[0] == 0:
            raise ValueError("waypoints must be a non-empty two-dimensional array.")
        total_compute = 0.0
        total_overrun = 0.0
        backend_result: BackendStepResult | None = None
        for command_index, command in enumerate(commands):
            command_started_at = frame_started_at if command_index == 0 else self._clock()
            backend_result, compute_elapsed, overrun = self._execute_command_period(command_started_at, command)
            total_compute += compute_elapsed
            total_overrun += overrun
        if backend_result is None:
            raise RuntimeError("Execution produced no backend command period.")
        self.command_policy.reset(backend_result.command_qpos)
        diagnostics = {} if retargeted_frame is None else dict(retargeted_frame.diagnostics)
        diagnostics.update(backend_result.diagnostics)
        diagnostics.update(
            {
                "runtime_compute_time": float(total_compute),
                "runtime_wall_time": float(self._clock() - frame_started_at),
                "runtime_overrun": float(total_overrun),
                "command_periods_advanced": float(len(commands)),
                "startup_move_active": float(startup_move_active),
            }
        )
        self.source_frame_count += 1
        result = ExecutionStepResult(
            status=status,
            retargeted_frame=retargeted_frame,
            command_qpos=backend_result.command_qpos,
            actual_qpos=backend_result.actual_qpos,
            diagnostics=diagnostics,
            source_index=sample.source_index,
            timestamp=sample.timestamp,
        )
        for observer in self._step_observers:
            observer(result)
        return result

    def _hold(
        self,
        sample: SensorHandSample,
        status: ExecutionStatus,
        started_at: float,
    ) -> ExecutionStepResult:
        """Advance one command period while retaining the previous target.

        Args:
            sample: Source sample that could not produce a mapped observation.
            status: Waiting or held status explaining the missing command.
            started_at: Time captured before mapping was attempted.

        Returns:
            Completed hold result.
        """
        return self._execute_waypoints(
            frame_started_at=started_at,
            waypoints=self.backend.get_target_joint_pos()[None, :],
            status=status,
            retargeted_frame=None,
            startup_move_active=False,
            sample=sample,
        )

    def step(self, sample: SensorHandSample) -> ExecutionStepResult:
        """Map, retarget, plan, and execute exactly one sensor sample.

        Args:
            sample: Sensor-normalized hand sample from pull or callback acquisition.

        Returns:
            Completed execution result, including explicit hold or mapping state.
        """
        if not isinstance(sample, SensorHandSample):
            raise TypeError("ExecutionFlow.step() requires a SensorHandSample.")
        started_at = self._clock()
        if not self.mapping_initialized:
            self.mapping_initialized = self.observation_mapper.initialize(sample, self.backend.get_joint_pos())
            if not self.mapping_initialized:
                return self._hold(sample, ExecutionStatus.WAITING_FOR_MAPPING, started_at)
        observation = self.observation_mapper.map(sample)
        if observation is None:
            return self._hold(sample, ExecutionStatus.HELD, started_at)
        frame = retarget_frame(observation, sample, self.retargeter, self.output_filter, self.evaluator)
        startup_move_active = self.retarget_frame_count < self.startup_move_frames
        if startup_move_active:
            waypoints = self.command_policy.plan_move(
                self.backend.get_target_joint_pos(),
                frame.retargeted_qpos,
            )
        else:
            waypoints = self.command_policy.apply_range(frame.retargeted_qpos)[None, :]
        result = self._execute_waypoints(
            frame_started_at=started_at,
            waypoints=waypoints,
            status=ExecutionStatus.EXECUTED,
            retargeted_frame=frame,
            startup_move_active=startup_move_active,
            sample=sample,
        )
        self.retarget_frame_count += 1
        return result

    def run(self) -> FlowSummary:
        """Own a pull-input lifecycle until end, limit, or user interruption.

        Args:
            None.

        Returns:
            Aggregate counts and the last completed execution result.
        """
        if self.input is None:
            raise RuntimeError("ExecutionFlow.run() requires a pull-based input.")
        total_source_frames = 0
        total_retarget_frames = 0
        total_command_periods = 0
        cycles_completed = 0
        last_result: ExecutionStepResult | None = None
        self.input.open()
        try:
            while True:
                sample = self.input.read()
                if sample is None:
                    cycles_completed += 1
                    if not self.loop:
                        break
                    total_retarget_frames += self.retarget_frame_count
                    total_command_periods += self.command_period_count
                    self.reset()
                    continue
                last_result = self.step(sample)
                total_source_frames += 1
                if self.max_frames is not None and total_source_frames >= self.max_frames:
                    cycles_completed += 1
                    break
        except KeyboardInterrupt:
            cycles_completed += 1
        finally:
            self.input.close()
        total_retarget_frames += self.retarget_frame_count
        total_command_periods += self.command_period_count
        return FlowSummary(
            source_frames_processed=total_source_frames,
            retarget_frames_processed=total_retarget_frames,
            command_periods_advanced=total_command_periods,
            cycles_completed=max(1, cycles_completed),
            last_result=last_result,
        )
