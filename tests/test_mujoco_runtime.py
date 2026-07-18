from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeting.core.types import RetargetingHandObservation, RetargetingResult
from teleoperation.backends.base import BackendStepResult
from teleoperation.types import ExecutionStatus, SensorHandSample


class _FakeClock:
    """Deterministic wall clock used for pacing and overrun tests."""

    def __init__(self) -> None:
        """Initialize fake wall time at zero.

        Args:
            None.

        Returns:
            None.
        """
        self.now = 0.0

    def __call__(self) -> float:
        """Return current fake wall time.

        Args:
            None.

        Returns:
            Current fake seconds.
        """
        return self.now

    def sleep(self, duration: float) -> None:
        """Advance fake time by a requested sleep duration.

        Args:
            duration: Non-negative duration in seconds.

        Returns:
            None.
        """
        self.now += duration


class _FakeBackend:
    """Two-joint backend with atomic 20 Hz command periods."""

    control_period = 0.05

    def __init__(self, clock: _FakeClock | None = None, compute_time: float = 0.0) -> None:
        """Initialize target, measured state, and command history.

        Args:
            clock: Optional fake clock advanced by backend compute.
            compute_time: Seconds consumed by each atomic execute call.

        Returns:
            None.
        """
        self.target = np.zeros(2)
        self.actual = np.zeros(2)
        self.simulation_time = 0.0
        self.command_history: list[np.ndarray] = []
        self.reset_history: list[np.ndarray] = []
        self.clock = clock
        self.compute_time = compute_time

    def reset(self, qpos=None) -> None:
        """Reset target, measured state, and simulated time.

        Args:
            qpos: Optional two-joint reset position.

        Returns:
            None.
        """
        self.target = np.zeros(2) if qpos is None else np.asarray(qpos, dtype=float).copy()
        self.actual = self.target.copy()
        self.simulation_time = 0.0
        self.reset_history.append(self.target.copy())

    def get_joint_pos(self) -> np.ndarray:
        """Return current measured positions.

        Args:
            None.

        Returns:
            Two-joint measured state.
        """
        return self.actual.copy()

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the last accepted target.

        Args:
            None.

        Returns:
            Two-joint target state.
        """
        return self.target.copy()

    def execute(self, qpos: np.ndarray) -> BackendStepResult:
        """Execute one atomic command period and realize the target exactly.

        Args:
            qpos: Two-joint target command.

        Returns:
            Immutable post-period backend state.
        """
        self.target = np.asarray(qpos, dtype=float).copy()
        self.command_history.append(self.target.copy())
        if self.clock is not None:
            self.clock.now += self.compute_time
        self.actual = self.target.copy()
        self.simulation_time += self.control_period
        return BackendStepResult(
            command_qpos=self.target,
            actual_qpos=self.actual,
            diagnostics={"simulation_time": self.simulation_time},
        )


class _FakeRetargeter:
    """Return qpos encoded in canonical observation keypoints."""

    def __init__(self) -> None:
        """Initialize temporal state and call records.

        Args:
            None.

        Returns:
            None.
        """
        self.qpos_init = np.zeros(2)
        self.previous_qpos = self.qpos_init.copy()
        self.previous_references: list[np.ndarray] = []
        self.reset_history: list[np.ndarray] = []

    def solve(self, observation, previous_qpos=None) -> RetargetingResult:
        """Return the first two values encoded by the observation.

        Args:
            observation: Canonical observation carrying a fake target.
            previous_qpos: Temporal reference supplied by the flow.

        Returns:
            Raw qpos and deterministic solver diagnostics.
        """
        self.previous_references.append(np.asarray(previous_qpos, dtype=float).copy())
        return RetargetingResult(
            qpos=np.asarray(observation.keypoints_wrist[0, :2], dtype=float).copy(),
            diagnostics={"optimization_time": 0.001},
        )

    def reset(self, previous_qpos=None) -> None:
        """Reset the fake temporal reference.

        Args:
            previous_qpos: Reset reference supplied by the flow.

        Returns:
            None.
        """
        self.previous_qpos = np.asarray(previous_qpos, dtype=float).copy()
        self.reset_history.append(self.previous_qpos.copy())


class _FakeMapper:
    """Identity-like mapper with configurable initialization outcomes."""

    def __init__(self, initialization_results: tuple[bool, ...] = ()) -> None:
        """Initialize mapping and reset call records.

        Args:
            initialization_results: Optional deterministic outcomes before normal validity.

        Returns:
            None.
        """
        self._initialization_results = iter(initialization_results)
        self.initialize_qpos: list[np.ndarray] = []
        self.reset_count = 0

    def initialize(self, sample: SensorHandSample, robot_qpos: np.ndarray) -> bool:
        """Record measured qpos and return a configured mapping outcome.

        Args:
            sample: Current sensor sample.
            robot_qpos: Current measured backend state.

        Returns:
            Configured result, or sample validity after outcomes are exhausted.
        """
        self.initialize_qpos.append(np.asarray(robot_qpos, dtype=float).copy())
        try:
            return next(self._initialization_results)
        except StopIteration:
            return sample.has_hand

    def map(self, sample: SensorHandSample) -> RetargetingHandObservation | None:
        """Map complete samples directly into canonical observations.

        Args:
            sample: Current sensor-normalized sample.

        Returns:
            Canonical observation, or None for missing hand data.
        """
        if not sample.has_hand:
            return None
        return RetargetingHandObservation(
            keypoints_wrist=sample.keypoints_wrist,
            wrist_pose_world=sample.wrist_pose_sensor,
            raw=sample.raw,
        )

    def reset(self) -> None:
        """Record one mapper reset.

        Args:
            None.

        Returns:
            None.
        """
        self.reset_count += 1


class _FakeEvaluator:
    """Record the raw qpos supplied before output filtering."""

    def __init__(self) -> None:
        """Initialize an empty raw-qpos history.

        Args:
            None.

        Returns:
            None.
        """
        self.raw_qpos: list[np.ndarray] = []

    def _record(self, qpos: np.ndarray) -> float:
        """Record one raw result and return its first value.

        Args:
            qpos: Raw optimized qpos.

        Returns:
            First qpos value as a deterministic metric.
        """
        self.raw_qpos.append(np.asarray(qpos, dtype=float).copy())
        return float(qpos[0])

    def position_error(self, qpos, keypoints, scale):
        """Record raw qpos for the first benchmark metric.

        Args:
            qpos: Raw optimized qpos.
            keypoints: Unused canonical keypoints.
            scale: Unused metric scale.

        Returns:
            Deterministic scalar metric.
        """
        del keypoints, scale
        return self._record(qpos)

    def orientation_error(self, qpos, keypoints, scale):
        """Return the deterministic orientation metric.

        Args:
            qpos: Raw optimized qpos.
            keypoints: Unused canonical keypoints.
            scale: Unused metric scale.

        Returns:
            Deterministic scalar metric.
        """
        del keypoints, scale
        return float(qpos[0])

    def relative_position_error(self, qpos, keypoints, scale):
        """Return the deterministic relative-position metric.

        Args:
            qpos: Raw optimized qpos.
            keypoints: Unused canonical keypoints.
            scale: Unused metric scale.

        Returns:
            Deterministic scalar metric.
        """
        del keypoints, scale
        return float(qpos[0])

    def relative_position_to_wrist_error(self, qpos, keypoints, scale):
        """Return the deterministic wrist-relative metric.

        Args:
            qpos: Raw optimized qpos.
            keypoints: Unused canonical keypoints.
            scale: Unused metric scale.

        Returns:
            Deterministic scalar metric.
        """
        del keypoints, scale
        return float(qpos[0])


class _FiniteInput:
    """Two-frame finite input that records lifecycle operations."""

    def __init__(self) -> None:
        """Initialize lifecycle counters and cursor.

        Args:
            None.

        Returns:
            None.
        """
        self.samples = [_sample((0.1, -0.1), 0), _sample((0.2, -0.2), 1)]
        self.cursor = 0
        self.open_count = 0
        self.reset_count = 0
        self.close_count = 0

    def open(self) -> None:
        """Open the source at its first frame.

        Args:
            None.

        Returns:
            None.
        """
        self.cursor = 0
        self.open_count += 1

    def read(self) -> SensorHandSample | None:
        """Return the next sample or finite end-of-stream.

        Args:
            None.

        Returns:
            Next sample, or None after two frames.
        """
        if self.cursor >= len(self.samples):
            return None
        sample = self.samples[self.cursor]
        self.cursor += 1
        return sample

    def reset(self) -> None:
        """Rewind the source for a new cycle.

        Args:
            None.

        Returns:
            None.
        """
        self.cursor = 0
        self.reset_count += 1

    def close(self) -> None:
        """Record source closure.

        Args:
            None.

        Returns:
            None.
        """
        self.close_count += 1


def _sample(qpos: tuple[float, float] | None, source_index: int = 0) -> SensorHandSample:
    """Build a complete or missing sensor sample.

    Args:
        qpos: Target encoded in the first keypoint, or None for missing hand data.
        source_index: Source metadata attached to the sample.

    Returns:
        Sensor-normalized sample for flow tests.
    """
    if qpos is None:
        return SensorHandSample(None, None, raw=None, source_index=source_index)
    keypoints = np.zeros((21, 3))
    keypoints[0, :2] = qpos
    return SensorHandSample(keypoints, np.eye(4), raw={"qpos": qpos}, source_index=source_index)


def _build_flow(
    *,
    backend: _FakeBackend | None = None,
    mapper: _FakeMapper | None = None,
    clock: _FakeClock | None = None,
    startup_move_frames: int = 0,
    smooth: bool = False,
    evaluator=None,
    realtime: bool = False,
):
    """Build a complete fake execution flow and its stateful components.

    Args:
        backend: Optional fake backend.
        mapper: Optional fake mapper.
        clock: Optional fake wall clock.
        startup_move_frames: Valid frames using waypoint interpolation.
        smooth: Whether output filtering uses alpha 0.3.
        evaluator: Optional fake evaluator.
        realtime: Whether the flow sleeps to pace command periods.

    Returns:
        Flow, backend, mapper, retargeter, output filter, and command policy.
    """
    from teleoperation.config import load_teleoperation_mode_config
    from teleoperation.flow import ExecutionFlow
    from teleoperation.output import QposCommandLimiter, QposOutputFilter

    clock = _FakeClock() if clock is None else clock
    backend = _FakeBackend() if backend is None else backend
    mapper = _FakeMapper() if mapper is None else mapper
    retargeter = _FakeRetargeter()
    mode = load_teleoperation_mode_config(
        {
            "name": "test",
            "output": {"smooth_output_qpos": smooth, "smoothing_alpha": 0.3},
        }
    )
    output_filter = QposOutputFilter(np.zeros(2), mode)
    command_policy = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.array([1.0, 2.0]),
        command_hz=20.0,
        lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
    )
    flow = ExecutionFlow(
        input=None,
        observation_mapper=mapper,
        retargeter=retargeter,
        output_filter=output_filter,
        evaluator=evaluator,
        command_policy=command_policy,
        backend=backend,
        realtime=realtime,
        startup_move_frames=startup_move_frames,
        clock=clock,
        sleep=clock.sleep,
    )
    return flow, backend, mapper, retargeter, output_filter, command_policy


def test_execution_flow_direct_commands_have_no_explicit_speed_limit():
    """Direct execution sends each requested target in one command period."""
    clock = _FakeClock()
    backend = _FakeBackend()
    flow, _, _, _, _, _ = _build_flow(backend=backend, clock=clock, realtime=True)

    first = flow.step(_sample((1.0, -1.0)))
    for index in range(1, 20):
        last = flow.step(_sample((1.0, -1.0), source_index=index))

    assert first.status is ExecutionStatus.EXECUTED
    np.testing.assert_allclose(first.command_qpos, [1.0, -1.0])
    np.testing.assert_allclose(last.command_qpos, [1.0, -1.0])
    assert flow.source_frame_count == 20
    assert flow.retarget_frame_count == 20
    assert flow.command_period_count == 20
    assert backend.simulation_time == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)
    assert last.diagnostics["runtime_overrun"] == 0.0


def test_execution_flow_blocks_on_startup_waypoints_then_switches_to_direct():
    """Startup frames complete synchronized waypoints before direct execution."""
    clock = _FakeClock()
    backend = _FakeBackend()
    observed_commands: list[np.ndarray] = []
    flow, _, _, _, _, _ = _build_flow(
        backend=backend,
        clock=clock,
        startup_move_frames=1,
        realtime=True,
    )
    flow.add_command_observer(lambda result: observed_commands.append(result.actual_qpos.copy()))

    startup = flow.step(_sample((0.11, -0.21)))
    direct = flow.step(_sample((1.0, -1.0), source_index=1))

    startup_commands = np.asarray(backend.command_history[:3])
    startup_deltas = np.diff(np.vstack([np.zeros(2), startup_commands]), axis=0)
    assert startup_commands.shape == (3, 2)
    np.testing.assert_allclose(startup_commands[-1], [0.11, -0.21])
    assert np.all(np.abs(startup_deltas) <= np.array([0.05, 0.1]) + 1e-12)
    np.testing.assert_allclose(startup_deltas, np.repeat(startup_deltas[:1], 3, axis=0))
    np.testing.assert_allclose(backend.command_history[-1], [1.0, -1.0])
    assert startup.diagnostics["command_periods_advanced"] == 3.0
    assert startup.diagnostics["startup_move_active"] == 1.0
    assert direct.diagnostics["command_periods_advanced"] == 1.0
    assert flow.command_period_count == 4
    assert len(observed_commands) == 4


def test_missing_and_premapping_frames_hold_without_consuming_startup_count():
    """Waiting and missing samples each hold one period and preserve startup state."""
    backend = _FakeBackend()
    backend.target = np.array([0.2, -0.2])
    backend.actual = backend.target.copy()
    mapper = _FakeMapper((False, True))
    flow, _, _, _, _, _ = _build_flow(backend=backend, mapper=mapper, startup_move_frames=1)

    waiting = flow.step(_sample((0.1, -0.1)))
    startup = flow.step(_sample((0.1, -0.1), source_index=1))
    missing = flow.step(_sample(None, source_index=2))

    assert waiting.status is ExecutionStatus.WAITING_FOR_MAPPING
    np.testing.assert_allclose(waiting.command_qpos, [0.2, -0.2])
    assert startup.status is ExecutionStatus.EXECUTED
    assert startup.diagnostics["startup_move_active"] == 1.0
    assert missing.status is ExecutionStatus.HELD
    assert flow.retarget_frame_count == 1
    assert flow.source_frame_count == 3
    assert backend.simulation_time == pytest.approx(0.2)


def test_kinematic_backend_preserves_clipping_startup_and_hold_semantics():
    """Ideal execution visualizes every bounded waypoint and missing-frame hold.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.backends import KinematicRobotBackend

    clock = _FakeClock()
    backend = KinematicRobotBackend(initial_qpos=np.zeros(2), control_period=0.05)
    flow, _, _, _, _, _ = _build_flow(
        backend=backend,
        clock=clock,
        realtime=True,
        startup_move_frames=1,
    )
    observed_commands: list[tuple[np.ndarray, np.ndarray]] = []
    flow.add_command_observer(
        lambda result: observed_commands.append((result.command_qpos.copy(), result.actual_qpos.copy()))
    )

    startup = flow.step(_sample((3.0, -3.0)))
    held = flow.step(_sample(None, source_index=1))

    assert startup.status is ExecutionStatus.EXECUTED
    assert startup.diagnostics["command_periods_advanced"] == 40.0
    np.testing.assert_allclose(startup.command_qpos, [2.0, -2.0])
    np.testing.assert_allclose(startup.actual_qpos, [2.0, -2.0])
    assert held.status is ExecutionStatus.HELD
    np.testing.assert_allclose(held.command_qpos, [2.0, -2.0])
    assert len(observed_commands) == 41
    for command_qpos, actual_qpos in observed_commands:
        np.testing.assert_allclose(actual_qpos, command_qpos)
    np.testing.assert_allclose(observed_commands[-1][1], [2.0, -2.0])
    assert flow.command_period_count == 41
    assert clock.now == pytest.approx(2.05)


def test_flow_reset_synchronizes_qpos_state_counters_input_and_mapping():
    """The single reset entrypoint synchronizes all stateful dependencies."""
    from unittest.mock import Mock

    backend = _FakeBackend()
    mapper = _FakeMapper()
    flow, _, _, retargeter, output_filter, command_policy = _build_flow(backend=backend, mapper=mapper)
    fake_input = Mock()
    flow.input = fake_input
    reset_states: list[np.ndarray] = []
    flow.add_reset_observer(lambda qpos: reset_states.append(qpos.copy()))
    flow.step(_sample((0.5, -0.5)))

    flow.reset(np.array([0.3, -0.4]))

    np.testing.assert_allclose(backend.get_target_joint_pos(), [0.3, -0.4])
    np.testing.assert_allclose(retargeter.reset_history[-1], [0.3, -0.4])
    np.testing.assert_allclose(output_filter.previous_qpos, [0.3, -0.4])
    np.testing.assert_allclose(command_policy.previous_qpos, [0.3, -0.4])
    fake_input.reset.assert_called_once_with()
    assert mapper.reset_count == 1
    assert flow.mapping_initialized is False
    assert (flow.source_frame_count, flow.retarget_frame_count, flow.command_period_count) == (0, 0, 0)
    np.testing.assert_allclose(reset_states[-1], [0.3, -0.4])


def test_filtered_qpos_is_next_temporal_reference_but_evaluation_uses_raw_result():
    """Preserve filter and evaluation ordering around the pure solver boundary."""
    evaluator = _FakeEvaluator()
    flow, _, _, retargeter, _, _ = _build_flow(smooth=True, evaluator=evaluator)

    first = flow.step(_sample((1.0, -1.0)))
    flow.step(_sample((0.5, -0.5), source_index=1))

    np.testing.assert_allclose(first.retargeted_frame.retargeted_qpos, [0.3, -0.3])
    np.testing.assert_allclose(evaluator.raw_qpos[0], [1.0, -1.0])
    np.testing.assert_allclose(retargeter.previous_references[1], [0.3, -0.3])
    assert first.retargeted_frame.diagnostics["position_err"] == 1.0


def test_execution_flow_accounts_for_backend_timing_overrun():
    """Injected timing reports compute overrun without a negative sleep."""
    clock = _FakeClock()
    backend = _FakeBackend(clock=clock, compute_time=0.06)
    flow, _, _, _, _, _ = _build_flow(backend=backend, clock=clock, realtime=True)

    result = flow.step(_sample((0.1, -0.1)))

    assert result.diagnostics["runtime_compute_time"] == pytest.approx(0.06)
    assert result.diagnostics["runtime_overrun"] == pytest.approx(0.01)
    assert result.diagnostics["runtime_wall_time"] == pytest.approx(0.06)
    assert clock.now == pytest.approx(0.06)


def test_pull_run_resets_and_realigns_between_finite_cycles():
    """Loop playback performs one full reset before reading the next cycle."""
    hand_input = _FiniteInput()
    mapper = _FakeMapper()
    flow, backend, _, retargeter, _, _ = _build_flow(mapper=mapper)
    flow.input = hand_input
    flow.loop = True
    flow.max_frames = 3

    summary = flow.run()

    assert summary.source_frames_processed == 3
    assert summary.retarget_frames_processed == 3
    assert summary.command_periods_advanced == 3
    assert summary.cycles_completed == 2
    assert hand_input.open_count == 1
    assert hand_input.reset_count == 1
    assert hand_input.close_count == 1
    assert mapper.reset_count == 1
    assert len(mapper.initialize_qpos) == 2
    assert len(backend.reset_history) == 1
    assert len(retargeter.reset_history) == 1


def test_execution_results_detach_mutable_backend_and_diagnostic_arrays():
    """Published results remain stable when producers mutate their buffers."""
    flow, backend, _, _, _, _ = _build_flow()

    result = flow.step(_sample((0.2, -0.3)))
    backend.target[:] = 9.0

    np.testing.assert_allclose(result.command_qpos, [0.2, -0.3])
    with pytest.raises(ValueError):
        result.command_qpos[0] = 1.0


def test_flat_flow_source_has_no_nested_controller_or_artifact_dependency():
    """Protect the final flow from old controllers and application artifacts."""
    flow_source = Path("src/teleoperation/flow.py").read_text(encoding="utf-8")
    composition_source = Path("src/retargeting_apps/composition.py").read_text(encoding="utf-8")
    teleop_exe_source = Path("src/retargeting_apps/apps/teleop_exe.py").read_text(encoding="utf-8")

    for obsolete in (
        "TeleoperationSession",
        "MujocoTeleoperationRuntime",
        "AlignedMujocoTeleoperationDriver",
        "MujocoStepResult",
    ):
        assert obsolete not in flow_source
        assert obsolete not in composition_source
    assert "retargeting_apps.artifacts" not in flow_source
    assert "def build_execution_flow(" in composition_source
    assert "build_execution_flow(config_data)" in teleop_exe_source
    assert "flow.run()" in teleop_exe_source
    assert not Path("src/retargeting_apps/pipelines").exists()
    assert not Path("src/teleoperation/session.py").exists()
    assert not Path("src/teleoperation/mujoco_runtime.py").exists()
    assert not Path("src/retargeting_apps/apps/mujoco_online_simulation.py").exists()
    assert not Path("src/retargeting_apps/apps/mujoco_offline_simulation.py").exists()
