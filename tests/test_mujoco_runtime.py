from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeting.core.types import HandObservation


class _FakeClock:
    """Deterministic wall clock used to test real-time pacing."""

    def __init__(self) -> None:
        """Initialize wall time at zero.

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
        """Advance fake wall time by a requested sleep duration.

        Args:
            duration: Non-negative duration in seconds.

        Returns:
            None.
        """
        self.now += duration


class _FakeBackend:
    """Minimal deterministic backend with a 20 Hz command period."""

    control_period = 0.05

    def __init__(self) -> None:
        """Initialize a two-joint state.

        Args:
            None.

        Returns:
            None.
        """
        self.target = np.zeros(2)
        self.actual = np.zeros(2)
        self.simulation_time = 0.0
        self.command_history = []

    def reset(self, qpos=None) -> None:
        """Reset target and actual joint state.

        Args:
            qpos: Optional two-joint reset position.

        Returns:
            None.
        """
        self.target = np.zeros(2) if qpos is None else np.asarray(qpos, dtype=float).copy()
        self.actual = self.target.copy()
        self.simulation_time = 0.0

    def ctrl_joint_pos(self, qpos) -> np.ndarray:
        """Accept one target command.

        Args:
            qpos: Two-joint target command.

        Returns:
            Applied target command.
        """
        self.target = np.asarray(qpos, dtype=float).copy()
        self.command_history.append(self.target.copy())
        return self.target.copy()

    def step(self) -> None:
        """Advance one command period and realize the target exactly.

        Args:
            None.

        Returns:
            None.
        """
        self.actual = self.target.copy()
        self.simulation_time += self.control_period

    def get_joint_pos(self) -> np.ndarray:
        """Return current actual positions.

        Args:
            None.

        Returns:
            Two-joint actual state.
        """
        return self.actual.copy()

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the current target positions.

        Args:
            None.

        Returns:
            Two-joint target state.
        """
        return self.target.copy()

    def get_diagnostics(self) -> dict[str, float]:
        """Return simulated time for runtime diagnostic merging.

        Args:
            None.

        Returns:
            Mapping containing simulation time.
        """
        return {"simulation_time": self.simulation_time}


class _FakeSession:
    """Session that exposes the requested qpos through observation keypoints."""

    def __init__(self) -> None:
        """Initialize an empty reset record.

        Args:
            None.

        Returns:
            None.
        """
        self.reset_qpos = []

    def reset(self, qpos=None) -> None:
        """Record the runtime reset configuration.

        Args:
            qpos: Optional reset configuration supplied by the runtime.

        Returns:
            None.
        """
        self.reset_qpos.append(None if qpos is None else np.asarray(qpos, dtype=float).copy())

    def retarget_observation(self, observation: HandObservation):
        """Return a two-joint command encoded in the first keypoint.

        Args:
            observation: Canonical observation carrying the requested command.

        Returns:
            Requested qpos and deterministic diagnostics.
        """
        return observation.keypoints_wrist[0, :2].copy(), {"optimization_time": 0.001}

    def detect_observation(self, sensor_data, camera_K=None):
        """Treat a canonical observation as already detected sensor data.

        Args:
            sensor_data: Observation instance or None for a missed detection.
            camera_K: Unused camera intrinsics.

        Returns:
            Supplied observation or None.
        """
        del camera_K
        return sensor_data


def _observation(qpos: tuple[float, float]) -> HandObservation:
    """Build a canonical observation carrying a requested two-joint command.

    Args:
        qpos: Requested command encoded into the first keypoint.

    Returns:
        Canonical hand observation for the fake session.
    """
    keypoints = np.zeros((21, 3))
    keypoints[0, :2] = qpos
    return HandObservation(keypoints_wrist=keypoints, wrist_pose_world=np.eye(4))


def test_mujoco_runtime_direct_commands_have_no_explicit_speed_limit():
    """Verify direct execution sends every requested target in one command period.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
    from teleoperation.output import QposCommandLimiter

    clock = _FakeClock()
    backend = _FakeBackend()
    limiter = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.array([1.0, 2.0]),
        command_hz=20.0,
        lower=np.array([-2.0, -2.0]),
        upper=np.array([2.0, 2.0]),
    )
    runtime = MujocoTeleoperationRuntime(
        session=_FakeSession(),
        backend=backend,
        command_limiter=limiter,
        realtime=True,
        clock=clock,
        sleep=clock.sleep,
    )

    first = runtime.step_observation(_observation((1.0, -1.0)))
    for _ in range(19):
        last = runtime.step_observation(_observation((1.0, -1.0)))

    np.testing.assert_allclose(first.command_qpos, [1.0, -1.0])
    np.testing.assert_allclose(last.command_qpos, [1.0, -1.0])
    assert runtime.frame_count == 20
    assert runtime.command_step_count == 20
    assert runtime.retarget_frame_count == 20
    assert backend.simulation_time == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)
    assert last.diagnostics["runtime_overrun"] == 0.0
    assert last.diagnostics["command_periods_advanced"] == 1.0
    assert last.diagnostics["startup_move_active"] == 0.0


def test_mujoco_runtime_blocks_on_synchronized_startup_waypoints_then_switches_to_direct():
    """Verify startup frames finish strict linear waypoints before direct execution.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
    from teleoperation.output import QposCommandLimiter

    clock = _FakeClock()
    backend = _FakeBackend()
    observer_states = []
    limiter = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.array([1.0, 2.0]),
        command_hz=20.0,
        lower=np.array([-2.0, -2.0]),
        upper=np.array([2.0, 2.0]),
    )
    runtime = MujocoTeleoperationRuntime(
        session=_FakeSession(),
        backend=backend,
        command_limiter=limiter,
        realtime=True,
        startup_move_frames=1,
        post_command_step=lambda: observer_states.append(backend.get_joint_pos()),
        clock=clock,
        sleep=clock.sleep,
    )

    startup = runtime.step_observation(_observation((0.11, -0.21)))
    direct = runtime.step_observation(_observation((1.0, -1.0)))

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
    assert direct.diagnostics["startup_move_active"] == 0.0
    assert runtime.frame_count == 2
    assert runtime.command_step_count == 4
    assert runtime.retarget_frame_count == 2
    assert backend.simulation_time == pytest.approx(0.2)
    assert clock.now == pytest.approx(0.2)
    assert len(observer_states) == 4


def test_mujoco_runtime_holds_command_when_detection_is_missing():
    """Verify a missing hand still advances physics without a new retarget result.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
    from teleoperation.output import QposCommandLimiter

    backend = _FakeBackend()
    backend.ctrl_joint_pos(np.array([0.2, -0.2]))
    limiter = QposCommandLimiter(
        initial_qpos=backend.get_target_joint_pos(),
        max_joint_speed=np.ones(2),
        command_hz=20.0,
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
    )
    runtime = MujocoTeleoperationRuntime(_FakeSession(), backend, limiter, realtime=False)

    result = runtime.step_sensor_data(None)

    assert result.observation is None
    assert result.requested_qpos is None
    np.testing.assert_allclose(result.command_qpos, [0.2, -0.2])
    assert backend.simulation_time == 0.05


def test_mujoco_runtime_hold_does_not_consume_a_startup_move_frame():
    """Verify only a successfully retargeted target advances the startup counter.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
    from teleoperation.output import QposCommandLimiter

    backend = _FakeBackend()
    limiter = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.ones(2),
        command_hz=20.0,
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
    )
    runtime = MujocoTeleoperationRuntime(
        _FakeSession(),
        backend,
        limiter,
        realtime=False,
        startup_move_frames=1,
    )

    hold = runtime.step_sensor_data(None)
    startup = runtime.step_observation(_observation((0.1, -0.1)))

    assert hold.diagnostics["startup_move_active"] == 0.0
    assert startup.diagnostics["startup_move_active"] == 1.0
    assert startup.diagnostics["command_periods_advanced"] == 2.0
    assert runtime.frame_count == 2
    assert runtime.retarget_frame_count == 1
    assert runtime.command_step_count == 3


def test_mujoco_runtime_reset_synchronizes_backend_session_and_limiter():
    """Verify reset removes temporal state from every command-layer owner.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
    from teleoperation.output import QposCommandLimiter

    backend = _FakeBackend()
    session = _FakeSession()
    limiter = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.ones(2),
        command_hz=20.0,
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
    )
    runtime = MujocoTeleoperationRuntime(
        session,
        backend,
        limiter,
        realtime=False,
        startup_move_frames=1,
    )
    runtime.step_observation(_observation((1.0, -1.0)))

    runtime.reset()

    assert runtime.frame_count == 0
    assert runtime.command_step_count == 0
    assert runtime.retarget_frame_count == 0
    assert backend.simulation_time == 0.0
    np.testing.assert_allclose(backend.get_joint_pos(), np.zeros(2))
    np.testing.assert_allclose(backend.get_target_joint_pos(), np.zeros(2))
    np.testing.assert_allclose(limiter.previous_qpos, np.zeros(2))
    assert len(session.reset_qpos) == 1
    np.testing.assert_allclose(session.reset_qpos[0], np.zeros(2))


def test_aligned_mujoco_driver_shares_prealignment_hold_and_reset_policy():
    """Verify live and offline callers can share alignment lifecycle behavior.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.mujoco_runtime import (
        AlignedMujocoTeleoperationDriver,
        MujocoTeleoperationRuntime,
    )
    from teleoperation.output import QposCommandLimiter

    backend = _FakeBackend()
    session = _FakeSession()
    limiter = QposCommandLimiter(
        initial_qpos=np.zeros(2),
        max_joint_speed=np.ones(2),
        command_hz=20.0,
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
    )
    runtime = MujocoTeleoperationRuntime(session, backend, limiter, realtime=False)
    alignment_results = iter((False, True, True))
    alignment_frames = []

    def initialize_alignment(session_arg, sensor_data, robot_qpos):
        """Record alignment attempts and return the configured result.

        Args:
            session_arg: Session supplied by the shared driver.
            sensor_data: Raw frame used for the alignment attempt.
            robot_qpos: Current backend positions supplied by the driver.

        Returns:
            Next deterministic alignment outcome.
        """
        assert session_arg is session
        np.testing.assert_allclose(robot_qpos, backend.get_joint_pos())
        alignment_frames.append(sensor_data)
        return next(alignment_results)

    driver = AlignedMujocoTeleoperationDriver(runtime, initialize_alignment)
    first_frame = _observation((0.1, -0.1))
    second_frame = _observation((0.2, -0.2))
    third_frame = _observation((0.3, -0.3))

    first = driver.step(first_frame)
    second = driver.step(second_frame)
    driver.step(third_frame)

    assert first.observation is None
    assert second.observation is second_frame
    assert alignment_frames == [first_frame, second_frame]
    assert runtime.frame_count == 3
    assert backend.simulation_time == pytest.approx(0.15)

    driver.reset()
    reset_frame = _observation((0.4, -0.4))
    reset_result = driver.step(reset_frame)

    assert alignment_frames == [first_frame, second_frame, reset_frame]
    assert reset_result.observation is reset_frame
    assert runtime.frame_count == 1
    assert backend.simulation_time == pytest.approx(0.05)
    assert len(session.reset_qpos) == 1


def test_mujoco_runtime_source_has_no_artifact_replay_dependency():
    """Protect the shared runtime from trajectory and replay coupling.

    Args:
        None.

    Returns:
        None.
    """
    source = Path("src/teleoperation/mujoco_runtime.py").read_text(encoding="utf-8")

    assert "retargeting_apps.artifacts" not in source
    assert "offline_retargeting" not in source
    assert "retarget_qpos" not in source
    assert "teleoperation.avp_alignment" not in source
    assert "parse_avp_stream_frame" not in source


def test_online_and_offline_apps_share_neutral_mujoco_runtime_builder():
    """Keep shared construction outside both mode-specific application runners.

    Args:
        None.

    Returns:
        None.
    """
    builder_source = Path("src/retargeting_apps/pipelines/mujoco_runtime_builder.py").read_text(encoding="utf-8")
    runtime_source = Path("src/teleoperation/mujoco_runtime.py").read_text(encoding="utf-8")
    online_path = Path("src/retargeting_apps/apps/mujoco_online_simulation.py")
    online_source = online_path.read_text(encoding="utf-8")
    offline_source = Path("src/retargeting_apps/apps/mujoco_offline_simulation.py").read_text(encoding="utf-8")

    assert online_path.is_file()
    assert not Path("src/retargeting_apps/apps/mujoco_simulation.py").exists()
    assert "def run(" in online_source
    assert "def build_mujoco_runtime(" in builder_source
    assert "def build_mujoco_runtime(" not in online_source
    assert "def build_mujoco_runtime(" not in offline_source
    assert "mujoco_runtime_builder.build_mujoco_runtime" in online_source
    assert "mujoco_runtime_builder.build_mujoco_runtime" in offline_source
    assert "class AlignedMujocoTeleoperationDriver" in runtime_source
    assert "AlignedMujocoTeleoperationDriver(" in online_source
    assert "AlignedMujocoTeleoperationDriver(" in offline_source
    assert "driver.step(sensor_data)" in online_source
    assert "driver.step(sensor_data)" in offline_source
    assert "alignment_initialized" not in online_source
    assert "initialize_avp_runtime_alignment" not in online_source
    assert "initialize_avp_runtime_alignment" not in offline_source
    assert not Path("src/teleoperation/offline.py").exists()
    assert "mujoco_online_simulation import build_mujoco_runtime" not in offline_source
