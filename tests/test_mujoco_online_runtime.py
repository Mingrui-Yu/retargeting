from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeting.inputs import HandObservation


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


def test_online_runtime_retargets_then_steps_each_frame_at_20_hz():
    """Verify 20 immediate retarget-command-step cycles advance one simulated second.

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

    np.testing.assert_allclose(first.command_qpos, [0.05, -0.1])
    np.testing.assert_allclose(last.command_qpos, [1.0, -1.0])
    assert runtime.frame_count == 20
    assert backend.simulation_time == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)
    assert last.diagnostics["runtime_overrun"] == 0.0


def test_online_runtime_holds_command_when_detection_is_missing():
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


def test_online_runtime_source_has_no_artifact_replay_dependency():
    """Protect the online simulation path from trajectory/replay coupling.

    Args:
        None.

    Returns:
        None.
    """
    source = Path("src/teleoperation/mujoco_runtime.py").read_text(encoding="utf-8")

    assert "retargeting.artifacts" not in source
    assert "offline_retargeting" not in source
    assert "retarget_qpos" not in source
