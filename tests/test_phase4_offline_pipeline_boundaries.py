from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from retargeting.core.types import HandObservation, RetargetingResult


REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation(qpos: tuple[float, float]) -> HandObservation:
    """Build a canonical observation carrying a compact fake target.

    Args:
        qpos: Two values encoded in the first hand keypoint.

    Returns:
        Canonical hand observation for computation-only tests.
    """
    keypoints = np.zeros((21, 3))
    keypoints[0, :2] = qpos
    return HandObservation(keypoints_wrist=keypoints, wrist_pose_world=np.eye(4))


class _FakeRetargeter:
    """Record temporal references passed by the pure sequence helper."""

    def __init__(self) -> None:
        """Initialize an empty temporal-reference record.

        Args:
            None.

        Returns:
            None.
        """
        self.previous_references: list[np.ndarray | None] = []

    def solve(
        self,
        observation: HandObservation,
        previous_qpos: np.ndarray | None = None,
    ) -> RetargetingResult:
        """Return the target encoded in one canonical observation.

        Args:
            observation: Canonical observation containing a fake qpos target.
            previous_qpos: Temporal reference supplied by sequence processing.

        Returns:
            Fake retargeting result and a deterministic metric.
        """
        self.previous_references.append(None if previous_qpos is None else previous_qpos.copy())
        qpos = observation.keypoints_wrist[0, :2].copy()
        return RetargetingResult(qpos=qpos, diagnostics={"target_norm": float(np.linalg.norm(qpos))})


def test_core_retargets_canonical_sequence_with_explicit_temporal_state():
    """Verify the computation-only sequence propagates each raw result.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.core.sequence import retarget_observation_sequence

    retargeter = _FakeRetargeter()
    results = retarget_observation_sequence(
        [_observation((1.0, -1.0)), _observation((0.25, 0.5))],
        retargeter,
        previous_qpos=np.array([0.1, -0.1]),
    )

    np.testing.assert_allclose(retargeter.previous_references[0], [0.1, -0.1])
    np.testing.assert_allclose(retargeter.previous_references[1], [1.0, -1.0])
    np.testing.assert_allclose(results[0].qpos, [1.0, -1.0])
    np.testing.assert_allclose(results[1].qpos, [0.25, 0.5])
    assert results[1].diagnostics["target_norm"] == np.linalg.norm([0.25, 0.5])


class _FakeRobotAdaptor:
    """Expose an identity actuated-to-model qpos mapping for service tests."""

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Return model positions unchanged.

        Args:
            qpos: Actuated-joint positions.

        Returns:
            Identical fake model positions.
        """
        return np.asarray(qpos, dtype=float).copy()


class _FakeRobotModel:
    """Return a deterministic initial robot wrist pose."""

    def get_frame_pose(self, frame_name: str, qpos: np.ndarray) -> np.ndarray:
        """Build a wrist pose from fake model positions.

        Args:
            frame_name: Requested robot frame name.
            qpos: Fake model positions.

        Returns:
            Deterministic robot wrist pose.
        """
        assert frame_name == "wrist"
        pose = np.eye(4)
        pose[:2, 3] = qpos[:2]
        return pose


class _FakeOfflineSession:
    """Minimal session exposing AVP alignment boundaries."""

    def __init__(self) -> None:
        """Initialize fake robot and frame result state.

        Args:
            None.

        Returns:
            None.
        """
        self.robot_adaptor = _FakeRobotAdaptor()
        self.robot_model = _FakeRobotModel()
        self.robot_config = SimpleNamespace(wrist_frame_name="wrist")
        self.robot_initial_wrist_pose = None
        self.detection_initial_wrist_pose = None

    def set_robot_init_wrist_pose(self, pose: np.ndarray) -> None:
        """Record the robot alignment origin.

        Args:
            pose: Initial robot wrist pose.

        Returns:
            None.
        """
        self.robot_initial_wrist_pose = pose.copy()

    def set_detection_source_init_wrist_pose(self, pose: np.ndarray) -> None:
        """Record the calibrated detector alignment origin.

        Args:
            pose: Initial detector wrist pose in robot world coordinates.

        Returns:
            None.
        """
        self.detection_initial_wrist_pose = pose.copy()

    def pose_from_detection_world_to_robot_world(self, pose: np.ndarray) -> np.ndarray:
        """Apply a deterministic fake detector calibration.

        Args:
            pose: Wrist pose in detector world coordinates.

        Returns:
            Wrist pose translated into fake robot world coordinates.
        """
        result = pose.copy()
        result[2, 3] += 0.5
        return result


def _raw_avp_frame() -> dict[str, np.ndarray]:
    """Build a valid raw AVP frame for alignment parsing.

    Args:
        None.

    Returns:
        Raw wrist and finger transforms accepted by the AVP parser.
    """
    return {
        "right_wrist": np.eye(4)[None, :, :],
        "right_fingers": np.repeat(np.eye(4)[None, :, :], 25, axis=0),
    }


def test_avp_alignment_uses_the_selected_robot_pose():
    """Verify the shared helper aligns a session without an offline wrapper.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.avp_alignment import initialize_avp_alignment

    session = _FakeOfflineSession()

    assert initialize_avp_alignment(
        session,
        _raw_avp_frame(),
        np.array([0.2, -0.3]),
    ) is True
    np.testing.assert_allclose(session.robot_initial_wrist_pose[:2, 3], [0.2, -0.3])
    assert session.detection_initial_wrist_pose[2, 3] == 0.5


def test_phase4_source_ownership_separates_core_runtime_and_app_orchestration():
    """Keep I/O, device policy, and app orchestration in their declared layers.

    Args:
        None.

    Returns:
        None.
    """
    core_source = (REPO_ROOT / "src" / "retargeting" / "core" / "sequence.py").read_text(encoding="utf-8")
    alignment_source = (REPO_ROOT / "src" / "teleoperation" / "avp_alignment.py").read_text(encoding="utf-8")
    app_source = (
        REPO_ROOT / "src" / "retargeting_apps" / "pipelines" / "offline_retargeting.py"
    ).read_text(encoding="utf-8")

    for forbidden_token in (
        "np.load",
        "pathlib",
        "teleoperation",
        "retargeting_apps",
        "tqdm",
        "viser",
    ):
        assert forbidden_token not in core_source

    assert "parse_avp_stream_frame" in alignment_source
    assert "TeleoperationSession" in alignment_source
    assert alignment_source.count("def initialize_avp_alignment(") == 1
    assert "initialize_avp_session_alignment" not in alignment_source
    assert "initialize_avp_runtime_alignment" not in alignment_source
    assert "OfflineRetargetingService" not in alignment_source
    assert "OfflineRetargetedFrame" not in alignment_source
    assert "OfflineRetargetingService" not in app_source
    assert "initialize_avp_alignment" in app_source
    assert "TeleoperationSession" in app_source
    assert "session.retarget_input" in app_source
    assert "if observation is None or qpos is None or diagnostics is None:" in app_source
    assert "load_offline_avp_trajectory" in app_source
    assert "tqdm" in app_source
    for moved_runtime_concern in (
        "parse_avp_stream_frame",
        "Retargeter",
        "QposOutputFilter",
        "load_offline_replay",
    ):
        assert moved_runtime_concern not in app_source
