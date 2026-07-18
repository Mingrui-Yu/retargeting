from __future__ import annotations

from pathlib import Path
import numpy as np

from retargeting.core.types import RetargetingHandObservation, RetargetingResult
from teleoperation.types import SensorHandSample


REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation(qpos: tuple[float, float]) -> RetargetingHandObservation:
    """Build a canonical observation carrying a compact fake target.

    Args:
        qpos: Two values encoded in the first hand keypoint.

    Returns:
        Canonical hand observation for computation-only tests.
    """
    keypoints = np.zeros((21, 3))
    keypoints[0, :2] = qpos
    return RetargetingHandObservation(keypoints_wrist=keypoints, wrist_pose_world=np.eye(4))


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
        observation: RetargetingHandObservation,
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

    def reset(self, previous_qpos: np.ndarray | None = None) -> None:
        """Accept batch reset state for protocol completeness.

        Args:
            previous_qpos: Optional reset reference.

        Returns:
            None.
        """
        del previous_qpos


class _FiniteBatchInput:
    """Finite sensor input for batch skip and lifecycle tests."""

    def __init__(self, samples: list[SensorHandSample]) -> None:
        """Store selected samples and initialize lifecycle state.

        Args:
            samples: Samples returned before finite end-of-stream.

        Returns:
            None.
        """
        self.samples = samples
        self.cursor = 0
        self.closed = False

    def open(self) -> None:
        """Open the source at its first selected sample.

        Args:
            None.

        Returns:
            None.
        """
        self.cursor = 0
        self.closed = False

    def read(self) -> SensorHandSample | None:
        """Read one selected sample or finite end-of-stream.

        Args:
            None.

        Returns:
            Next sample, or None at source end.
        """
        if self.cursor >= len(self.samples):
            return None
        sample = self.samples[self.cursor]
        self.cursor += 1
        return sample

    def reset(self) -> None:
        """Rewind the selected sample list.

        Args:
            None.

        Returns:
            None.
        """
        self.cursor = 0

    def close(self) -> None:
        """Record batch source closure.

        Args:
            None.

        Returns:
            None.
        """
        self.closed = True


def _sensor_sample(qpos: tuple[float, float] | None, source_index: int) -> SensorHandSample:
    """Build a complete or missing batch sensor sample.

    Args:
        qpos: Target encoded in keypoints, or None for missing hand data.
        source_index: Source frame metadata.

    Returns:
        Sensor-normalized test sample.
    """
    if qpos is None:
        return SensorHandSample(None, None, raw=None, source_index=source_index)
    keypoints = np.zeros((21, 3))
    keypoints[0, :2] = qpos
    return SensorHandSample(keypoints, np.eye(4), raw={}, source_index=source_index)


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


def test_batch_flow_skips_missing_frames_without_execution_results():
    """Batch output contains valid retargeting records and no hold placeholders."""
    from teleoperation.config import load_teleoperation_mode_config
    from teleoperation.flow import BatchRetargetFlow
    from teleoperation.observation_mapping import IdentityHandObservationMapper
    from teleoperation.output import QposOutputFilter
    from teleoperation.types import RetargetedFrameResult

    hand_input = _FiniteBatchInput(
        [_sensor_sample((0.1, -0.1), 0), _sensor_sample(None, 1), _sensor_sample((0.2, -0.2), 2)]
    )
    retargeter = _FakeRetargeter()
    output_filter = QposOutputFilter(np.zeros(2), load_teleoperation_mode_config(None))
    flow = BatchRetargetFlow(
        input=hand_input,
        observation_mapper=IdentityHandObservationMapper(),
        retargeter=retargeter,
        output_filter=output_filter,
        initial_robot_qpos=np.zeros(2),
    )

    records = flow.run()

    assert all(isinstance(record, RetargetedFrameResult) for record in records)
    assert [record.source_index for record in records] == [0, 2]
    assert hand_input.closed is True


def test_batch_flow_reports_first_frame_mapping_failure_and_closes_input():
    """Batch mapping initialization failure is explicit and still closes input."""
    import pytest

    from teleoperation.config import load_teleoperation_mode_config
    from teleoperation.flow import BatchRetargetFlow
    from teleoperation.observation_mapping import IdentityHandObservationMapper
    from teleoperation.output import QposOutputFilter

    hand_input = _FiniteBatchInput([_sensor_sample(None, 4)])
    flow = BatchRetargetFlow(
        input=hand_input,
        observation_mapper=IdentityHandObservationMapper(),
        retargeter=_FakeRetargeter(),
        output_filter=QposOutputFilter(np.zeros(2), load_teleoperation_mode_config(None)),
        initial_robot_qpos=np.zeros(2),
    )

    with pytest.raises(ValueError, match="source frame 4"):
        flow.run()

    assert hand_input.closed is True


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


def test_avp_mapper_alignment_uses_the_selected_robot_pose():
    """Verify AVP mapping initializes without a session or backend dependency.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.config import DetectionSourceConfig
    from teleoperation.inputs.avp import decode_avp_sample
    from teleoperation.observation_mapping import AvpRelativeWristMapper

    config = DetectionSourceConfig(
        name="avp",
        input_device="avp",
        rotation_euler_xyz_deg=(0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 0.5),
        use_relative_wrist_alignment=True,
    )
    mapper = AvpRelativeWristMapper(config, 1.0, _FakeRobotAdaptor(), _FakeRobotModel(), "wrist")
    sample = decode_avp_sample(_raw_avp_frame())

    assert mapper.initialize(sample, np.array([0.2, -0.3])) is True
    np.testing.assert_allclose(mapper._robot_initial_wrist_pose[:2, 3], [0.2, -0.3])
    assert mapper._sensor_initial_wrist_pose[2, 3] == 0.5


def test_phase4_source_ownership_separates_core_runtime_and_app_orchestration():
    """Keep I/O, device policy, and app orchestration in their declared layers.

    Args:
        None.

    Returns:
        None.
    """
    core_source = (REPO_ROOT / "src" / "retargeting" / "core" / "sequence.py").read_text(encoding="utf-8")
    mapper_source = (REPO_ROOT / "src" / "teleoperation" / "observation_mapping.py").read_text(encoding="utf-8")
    flow_source = (REPO_ROOT / "src" / "teleoperation" / "flow.py").read_text(encoding="utf-8")
    batch_source = flow_source.split("class BatchRetargetFlow:", maxsplit=1)[1].split(
        "class ExecutionFlow:", maxsplit=1
    )[0]
    app_source = (REPO_ROOT / "src" / "retargeting_apps" / "offline_retargeting.py").read_text(encoding="utf-8")

    for forbidden_token in (
        "np.load",
        "pathlib",
        "teleoperation",
        "retargeting_apps",
        "tqdm",
        "viser",
    ):
        assert forbidden_token not in core_source

    assert "class AvpRelativeWristMapper" in mapper_source
    assert "TeleoperationSession" not in mapper_source
    assert "class BatchRetargetFlow" in flow_source
    assert "RobotBackend" not in batch_source
    assert "ExecutionStepResult" not in batch_source
    assert "build_batch_retarget_flow" in app_source
    assert "TeleoperationSession" not in app_source
    assert "session.retarget_input" not in app_source
    assert "retargeted_qpos" in app_source
    for moved_runtime_concern in (
        "decode_avp_sample",
        "Retargeter",
        "QposOutputFilter",
        "RobotBackend",
    ):
        assert moved_runtime_concern not in app_source
