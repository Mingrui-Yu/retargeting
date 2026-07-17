from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _TrackingNpz:
    """NPZ-like object that records which arrays the loader materializes."""

    files = ["stream_right_wrist", "stream_right_fingers", "retarget_qpos"]

    def __init__(self) -> None:
        """Create two raw frames and a forbidden robot-qpos array.

        Args:
            None.

        Returns:
            None.
        """
        self.accessed = []
        self.arrays = {
            "stream_right_wrist": np.repeat(np.eye(4)[None, None, :, :], 2, axis=0),
            "stream_right_fingers": np.zeros((2, 25, 4, 4)),
            "retarget_qpos": np.zeros((2, 23)),
        }

    def __enter__(self):
        """Return this fake archive from a context manager.

        Args:
            None.

        Returns:
            This tracking archive.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Leave the fake archive context without suppressing errors.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception instance.
            traceback: Optional exception traceback.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback

    def __getitem__(self, key):
        """Record and return one requested NPZ array.

        Args:
            key: Array name requested by the loader.

        Returns:
            Requested test array.
        """
        self.accessed.append(key)
        return self.arrays[key]


def test_offline_human_loader_never_accesses_retarget_qpos(monkeypatch):
    """Verify the dedicated loader materializes only raw human streams.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting.inputs import offline_avp_replay

    archive = _TrackingNpz()
    monkeypatch.setattr(offline_avp_replay.np, "load", lambda path: archive)

    trajectory = offline_avp_replay.load_offline_human_trajectory("human.npz")

    assert archive.accessed == ["stream_right_wrist", "stream_right_fingers"]
    assert trajectory.n_frames == 2
    assert set(trajectory.get_frame(0)) == {"right_wrist", "right_fingers"}


class _FakeOfflineRuntime:
    """Runtime fake that advances 0.05 seconds for every source frame."""

    def __init__(self) -> None:
        """Initialize frame records and simulated time.

        Args:
            None.

        Returns:
            None.
        """
        self.frame_count = 0
        self.simulation_time = 0.0
        self.sensor_frames = []
        self.hold_count = 0

    def _result(self):
        """Advance one command period and build a runtime-like result.

        Args:
            None.

        Returns:
            Namespace containing simulation diagnostics.
        """
        self.frame_count += 1
        self.simulation_time += 0.05
        return SimpleNamespace(
            diagnostics={"simulation_time": self.simulation_time, "tracking_error_max": 0.0}
        )

    def step_sensor_data(self, sensor_data):
        """Record and execute one raw source frame.

        Args:
            sensor_data: Raw source frame mapping.

        Returns:
            Runtime-like result for the frame.
        """
        self.sensor_frames.append(sensor_data["frame_idx"])
        return self._result()

    def step_hold(self):
        """Execute one hold frame before alignment is available.

        Args:
            None.

        Returns:
            Runtime-like result for the frame.
        """
        self.hold_count += 1
        return self._result()


class _FakeHumanTrajectory:
    """Three-frame source used to verify runner iteration semantics."""

    n_frames = 3
    source = Path("human.npz")

    def get_frame(self, frame_idx):
        """Return one identifiable raw source frame.

        Args:
            frame_idx: Requested source index.

        Returns:
            Minimal frame mapping used by the runtime fake.
        """
        return {"frame_idx": frame_idx}


def test_offline_runner_executes_first_aligned_frame_without_skipping(monkeypatch):
    """Verify every selected human frame immediately produces one runtime step.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    monkeypatch.setattr(mujoco_offline_simulation, "build_mujoco_runtime", lambda config: runtime)
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_human_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(mujoco_offline_simulation, "initialize_avp_alignment", lambda runtime, frame: True)

    summary = mujoco_offline_simulation.run_mujoco_offline_simulation_from_config(
        {"data": "human.npz", "start": 0, "end": -1, "log_every_frames": 20}
    )

    assert runtime.sensor_frames == [0, 1, 2]
    assert runtime.hold_count == 0
    assert runtime.simulation_time == pytest.approx(0.15)
    assert summary["source_frames_processed"] == 3.0
    assert summary["source_frame_start"] == 0.0
    assert summary["source_frame_end"] == 2.0


def test_offline_runner_preserves_invalid_prealignment_frame_time(monkeypatch):
    """Verify a frame without alignment still consumes one 20 Hz simulation tick.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    alignment_results = iter([False, True])
    monkeypatch.setattr(mujoco_offline_simulation, "build_mujoco_runtime", lambda config: runtime)
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_human_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "initialize_avp_alignment",
        lambda runtime, frame: next(alignment_results),
    )

    mujoco_offline_simulation.run_mujoco_offline_simulation_from_config(
        {"data": "human.npz", "start": 0, "end": 1, "log_every_frames": 20}
    )

    assert runtime.hold_count == 1
    assert runtime.sensor_frames == [1]
    assert runtime.simulation_time == pytest.approx(0.1)


def test_offline_runner_rejects_source_rate_without_resampling():
    """Verify non-20-Hz human trajectories require an explicit future resampler.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.apps.mujoco_offline_simulation import run_mujoco_offline_simulation_from_config

    with pytest.raises(ValueError, match="source_hz must match"):
        run_mujoco_offline_simulation_from_config(
            {"data": "human.npz", "source_hz": 30.0}
        )


def test_offline_raw_avp_frames_retarget_directly_into_headless_mujoco():
    """Run the real two-frame raw-human path without using stored robot qpos.

    Args:
        None.

    Returns:
        None.
    """
    pytest.importorskip("mujoco")
    from retargeting.apps.mujoco_offline_simulation import run_mujoco_offline_simulation_from_config
    from retargeting.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=mujoco_offline_simulation",
            "data=tests/fixtures/avp_short_replay.npz",
            "end=1",
            "log_every_frames=2",
        ]
    )

    summary = run_mujoco_offline_simulation_from_config(config)

    assert summary["source_frames_processed"] == 2.0
    assert summary["simulation_time"] == pytest.approx(0.1)
    assert np.isfinite(summary["tracking_error_max"])
