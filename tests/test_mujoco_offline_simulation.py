from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleoperation.backends.base import BackendStepResult
from teleoperation.types import ExecutionStatus, ExecutionStepResult, FlowSummary


class _TrackingNpz:
    """NPZ-like archive that records arrays materialized by offline input."""

    files = ["stream_right_wrist", "stream_right_fingers", "retarget_qpos"]

    def __init__(self) -> None:
        """Create two raw frames plus a forbidden robot-qpos array.

        Args:
            None.

        Returns:
            None.
        """
        self.accessed: list[str] = []
        self.arrays = {
            "stream_right_wrist": np.repeat(np.eye(4)[None, None, :, :], 2, axis=0),
            "stream_right_fingers": np.repeat(np.eye(4)[None, None, :, :], 50, axis=0).reshape(2, 25, 4, 4),
            "retarget_qpos": np.zeros((2, 23)),
        }

    def __enter__(self):
        """Return this archive from its context manager.

        Args:
            None.

        Returns:
            Tracking archive.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Leave the archive context without suppressing exceptions.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception instance.
            traceback: Optional exception traceback.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback

    def __getitem__(self, key: str):
        """Record and return one requested archive array.

        Args:
            key: NPZ array name.

        Returns:
            Requested fake array.
        """
        self.accessed.append(key)
        return self.arrays[key]


def test_offline_avp_input_never_accesses_retarget_qpos(monkeypatch):
    """Archived input materializes only raw AVP stream arrays."""
    from teleoperation.inputs.avp import offline

    archive = _TrackingNpz()
    monkeypatch.setattr(offline.np, "load", lambda path: archive)

    hand_input = offline.AvpOfflineInput("avp.npz")
    hand_input.open()
    first = hand_input.read()

    assert archive.accessed == ["stream_right_wrist", "stream_right_fingers"]
    assert hand_input.n_frames == 2
    assert first.source_index == 0
    assert first.keypoints_wrist.shape == (21, 3)


def test_offline_input_distinguishes_missing_detection_from_end_of_stream(monkeypatch):
    """Missing hand data is a sample while finite input end is None."""
    from teleoperation.inputs.avp import decode_avp_sample, offline

    archive = _TrackingNpz()
    monkeypatch.setattr(offline.np, "load", lambda path: archive)
    hand_input = offline.AvpOfflineInput("avp.npz", start=1, end=1)
    hand_input.open()

    valid = hand_input.read()
    end = hand_input.read()
    missing = decode_avp_sample(None, source_index=7)

    assert valid is not None and valid.has_hand
    assert end is None
    assert missing is not None and not missing.has_hand
    assert missing.source_index == 7


def test_online_and_offline_avp_inputs_share_equivalent_decoding(tmp_path, monkeypatch):
    """Live and archived acquisition produce identical normalized AVP samples."""
    from teleoperation.inputs.avp import AvpOfflineInput, AvpOnlineInput

    wrist = np.eye(4)[None, :, :]
    fingers = np.repeat(np.eye(4)[None, :, :], 25, axis=0)
    archive_path = tmp_path / "avp.npz"
    np.savez(
        archive_path,
        stream_right_wrist=wrist[None, ...],
        stream_right_fingers=fingers[None, ...],
    )

    class FakeStreamer:
        """Expose one raw frame through the live streamer contract."""

        def __init__(self, ip, record):
            """Capture connection options and publish the test frame.

            Args:
                ip: Configured live address.
                record: Configured recording flag.

            Returns:
                None.
            """
            assert ip == "127.0.0.1"
            assert record is True
            self.latest = {"right_wrist": wrist, "right_fingers": fingers}

    fake_module = types.ModuleType("avp_stream")
    fake_module.VisionProStreamer = FakeStreamer
    monkeypatch.setitem(sys.modules, "avp_stream", fake_module)
    online = AvpOnlineInput("127.0.0.1")
    offline = AvpOfflineInput(archive_path)
    online.open()
    offline.open()

    online_sample = online.read()
    offline_sample = offline.read()

    np.testing.assert_allclose(online_sample.keypoints_wrist, offline_sample.keypoints_wrist)
    np.testing.assert_allclose(online_sample.wrist_pose_sensor, offline_sample.wrist_pose_sensor)


class _FakeVisualizer:
    """Record passive viewer lifecycle calls without opening a server."""

    def __init__(self) -> None:
        """Initialize viewer lifecycle counters.

        Args:
            None.

        Returns:
            None.
        """
        self.update_count = 0
        self.wait_for_client_count = 0
        self.wait_after_completion_count = 0
        self.close_count = 0

    def update(self) -> None:
        """Record one passive state publication."""
        self.update_count += 1

    def wait_for_client(self) -> None:
        """Record a pre-run client wait."""
        self.wait_for_client_count += 1

    def wait_after_completion(self) -> None:
        """Record a final-state wait."""
        self.wait_after_completion_count += 1

    def close(self) -> None:
        """Record guaranteed viewer cleanup."""
        self.close_count += 1


class _FakeAutoVisualizerFactory:
    """Record app-level execution viewer creation without opening a server."""

    def __init__(self, visualizer: _FakeVisualizer) -> None:
        """Store the visualizer returned to the app under test.

        Args:
            visualizer: Fake viewer returned for every enabled request.

        Returns:
            None.
        """
        self.visualizer = visualizer
        self.calls = []

    def __call__(self, config_data, flow) -> _FakeVisualizer:
        """Register the factory inputs and attach passive observer updates.

        Args:
            config_data: Plain composed app config.
            flow: Fake execution flow receiving observer callbacks.

        Returns:
            Fake visualizer with lifecycle counters.
        """
        self.calls.append((config_data, flow))
        self.visualizer.update()
        flow.add_command_observer(lambda result: self.visualizer.update())
        flow.add_reset_observer(lambda qpos: self.visualizer.update())
        self.visualizer.wait_for_client()
        return self.visualizer


class _FakeAppFlow:
    """Complete-flow fake used to test thin application lifecycle behavior."""

    def __init__(self, *, loop: bool = False, fail: bool = False) -> None:
        """Configure finite run behavior and observer storage.

        Args:
            loop: Whether to simulate one cycle reset before interruption.
            fail: Whether ``run`` raises a frame-processing failure.

        Returns:
            None.
        """
        from teleoperation.inputs.avp import AvpOfflineInput

        self.input = AvpOfflineInput.__new__(AvpOfflineInput)
        self.input.frame_indices = (0, 1, 2)
        self.backend = SimpleNamespace(model=object(), data=object())
        self.loop = loop
        self.fail = fail
        self.source_frame_count = 0
        self.command_observers = []
        self.step_observers = []
        self.reset_observers = []

    def add_command_observer(self, observer) -> None:
        """Register one command-period observer.

        Args:
            observer: Passive backend result callback.

        Returns:
            None.
        """
        self.command_observers.append(observer)

    def add_step_observer(self, observer) -> None:
        """Register one source-frame observer.

        Args:
            observer: Passive execution result callback.

        Returns:
            None.
        """
        self.step_observers.append(observer)

    def add_reset_observer(self, observer) -> None:
        """Register one reset-state observer.

        Args:
            observer: Passive reset callback.

        Returns:
            None.
        """
        self.reset_observers.append(observer)

    def run(self) -> FlowSummary:
        """Publish deterministic command/step results or raise a failure.

        Args:
            None.

        Returns:
            Aggregate flow summary.
        """
        if self.fail:
            raise RuntimeError("frame failure")
        total_frames = 4 if self.loop else 3
        last_result = None
        for index in range(total_frames):
            if self.loop and index == 3:
                for observer in self.reset_observers:
                    observer(np.zeros(2))
                self.source_frame_count = 0
            backend_result = BackendStepResult(
                command_qpos=np.zeros(2),
                actual_qpos=np.zeros(2),
                diagnostics={"simulation_time": 0.05 * (self.source_frame_count + 1)},
            )
            for observer in self.command_observers:
                observer(backend_result)
            self.source_frame_count += 1
            source_index = index if index < 3 else 0
            last_result = ExecutionStepResult(
                status=ExecutionStatus.EXECUTED,
                retargeted_frame=None,
                command_qpos=np.zeros(2),
                actual_qpos=np.zeros(2),
                diagnostics={"simulation_time": 0.05 * self.source_frame_count, "tracking_error_max": 0.0},
                source_index=source_index,
            )
            for observer in self.step_observers:
                observer(last_result)
        return FlowSummary(total_frames, total_frames, total_frames, 2 if self.loop else 1, last_result)


def test_offline_app_uses_flow_run_and_passive_viewer_observers(monkeypatch):
    """The app owns viewer lifecycle but does not own frame execution."""
    from retargeting_apps.apps import teleop_exe

    flow = _FakeAppFlow()
    visualizer = _FakeVisualizer()
    factory = _FakeAutoVisualizerFactory(visualizer)
    monkeypatch.setattr(teleop_exe, "build_execution_flow", lambda config: flow)
    monkeypatch.setattr(teleop_exe, "create_optional_execution_visualizer", factory)

    summary = teleop_exe.run(
        {"data": "human.npz", "viewer": {"enabled": True}, "log_every_frames": 20},
        [],
    )

    assert factory.calls == [({"data": "human.npz", "viewer": {"enabled": True}, "log_every_frames": 20}, flow)]
    assert visualizer.update_count == 4
    assert visualizer.wait_for_client_count == 1
    assert visualizer.wait_after_completion_count == 1
    assert visualizer.close_count == 1
    assert summary["source_frames_processed"] == 3.0
    assert summary["source_frame_start"] == 0.0
    assert summary["source_frame_end"] == 2.0


def test_offline_app_observes_reset_between_loop_cycles(monkeypatch):
    """Loop reset state is published by a passive reset observer."""
    from retargeting_apps.apps import teleop_exe

    flow = _FakeAppFlow(loop=True)
    visualizer = _FakeVisualizer()
    factory = _FakeAutoVisualizerFactory(visualizer)
    monkeypatch.setattr(teleop_exe, "build_execution_flow", lambda config: flow)
    monkeypatch.setattr(teleop_exe, "create_optional_execution_visualizer", factory)

    summary = teleop_exe.run(
        {"data": "human.npz", "loop": True, "viewer": {"enabled": True}},
        [],
    )

    assert visualizer.update_count == 6
    assert visualizer.wait_after_completion_count == 0
    assert visualizer.close_count == 1
    assert summary["source_frames_processed"] == 4.0


def test_offline_app_closes_viewer_when_flow_fails(monkeypatch):
    """Viewer cleanup remains guaranteed when flow execution raises."""
    from retargeting_apps.apps import teleop_exe

    flow = _FakeAppFlow(fail=True)
    visualizer = _FakeVisualizer()
    factory = _FakeAutoVisualizerFactory(visualizer)
    monkeypatch.setattr(teleop_exe, "build_execution_flow", lambda config: flow)
    monkeypatch.setattr(teleop_exe, "create_optional_execution_visualizer", factory)

    with pytest.raises(RuntimeError, match="frame failure"):
        teleop_exe.run({"data": "human.npz", "viewer": {"enabled": True}}, [])

    assert visualizer.update_count == 1
    assert visualizer.close_count == 1


def test_offline_runner_rejects_source_rate_without_resampling():
    """Non-20-Hz archived input requires a future explicit resampler."""
    from retargeting_apps.apps.teleop_exe import run

    with pytest.raises(ValueError, match="source_hz must match"):
        run({"data": "human.npz", "source_hz": 30.0}, [])


def test_offline_runner_rejects_non_boolean_loop_option():
    """Continuous playback selection requires an explicit boolean."""
    from retargeting_apps.apps.teleop_exe import run

    with pytest.raises(ValueError, match="loop must be a boolean"):
        run({"data": "human.npz", "loop": "true"}, [])


def test_teleop_exe_runs_offline_input_with_kinematic_backend():
    """The unified execution app can run archived input without MuJoCo."""
    from retargeting_apps.apps.teleop_exe import run
    from retargeting_apps.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=teleop_exe",
            "input.data=tests/fixtures/avp_short_replay.npz",
            "input.end=1",
            "teleoperation_mode.pipeline.realtime=false",
            "log_every_frames=2",
        ]
    )

    summary = run(config, [])

    assert summary["source_frames_processed"] == 2.0
    assert summary["command_periods_advanced_total"] == 2.0
    assert summary["source_frame_start"] == 0.0
    assert summary["source_frame_end"] == 1.0


def test_offline_raw_avp_frames_retarget_directly_into_headless_mujoco():
    """Real raw AVP frames execute without using archived robot qpos."""
    pytest.importorskip("mujoco")
    from retargeting_apps.apps.teleop_exe import run
    from retargeting_apps.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=teleop_exe",
            "teleoperation_modes=offline_mujoco",
            "input.data=tests/fixtures/avp_short_replay.npz",
            "input.end=1",
            "log_every_frames=2",
            "teleoperation_mode.pipeline.startup_move_frames=0",
        ]
    )

    summary = run(config, [])

    assert summary["source_frames_processed"] == 2.0
    assert summary["simulation_time"] == pytest.approx(0.1)
    assert np.isfinite(summary["tracking_error_max"])
