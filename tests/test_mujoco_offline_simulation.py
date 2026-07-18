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


def test_offline_avp_loader_never_accesses_retarget_qpos(monkeypatch):
    """Verify the dedicated loader materializes only raw AVP streams.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from teleoperation.inputs import offline_avp

    archive = _TrackingNpz()
    monkeypatch.setattr(offline_avp.np, "load", lambda path: archive)

    trajectory = offline_avp.load_offline_avp_trajectory("avp.npz")

    assert archive.accessed == ["stream_right_wrist", "stream_right_fingers"]
    assert isinstance(trajectory, offline_avp.OfflineAvpTrajectory)
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
        self.reset_count = 0
        self.session = object()
        self.backend = SimpleNamespace(
            model=object(),
            data=object(),
            get_joint_pos=lambda: np.array([0.25, -0.5]),
        )
        self.post_command_step = None

    def set_post_command_step(self, callback) -> None:
        """Set the observer called after each fake command period.

        Args:
            callback: Zero-argument observer, or None to disable observation.

        Returns:
            None.
        """
        self.post_command_step = callback

    def reset(self) -> None:
        """Reset per-cycle simulation state while retaining call history.

        Args:
            None.

        Returns:
            None.
        """
        self.reset_count += 1
        self.frame_count = 0
        self.simulation_time = 0.0

    def _result(self):
        """Advance one command period and build a runtime-like result.

        Args:
            None.

        Returns:
            Namespace containing simulation diagnostics.
        """
        self.frame_count += 1
        self.simulation_time += 0.05
        if self.post_command_step is not None:
            self.post_command_step()
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
    from retargeting_apps.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    monkeypatch.setattr(
        mujoco_offline_simulation.mujoco_runtime_builder,
        "build_mujoco_runtime",
        lambda config: runtime,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "create_mujoco_web_visualizer",
        lambda model, data, config: pytest.fail("Disabled viewer must not be created."),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_avp_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "initialize_avp_alignment",
        lambda session, frame, robot_qpos: True,
    )

    summary = mujoco_offline_simulation.run(
        {"data": "human.npz", "start": 0, "end": -1, "log_every_frames": 20},
        [],
    )

    assert runtime.sensor_frames == [0, 1, 2]
    assert runtime.hold_count == 0
    assert runtime.simulation_time == pytest.approx(0.15)
    assert summary["source_frames_processed"] == 3.0
    assert summary["source_frame_start"] == 0.0
    assert summary["source_frame_end"] == 2.0


class _FakeMujocoWebVisualizer:
    """Record passive viewer lifecycle calls without starting a Web server."""

    def __init__(self) -> None:
        """Initialize lifecycle counters.

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
        """Record one scene publication.

        Args:
            None.

        Returns:
            None.
        """
        self.update_count += 1

    def wait_for_client(self) -> None:
        """Record the pre-simulation client wait.

        Args:
            None.

        Returns:
            None.
        """
        self.wait_for_client_count += 1

    def wait_after_completion(self) -> None:
        """Record the optional final-state wait.

        Args:
            None.

        Returns:
            None.
        """
        self.wait_after_completion_count += 1

    def close(self) -> None:
        """Record viewer cleanup.

        Args:
            None.

        Returns:
            None.
        """
        self.close_count += 1


def test_offline_runner_publishes_initial_and_completed_mujoco_states(monkeypatch):
    """Verify the passive viewer observes every state without owning stepping.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting_apps.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    visualizer = _FakeMujocoWebVisualizer()
    factory_args = []

    def create_visualizer(model, data, config):
        """Capture the backend state objects supplied to the viewer factory.

        Args:
            model: Backend MuJoCo model marker.
            data: Backend MuJoCo data marker.
            config: Parsed viewer configuration.

        Returns:
            Fake visualizer used by this test.
        """
        factory_args.append((model, data, config))
        return visualizer

    monkeypatch.setattr(
        mujoco_offline_simulation.mujoco_runtime_builder,
        "build_mujoco_runtime",
        lambda config: runtime,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_avp_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "initialize_avp_alignment",
        lambda session, frame, robot_qpos: True,
    )
    monkeypatch.setattr(mujoco_offline_simulation, "create_mujoco_web_visualizer", create_visualizer)

    summary = mujoco_offline_simulation.run(
        {
            "data": "human.npz",
            "start": 0,
            "end": -1,
            "log_every_frames": 20,
            "viewer": {"enabled": True},
        },
        [],
    )

    assert len(factory_args) == 1
    assert factory_args[0][0] is runtime.backend.model
    assert factory_args[0][1] is runtime.backend.data
    assert factory_args[0][2].enabled is True
    assert visualizer.update_count == 4
    assert visualizer.wait_for_client_count == 1
    assert visualizer.wait_after_completion_count == 1
    assert visualizer.close_count == 1
    assert runtime.sensor_frames == [0, 1, 2]
    assert summary["simulation_time"] == pytest.approx(0.15)


def test_offline_runner_loops_with_full_reset_and_fresh_first_frame_alignment(monkeypatch):
    """Verify a repeated interval resets before realigning and executing frame zero.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting_apps.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    visualizer = _FakeMujocoWebVisualizer()
    alignment_frames = []
    original_step_sensor_data = runtime.step_sensor_data

    def initialize_alignment(session, frame, robot_qpos):
        """Record the first aligned frame of each playback cycle.

        Args:
            session: Session supplied by the shared driver.
            frame: Raw source frame used as the new alignment origin.
            robot_qpos: Current backend positions supplied by the driver.

        Returns:
            True to complete alignment immediately.
        """
        assert session is runtime.session
        np.testing.assert_allclose(robot_qpos, [0.25, -0.5])
        alignment_frames.append(frame["frame_idx"])
        return True

    def interrupt_during_second_cycle(sensor_data):
        """Stop after the first frame of the second cycle has executed.

        Args:
            sensor_data: Raw source frame selected by the runner.

        Returns:
            Runtime-like result for completed frames.
        """
        if runtime.reset_count == 1 and sensor_data["frame_idx"] == 1:
            raise KeyboardInterrupt
        return original_step_sensor_data(sensor_data)

    runtime.step_sensor_data = interrupt_during_second_cycle
    monkeypatch.setattr(
        mujoco_offline_simulation.mujoco_runtime_builder,
        "build_mujoco_runtime",
        lambda config: runtime,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_avp_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(mujoco_offline_simulation, "initialize_avp_alignment", initialize_alignment)
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "create_mujoco_web_visualizer",
        lambda model, data, config: visualizer,
    )

    summary = mujoco_offline_simulation.run(
        {
            "data": "human.npz",
            "loop": True,
            "log_every_frames": 20,
            "viewer": {"enabled": True},
        },
        [],
    )

    assert runtime.sensor_frames == [0, 1, 2, 0]
    assert runtime.reset_count == 1
    assert alignment_frames == [0, 0]
    assert runtime.simulation_time == pytest.approx(0.05)
    assert summary["source_frames_processed"] == 4.0
    assert visualizer.update_count == 6
    assert visualizer.wait_after_completion_count == 0
    assert visualizer.close_count == 1


def test_offline_runner_closes_mujoco_viewer_when_frame_processing_fails(monkeypatch):
    """Verify viewer cleanup is guaranteed when retargeting raises.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting_apps.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    visualizer = _FakeMujocoWebVisualizer()

    def fail_step(sensor_data):
        """Raise the frame-processing failure exercised by this test.

        Args:
            sensor_data: Raw source frame ignored before failure.

        Returns:
            None because this function always raises.
        """
        del sensor_data
        raise RuntimeError("frame failure")

    runtime.step_sensor_data = fail_step
    monkeypatch.setattr(
        mujoco_offline_simulation.mujoco_runtime_builder,
        "build_mujoco_runtime",
        lambda config: runtime,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_avp_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "initialize_avp_alignment",
        lambda session, frame, robot_qpos: True,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "create_mujoco_web_visualizer",
        lambda model, data, config: visualizer,
    )

    with pytest.raises(RuntimeError, match="frame failure"):
        mujoco_offline_simulation.run(
            {"data": "human.npz", "viewer": {"enabled": True}},
            [],
        )

    assert visualizer.update_count == 1
    assert visualizer.wait_for_client_count == 1
    assert visualizer.wait_after_completion_count == 0
    assert visualizer.close_count == 1


def test_offline_runner_preserves_invalid_prealignment_frame_time(monkeypatch):
    """Verify a frame without alignment still consumes one 20 Hz simulation tick.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting_apps.apps import mujoco_offline_simulation

    runtime = _FakeOfflineRuntime()
    alignment_results = iter([False, True])
    monkeypatch.setattr(
        mujoco_offline_simulation.mujoco_runtime_builder,
        "build_mujoco_runtime",
        lambda config: runtime,
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "load_offline_avp_trajectory",
        lambda path: _FakeHumanTrajectory(),
    )
    monkeypatch.setattr(
        mujoco_offline_simulation,
        "initialize_avp_alignment",
        lambda session, frame, robot_qpos: next(alignment_results),
    )

    mujoco_offline_simulation.run(
        {"data": "human.npz", "start": 0, "end": 1, "log_every_frames": 20},
        [],
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
    from retargeting_apps.apps.mujoco_offline_simulation import run

    with pytest.raises(ValueError, match="source_hz must match"):
        run(
            {"data": "human.npz", "source_hz": 30.0},
            [],
        )


def test_offline_runner_rejects_non_boolean_loop_option():
    """Require an explicit boolean for continuous playback selection.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting_apps.apps.mujoco_offline_simulation import run

    with pytest.raises(ValueError, match="loop must be a boolean"):
        run({"data": "human.npz", "loop": "true"}, [])


def test_offline_raw_avp_frames_retarget_directly_into_headless_mujoco():
    """Run the real two-frame raw-human path without using stored robot qpos.

    Args:
        None.

    Returns:
        None.
    """
    pytest.importorskip("mujoco")
    from retargeting_apps.apps.mujoco_offline_simulation import run
    from retargeting_apps.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=mujoco_offline_simulation",
            "data=tests/fixtures/avp_short_replay.npz",
            "end=1",
            "log_every_frames=2",
            "simulator.startup_move_frames=0",
        ]
    )

    summary = run(config, [])

    assert summary["source_frames_processed"] == 2.0
    assert summary["simulation_time"] == pytest.approx(0.1)
    assert np.isfinite(summary["tracking_error_max"])


def test_offline_first_retargeted_frame_completes_startup_move_in_real_mujoco():
    """Verify the real MJCF executes all startup waypoints before frame advance.

    Args:
        None.

    Returns:
        None.
    """
    pytest.importorskip("mujoco")
    from retargeting_apps.apps.mujoco_offline_simulation import run
    from retargeting_apps.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=mujoco_offline_simulation",
            "data=tests/fixtures/avp_short_replay.npz",
            "end=0",
            "log_every_frames=1",
            "simulator.startup_move_frames=1",
        ]
    )

    summary = run(config, [])

    assert summary["source_frames_processed"] == 1.0
    assert summary["startup_move_active"] == 1.0
    assert summary["command_periods_advanced"] > 1.0
    assert summary["simulation_time"] == pytest.approx(0.05 * summary["command_periods_advanced"])
