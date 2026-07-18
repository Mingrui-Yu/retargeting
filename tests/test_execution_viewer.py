from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeViserVisualizer:
    """Passive Viser fake used by factory dispatch tests."""

    def __init__(self) -> None:
        """Initialize lifecycle and qpos records.

        Args:
            None.

        Returns:
            None.
        """
        self.qpos_updates = []
        self.observation_updates = []
        self.hide_observation_count = 0
        self.wait_for_client_count = 0
        self.wait_after_completion_count = 0
        self.close_count = 0

    def update_qpos(self, qpos) -> None:
        """Record one qpos update.

        Args:
            qpos: Backend qpos vector.

        Returns:
            None.
        """
        self.qpos_updates.append(np.asarray(qpos, dtype=float).copy())

    def update_observation(self, observation) -> None:
        """Record one canonical hand observation.

        Args:
            observation: Canonical hand observation.

        Returns:
            None.
        """
        self.observation_updates.append(observation)

    def hide_observation(self) -> None:
        """Record one request to hide human-hand geometry.

        Args:
            None.

        Returns:
            None.
        """
        self.hide_observation_count += 1

    def wait_for_client(self) -> None:
        """Record one startup wait.

        Args:
            None.

        Returns:
            None.
        """
        self.wait_for_client_count += 1

    def wait_after_completion(self) -> None:
        """Record one completion wait.

        Args:
            None.

        Returns:
            None.
        """
        self.wait_after_completion_count += 1

    def close(self) -> None:
        """Record one close call.

        Args:
            None.

        Returns:
            None.
        """
        self.close_count += 1


class _FakeMjviserVisualizer:
    """Passive mjviser fake used by factory dispatch tests."""

    def __init__(self) -> None:
        """Initialize lifecycle and scene update records.

        Args:
            None.

        Returns:
            None.
        """
        self.update_count = 0
        self.observation_updates = []
        self.hide_observation_count = 0
        self.wait_for_client_count = 0

    def update(self) -> None:
        """Record one scene update.

        Args:
            None.

        Returns:
            None.
        """
        self.update_count += 1

    def update_observation(self, observation) -> None:
        """Record one canonical hand observation.

        Args:
            observation: Canonical hand observation.

        Returns:
            None.
        """
        self.observation_updates.append(observation)

    def hide_observation(self) -> None:
        """Record one request to hide human-hand geometry.

        Args:
            None.

        Returns:
            None.
        """
        self.hide_observation_count += 1

    def wait_for_client(self) -> None:
        """Record one startup wait.

        Args:
            None.

        Returns:
            None.
        """
        self.wait_for_client_count += 1

    def wait_after_completion(self) -> None:
        """No-op completion wait for protocol compatibility.

        Args:
            None.

        Returns:
            None.
        """

    def close(self) -> None:
        """No-op close for protocol compatibility.

        Args:
            None.

        Returns:
            None.
        """


class _FakeFlow:
    """Minimal execution flow fake exposing backend and observer registration."""

    def __init__(self, backend) -> None:
        """Store backend and observer lists.

        Args:
            backend: Fake backend exposed to the viewer factory.

        Returns:
            None.
        """
        self.backend = backend
        self.command_observers = []
        self.step_observers = []
        self.reset_observers = []

    def add_command_observer(self, observer) -> None:
        """Record one command observer.

        Args:
            observer: Command-period callback.

        Returns:
            None.
        """
        self.command_observers.append(observer)

    def add_reset_observer(self, observer) -> None:
        """Record one reset observer.

        Args:
            observer: Reset callback.

        Returns:
            None.
        """
        self.reset_observers.append(observer)

    def add_step_observer(self, observer) -> None:
        """Record one source-frame observer.

        Args:
            observer: Source-frame callback.

        Returns:
            None.
        """
        self.step_observers.append(observer)


def test_auto_execution_viewer_selects_viser_for_kinematic_backend(monkeypatch):
    """Verify automatic dispatch binds kinematic execution to plain Viser."""
    from retargeting_apps.visualization.execution import manager

    assert manager.DEFAULT_VIEWER_TYPE_BY_BACKEND == {
        "mujoco": "mjviser",
        "kinematic": "viser",
    }

    visualizer = _FakeViserVisualizer()
    calls = []

    def create_viser_live_visualizer(**kwargs):
        """Record Viser factory inputs and return a fake visualizer.

        Args:
            **kwargs: Keyword arguments forwarded by the dispatch factory.

        Returns:
            Fake Viser visualizer.
        """
        calls.append(kwargs)
        return visualizer

    flow = _FakeFlow(SimpleNamespace(get_joint_pos=lambda: np.array([0.1, 0.2])))
    monkeypatch.setattr(manager, "create_viser_live_visualizer", create_viser_live_visualizer)

    attached = manager.create_optional_execution_visualizer(
        {
            "profile": "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml",
            "backend": {"name": "kinematic", "command_hz": 20.0},
            "viewer": {"enabled": True, "type": "auto", "wait_for_client": True},
        },
        flow,
    )

    assert attached is visualizer
    assert calls[0]["robot_file_path"].endswith("panda_leap_paxini/urdf/panda_leap_paxini.urdf")
    assert calls[0]["actuated_joint_names"][0] == "panda_joint1"
    assert visualizer.wait_for_client_count == 1
    np.testing.assert_allclose(visualizer.qpos_updates[0], [0.1, 0.2])
    flow.command_observers[0](SimpleNamespace(actual_qpos=np.array([0.3, 0.4])))
    for observer in flow.reset_observers:
        observer(np.array([0.5, 0.6]))
    np.testing.assert_allclose(visualizer.qpos_updates[1], [0.3, 0.4])
    np.testing.assert_allclose(visualizer.qpos_updates[2], [0.5, 0.6])
    observation = object()
    flow.step_observers[0](SimpleNamespace(retargeted_frame=SimpleNamespace(observation=observation)))
    flow.step_observers[0](SimpleNamespace(retargeted_frame=None))
    assert visualizer.observation_updates == [observation]
    assert visualizer.hide_observation_count == 2


def test_auto_execution_viewer_selects_mjviser_for_mujoco_backend(monkeypatch):
    """Verify automatic dispatch binds MuJoCo execution to mjviser."""
    from retargeting_apps.visualization.execution import manager

    visualizer = _FakeMjviserVisualizer()
    model = object()
    data = object()
    calls = []

    def create_mujoco_web_visualizer(created_model, created_data, config):
        """Record mjviser factory inputs and return a fake visualizer.

        Args:
            created_model: Backend MuJoCo model.
            created_data: Backend MuJoCo data.
            config: Viewer configuration.

        Returns:
            Fake mjviser visualizer.
        """
        calls.append((created_model, created_data, config))
        return visualizer

    flow = _FakeFlow(SimpleNamespace(model=model, data=data))
    monkeypatch.setattr(manager, "create_mujoco_web_visualizer", create_mujoco_web_visualizer)

    attached = manager.create_optional_execution_visualizer(
        {
            "backend": {"name": "mujoco", "command_hz": 20.0},
            "viewer": {"enabled": True, "type": "auto", "wait_for_client": True},
        },
        flow,
    )

    assert attached is visualizer
    assert calls[0][0] is model
    assert calls[0][1] is data
    assert visualizer.update_count == 1
    assert visualizer.wait_for_client_count == 1
    flow.command_observers[0](object())
    for observer in flow.reset_observers:
        observer(np.zeros(2))
    assert visualizer.update_count == 3
    observation = object()
    flow.step_observers[0](SimpleNamespace(retargeted_frame=SimpleNamespace(observation=observation)))
    flow.step_observers[0](SimpleNamespace(retargeted_frame=None))
    assert visualizer.observation_updates == [observation]
    assert visualizer.hide_observation_count == 2


def test_execution_viewer_disabled_avoids_visualizer_construction(monkeypatch):
    """Verify disabled viewer config does not touch concrete visualizer factories."""
    from retargeting_apps.visualization.execution import manager

    monkeypatch.setattr(
        manager,
        "create_viser_live_visualizer",
        lambda **kwargs: pytest.fail("viser factory should not run"),
    )
    monkeypatch.setattr(
        manager,
        "create_mujoco_web_visualizer",
        lambda *args: pytest.fail("mjviser factory should not run"),
    )

    assert (
        manager.create_optional_execution_visualizer(
            {"backend": {"name": "kinematic", "command_hz": 20.0}, "viewer": {"enabled": False}},
            _FakeFlow(SimpleNamespace()),
        )
        is None
    )


class _FakeInitialCamera:
    """Mutable camera pose container used by Viser server fakes."""

    position = None
    look_at = None


class _FakeViserServer:
    """Minimal Viser server fake for live URDF visualizer tests."""

    instances = []

    def __init__(self, host: str = "0.0.0.0", port: int = 0) -> None:
        """Record server construction without binding a socket.

        Args:
            host: Configured server host.
            port: Configured server port.

        Returns:
            None.
        """
        self.host = host
        self.port = port
        self.initial_camera = _FakeInitialCamera()
        self.scene = SimpleNamespace()
        self.clients = {}
        self.stop_count = 0
        self.instances.append(self)

    def get_port(self) -> int:
        """Return the configured or fake selected port.

        Args:
            None.

        Returns:
            Server port.
        """
        return self.port or 54321

    def get_clients(self) -> dict[int, object]:
        """Return fake connected clients.

        Args:
            None.

        Returns:
            Connected-client mapping.
        """
        return self.clients

    def stop(self) -> None:
        """Record one stop call.

        Args:
            None.

        Returns:
            None.
        """
        self.stop_count += 1


class _FakeViserUrdf:
    """Minimal ViserUrdf fake that exposes a different actuated-joint order."""

    instances = []

    def __init__(self, server, robot_file_path: Path, **options) -> None:
        """Record URDF construction.

        Args:
            server: Fake Viser server.
            robot_file_path: Robot URDF path.
            **options: URDF loading options.

        Returns:
            None.
        """
        self.server = server
        self.robot_file_path = robot_file_path
        self.options = options
        self.cfg_updates = []
        self.instances.append(self)

    def get_actuated_joint_names(self) -> tuple[str, str, str]:
        """Return URDF joint order.

        Args:
            None.

        Returns:
            Actuated joint names in ViserUrdf order.
        """
        return ("joint_b", "joint_a", "joint_c")

    def update_cfg(self, qpos) -> None:
        """Record one URDF-order qpos update.

        Args:
            qpos: Joint positions in URDF order.

        Returns:
            None.
        """
        self.cfg_updates.append(np.asarray(qpos, dtype=float).copy())


def test_viser_live_visualizer_maps_backend_qpos_to_urdf_order(monkeypatch):
    """Verify plain Viser live adapter reorders backend qpos for ViserUrdf."""
    from retargeting_apps.config import MujocoWebViewerConfig
    from retargeting_apps.visualization.execution import viser

    _FakeViserServer.instances.clear()
    _FakeViserUrdf.instances.clear()
    fake_viser_module = types.ModuleType("viser")
    fake_viser_module.ViserServer = _FakeViserServer
    fake_extras_module = types.ModuleType("viser.extras")
    fake_extras_module.ViserUrdf = _FakeViserUrdf
    monkeypatch.setitem(sys.modules, "viser", fake_viser_module)
    monkeypatch.setitem(sys.modules, "viser.extras", fake_extras_module)

    sleep_durations = []

    def sleep(duration: float) -> None:
        """Connect a fake client once, then interrupt final-state wait.

        Args:
            duration: Requested sleep duration.

        Returns:
            None.
        """
        sleep_durations.append(duration)
        server = _FakeViserServer.instances[0]
        if duration == 0.05:
            server.clients[1] = object()
        elif duration == 1.0:
            raise KeyboardInterrupt

    visualizer = viser.ViserLiveVisualizer(
        robot_file_path="robot.urdf",
        actuated_joint_names=("joint_a", "joint_b", "joint_c"),
        config=MujocoWebViewerConfig(
            enabled=True,
            type="viser",
            host="127.0.0.1",
            port=0,
            wait_for_client=True,
            keep_open_after_completion=True,
            initial_camera_position=(1.0, 2.0, 3.0),
            initial_camera_look_at=(0.1, 0.2, 0.3),
        ),
        sleep=sleep,
    )

    visualizer.update_qpos(np.array([10.0, 20.0, 30.0]))
    visualizer.wait_for_client()
    visualizer.wait_after_completion()
    visualizer.close()
    visualizer.close()

    server = _FakeViserServer.instances[0]
    urdf = _FakeViserUrdf.instances[0]
    assert server.host == "127.0.0.1"
    assert server.port == 0
    assert server.initial_camera.position == (1.0, 2.0, 3.0)
    assert server.initial_camera.look_at == (0.1, 0.2, 0.3)
    assert server.stop_count == 1
    assert urdf.robot_file_path == Path("robot.urdf")
    assert urdf.options["root_node_name"] == "/robot_mesh"
    np.testing.assert_allclose(urdf.cfg_updates[0], [20.0, 10.0, 30.0])
    assert sleep_durations == [0.05, 1.0]
