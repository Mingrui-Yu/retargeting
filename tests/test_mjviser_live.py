from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _FakeGuiNumberHandle:
    """Read-only number handle fake whose value remains server-updatable."""

    def __init__(self, label: str, initial_value: float, **options) -> None:
        """Record one number field and its presentation options.

        Args:
            label: Visible GUI field label.
            initial_value: Initial numeric value.
            **options: Additional number-field options.

        Returns:
            None.
        """
        self.label = label
        self.value = initial_value
        self.options = options


class _FakeGuiTab:
    """Context-manager fake for one Viser GUI tab."""

    def __enter__(self):
        """Enter the fake tab container.

        Args:
            None.

        Returns:
            This tab container.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Leave the fake tab without suppressing exceptions.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception value.
            traceback: Optional exception traceback.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback


class _FakeGuiTabGroup:
    """Record application tabs appended to mjviser's standard tabs."""

    def __init__(self) -> None:
        """Initialize the appended-tab record.

        Args:
            None.

        Returns:
            None.
        """
        self.tabs = []

    def add_tab(self, label: str, icon=None) -> _FakeGuiTab:
        """Record and return one fake tab context.

        Args:
            label: Visible tab label.
            icon: Optional tab icon.

        Returns:
            Fake tab context manager.
        """
        self.tabs.append((label, icon))
        return _FakeGuiTab()


class _FakeGui:
    """Minimal Viser GUI fake supporting read-only number fields."""

    def __init__(self) -> None:
        """Initialize the number-field record.

        Args:
            None.

        Returns:
            None.
        """
        self.numbers = []

    def add_number(self, label: str, initial_value: float, **options) -> _FakeGuiNumberHandle:
        """Create and record one fake number field.

        Args:
            label: Visible field label.
            initial_value: Initial numeric value.
            **options: Additional number-field options.

        Returns:
            Fake number handle with a mutable server-side value.
        """
        handle = _FakeGuiNumberHandle(label, initial_value, **options)
        self.numbers.append(handle)
        return handle


class _FakeAtomic:
    """Context-manager fake for one atomic Viser update batch."""

    def __init__(self, server) -> None:
        """Store the server whose atomic batches are counted.

        Args:
            server: Fake Viser server.

        Returns:
            None.
        """
        self.server = server

    def __enter__(self):
        """Count and enter one atomic batch.

        Args:
            None.

        Returns:
            This atomic context.
        """
        self.server.atomic_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Leave the atomic batch without suppressing exceptions.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception value.
            traceback: Optional exception traceback.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback


def _fake_mj_id2name(model, object_type, object_id: int) -> str:
    """Resolve a fake joint name by model index.

    Args:
        model: Fake MuJoCo model containing joint names.
        object_type: Fake object enum ignored by this resolver.
        object_id: Joint index whose name is requested.

    Returns:
        Joint name at the requested index.
    """
    del object_type
    return model.joint_names[object_id]


_FAKE_MUJOCO = SimpleNamespace(
    mjtJoint=SimpleNamespace(mjJNT_HINGE=3),
    mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
    mj_id2name=_fake_mj_id2name,
)


class _FakeMujocoModel:
    """Three-hinge model fake with deliberately non-sequential qpos addresses."""

    def __init__(self) -> None:
        """Create joint metadata used by readout mapping tests.

        Args:
            None.

        Returns:
            None.
        """
        self.njnt = 3
        self.jnt_type = np.array([3, 3, 3])
        self.jnt_qposadr = np.array([1, 0, 2])
        self.joint_names = ("joint_b", "joint_a", "joint_c")


class _FakeViserServer:
    """Minimal Viser server fake for passive adapter lifecycle tests."""

    instances = []

    def __init__(self, host: str, port: int) -> None:
        """Record server binding without opening a socket.

        Args:
            host: Configured bind host.
            port: Configured bind port.

        Returns:
            None.
        """
        self.host = host
        self.port = port
        self.clients = {}
        self.stop_count = 0
        self.atomic_count = 0
        self.gui = _FakeGui()
        self.instances.append(self)

    def get_port(self) -> int:
        """Return a deterministic port for configured ephemeral binding.

        Args:
            None.

        Returns:
            Configured port or a fake selected port.
        """
        return self.port or 43210

    def get_clients(self) -> dict[int, object]:
        """Return currently connected fake clients.

        Args:
            None.

        Returns:
            Mapping of fake client identifiers to markers.
        """
        return self.clients

    def stop(self) -> None:
        """Record one server stop request.

        Args:
            None.

        Returns:
            None.
        """
        self.stop_count += 1

    def atomic(self) -> _FakeAtomic:
        """Return a fake atomic update context.

        Args:
            None.

        Returns:
            Atomic context that records entry.
        """
        return _FakeAtomic(self)


class _FakeMujocoScene:
    """Minimal mjviser scene fake that records GUI and state updates."""

    instances = []

    def __init__(self, server, model, num_envs: int) -> None:
        """Capture the existing simulation model and server.

        Args:
            server: Fake Viser server.
            model: Existing backend model marker.
            num_envs: Number of visualized environments.

        Returns:
            None.
        """
        self.server = server
        self.model = model
        self.num_envs = num_envs
        self.gui_options = None
        self.tabs = _FakeGuiTabGroup()
        self.updated_data = []
        self.instances.append(self)

    def create_visualization_gui(self, **options) -> _FakeGuiTabGroup:
        """Record standard mjviser visualization GUI settings.

        Args:
            **options: Camera options forwarded by the adapter.

        Returns:
            Fake tab group available for application extensions.
        """
        self.gui_options = options
        return self.tabs

    def update_from_mjdata(self, data) -> None:
        """Record a passive update from the live backend data.

        Args:
            data: Existing backend data marker.

        Returns:
            None.
        """
        self.updated_data.append(data)


def test_mjviser_adapter_passively_updates_and_manages_lifecycle(monkeypatch):
    """Verify the adapter never steps physics and owns only viewer lifecycle.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    from retargeting_apps.config import MujocoWebViewerConfig
    from retargeting_apps.visualization import mjviser_live

    _FakeViserServer.instances.clear()
    _FakeMujocoScene.instances.clear()
    fake_viser = SimpleNamespace(
        ViserServer=_FakeViserServer,
        Icon=SimpleNamespace(ADJUSTMENTS="adjustments"),
    )
    monkeypatch.setattr(
        mjviser_live,
        "_load_mjviser_dependencies",
        lambda: (_FAKE_MUJOCO, fake_viser, _FakeMujocoScene),
    )

    sleep_durations = []

    def sleep(duration: float) -> None:
        """Connect a client once, then interrupt the final-state wait.

        Args:
            duration: Requested wait duration in seconds.

        Returns:
            None.
        """
        sleep_durations.append(duration)
        server = _FakeViserServer.instances[0]
        if duration == 0.05:
            server.clients[1] = object()
        elif duration == 1.0:
            raise KeyboardInterrupt

    model = _FakeMujocoModel()
    data = SimpleNamespace(qpos=np.array([0.1, 0.2, 0.3]))
    config = MujocoWebViewerConfig(
        enabled=True,
        host="127.0.0.1",
        port=0,
        wait_for_client=True,
        keep_open_after_completion=True,
        camera_distance=1.5,
        camera_azimuth=90.0,
        camera_elevation=30.0,
    )
    visualizer = mjviser_live.MujocoWebVisualizer(model, data, config, sleep=sleep)

    visualizer.update()
    visualizer.wait_for_client()
    visualizer.wait_after_completion()
    visualizer.close()
    visualizer.close()

    server = _FakeViserServer.instances[0]
    scene = _FakeMujocoScene.instances[0]
    assert server.host == "127.0.0.1"
    assert server.port == 0
    assert server.stop_count == 1
    assert server.atomic_count == 1
    assert scene.server is server
    assert scene.model is model
    assert scene.num_envs == 1
    assert scene.gui_options == {
        "camera_distance": 1.5,
        "camera_azimuth": 90.0,
        "camera_elevation": 30.0,
    }
    assert scene.updated_data == [data]
    assert scene.tabs.tabs == [("Joint angles", "adjustments")]
    assert [handle.label for handle in server.gui.numbers] == [
        "joint_b [rad]",
        "joint_a [rad]",
        "joint_c [rad]",
    ]
    assert [handle.value for handle in server.gui.numbers] == [0.2, 0.1, 0.3]
    assert all(handle.options["disabled"] is True for handle in server.gui.numbers)
    assert sleep_durations == [0.05, 1.0]


def test_joint_angle_readouts_cover_all_panda_leap_joints_and_qpos_addresses(monkeypatch):
    """Verify all 23 hinge joints follow compiled MuJoCo qpos addresses.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    import mujoco

    from retargeting.config import load_robot_config
    from retargeting_apps.config import MujocoWebViewerConfig
    from retargeting_apps.visualization import mjviser_live
    from teleoperation.config import load_mujoco_robot_binding_config

    robot_source = "configs/robots/panda_leap_paxini.yaml"
    robot_config = load_robot_config(robot_source)
    binding = load_mujoco_robot_binding_config(robot_source, robot_config=robot_config)
    model = mujoco.MjModel.from_xml_path(binding.simulation_file_path)
    data = mujoco.MjData(model)
    data.qpos[:] = np.linspace(-0.11, 0.11, model.nq)
    expected_qpos = data.qpos.copy()

    _FakeViserServer.instances.clear()
    _FakeMujocoScene.instances.clear()
    fake_viser = SimpleNamespace(
        ViserServer=_FakeViserServer,
        Icon=SimpleNamespace(ADJUSTMENTS="adjustments"),
    )
    monkeypatch.setattr(
        mjviser_live,
        "_load_mjviser_dependencies",
        lambda: (mujoco, fake_viser, _FakeMujocoScene),
    )

    visualizer = mjviser_live.MujocoWebVisualizer(
        model,
        data,
        MujocoWebViewerConfig(enabled=True, wait_for_client=False),
    )
    server = _FakeViserServer.instances[0]
    handles = {handle.label: handle for handle in server.gui.numbers}

    assert len(handles) == model.njnt == 23
    assert set(handles) == {
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)} [rad]"
        for joint_id in range(model.njnt)
    }
    assert handles["joint_1 [rad]"].value == round(float(data.qpos[7]), 5)
    assert handles["joint_0 [rad]"].value == round(float(data.qpos[8]), 5)

    data.qpos[:] = np.linspace(0.22, -0.22, model.nq)
    updated_qpos = data.qpos.copy()
    visualizer.update()

    np.testing.assert_allclose(data.qpos, updated_qpos)
    assert not np.array_equal(updated_qpos, expected_qpos)
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_address = int(model.jnt_qposadr[joint_id])
        assert handles[f"{name} [rad]"].value == round(float(updated_qpos[qpos_address]), 5)
    assert server.atomic_count == 1
    visualizer.close()


def test_mjviser_adapter_stops_server_when_scene_creation_fails(monkeypatch):
    """Verify partial viewer construction cannot leak its Web server.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        None.
    """
    import pytest

    from retargeting_apps.config import MujocoWebViewerConfig
    from retargeting_apps.visualization import mjviser_live

    class FailingScene:
        """Scene fake whose construction always fails."""

        def __init__(self, server, model, num_envs: int) -> None:
            """Raise after the server has already started.

            Args:
                server: Fake server ignored by the failure.
                model: Fake model ignored by the failure.
                num_envs: Environment count ignored by the failure.

            Returns:
                None because construction always raises.
            """
            del server, model, num_envs
            raise RuntimeError("scene failure")

    _FakeViserServer.instances.clear()
    fake_viser = SimpleNamespace(
        ViserServer=_FakeViserServer,
        Icon=SimpleNamespace(ADJUSTMENTS="adjustments"),
    )
    monkeypatch.setattr(
        mjviser_live,
        "_load_mjviser_dependencies",
        lambda: (_FAKE_MUJOCO, fake_viser, FailingScene),
    )

    with pytest.raises(RuntimeError, match="scene failure"):
        mjviser_live.MujocoWebVisualizer(object(), object(), MujocoWebViewerConfig(enabled=True))

    assert _FakeViserServer.instances[0].stop_count == 1
