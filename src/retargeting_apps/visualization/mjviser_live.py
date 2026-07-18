"""Passive mjviser adapter for an application-owned MuJoCo loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from retargeting_apps.config import MujocoWebViewerConfig


_JOINT_ANGLE_DECIMALS = 5


def _load_mjviser_dependencies() -> tuple[Any, Any, Any]:
    """Load optional Web viewer dependencies only for an enabled viewer.

    Args:
        None.

    Returns:
        The MuJoCo module, viser module, and mjviser scene class.
    """
    try:
        import mujoco
        import viser
        from mjviser import ViserMujocoScene
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo Web visualization is not installed. "
            "Install it with `pip install -e \".[mujoco-web]\"`."
        ) from exc
    return mujoco, viser, ViserMujocoScene


class MujocoWebVisualizer:
    """Publish one externally stepped MuJoCo simulation through mjviser."""

    def __init__(
        self,
        model: Any,
        data: Any,
        config: MujocoWebViewerConfig,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a passive scene around an existing MuJoCo model and data.

        Args:
            model: MuJoCo model owned by the execution backend.
            data: Live MuJoCo data owned and stepped by the execution backend.
            config: Validated Web viewer configuration.
            sleep: Wait dependency used for client and completion waits.

        Returns:
            None.
        """
        config.validate()
        mujoco, viser, scene_class = _load_mjviser_dependencies()
        self.config = config
        self.model = model
        self.data = data
        self._sleep = sleep
        self._closed = False
        self.server = viser.ViserServer(host=config.host, port=config.port)
        try:
            self.scene = scene_class(self.server, model, num_envs=1)
            tabs = self.scene.create_visualization_gui(
                camera_distance=config.camera_distance,
                camera_azimuth=config.camera_azimuth,
                camera_elevation=config.camera_elevation,
            )
            self._joint_angle_handles = self._create_joint_angle_gui(mujoco, viser, tabs)
        except Exception:
            self.server.stop()
            raise
        print(f"mjviser server listening on {config.host}:{self.server.get_port()}.")

    def _create_joint_angle_gui(self, mujoco: Any, viser: Any, tabs: Any) -> list[tuple[Any, int]]:
        """Create read-only angle fields for every hinge joint in model order.

        Args:
            mujoco: Imported MuJoCo module used for joint enums and names.
            viser: Imported Viser module used for the tab icon.
            tabs: Existing mjviser tab group extended by this adapter.

        Returns:
            Pairs of GUI number handles and their corresponding qpos addresses.
        """
        handles: list[tuple[Any, int]] = []
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        with tabs.add_tab("Joint angles", icon=viser.Icon.ADJUSTMENTS):
            for joint_id in range(self.model.njnt):
                if int(self.model.jnt_type[joint_id]) != hinge_type:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if not name:
                    name = f"joint_{joint_id}"
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                value = round(float(self.data.qpos[qpos_address]), _JOINT_ANGLE_DECIMALS)
                handle = self.server.gui.add_number(
                    f"{name} [rad]",
                    initial_value=value,
                    step=10.0 ** -_JOINT_ANGLE_DECIMALS,
                    disabled=True,
                    hint="Current MuJoCo joint angle in radians.",
                )
                handles.append((handle, qpos_address))
        return handles

    def _update_joint_angles(self) -> None:
        """Atomically refresh all joint-angle fields from the live qpos state.

        Args:
            None.

        Returns:
            None.
        """
        with self.server.atomic():
            for handle, qpos_address in self._joint_angle_handles:
                value = round(float(self.data.qpos[qpos_address]), _JOINT_ANGLE_DECIMALS)
                if handle.value != value:
                    handle.value = value

    def update(self) -> None:
        """Publish the backend's current MuJoCo state without stepping it.

        Args:
            None.

        Returns:
            None.
        """
        self.scene.update_from_mjdata(self.data)
        self._update_joint_angles()

    def wait_for_client(self) -> None:
        """Optionally wait until at least one browser has connected.

        Args:
            None.

        Returns:
            None.
        """
        if not self.config.wait_for_client:
            return
        print("Waiting for an mjviser browser client before starting simulation.")
        while not self.server.get_clients():
            self._sleep(0.05)

    def wait_after_completion(self) -> None:
        """Optionally retain the final scene until the user interrupts it.

        Args:
            None.

        Returns:
            None.
        """
        if not self.config.keep_open_after_completion:
            return
        print("Offline simulation complete; press Ctrl+C to stop the mjviser server.")
        try:
            while True:
                self._sleep(1.0)
        except KeyboardInterrupt:
            pass

    def close(self) -> None:
        """Stop the Viser server exactly once.

        Args:
            None.

        Returns:
            None.
        """
        if self._closed:
            return
        self._closed = True
        self.server.stop()


def create_mujoco_web_visualizer(
    model: Any,
    data: Any,
    config: MujocoWebViewerConfig,
) -> MujocoWebVisualizer:
    """Create the concrete passive mjviser adapter.

    Args:
        model: MuJoCo model owned by the execution backend.
        data: Live MuJoCo data owned by the execution backend.
        config: Validated Web viewer configuration.

    Returns:
        Ready passive MuJoCo Web visualizer.
    """
    return MujocoWebVisualizer(model=model, data=data, config=config)


__all__ = ["MujocoWebVisualizer", "create_mujoco_web_visualizer"]
