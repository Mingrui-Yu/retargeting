"""Passive Viser URDF adapter for backend-neutral execution flows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from retargeting.core.types import RetargetingHandObservation
from retargeting_apps.config import MujocoWebViewerConfig
from retargeting_apps.visualization.viser_scene import (
    ViserHandObservationRenderer,
    configure_initial_camera,
)


def _load_viser_dependencies() -> tuple[Any, Any]:
    """Load optional Viser dependencies only for an enabled live viewer.

    Args:
        None.

    Returns:
        The Viser module and ViserUrdf helper class.
    """
    try:
        import viser
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Viser execution visualization is not installed. Install it in the retargeting environment."
        ) from exc
    return viser, ViserUrdf


class ViserLiveVisualizer:
    """Publish backend joint states through a standard Viser URDF scene."""

    def __init__(
        self,
        *,
        robot_file_path: str | Path,
        actuated_joint_names: Sequence[str],
        config: MujocoWebViewerConfig,
        sleep: Any = time.sleep,
    ) -> None:
        """Create a passive URDF scene for an externally stepped backend.

        Args:
            robot_file_path: URDF path for the robot embodiment being commanded.
            actuated_joint_names: Backend qpos names in execution command order.
            config: Validated execution viewer configuration.
            sleep: Wait dependency used for client and completion waits.

        Returns:
            None.
        """
        config.validate()
        _, urdf_class = _load_viser_dependencies()
        self.config = config
        self.actuated_joint_names = tuple(str(name) for name in actuated_joint_names)
        if not self.actuated_joint_names or len(set(self.actuated_joint_names)) != len(self.actuated_joint_names):
            raise ValueError("actuated_joint_names must be non-empty and unique.")
        self._sleep = sleep
        self._closed = False
        self.server = self._create_server(config)
        try:
            configure_initial_camera(
                self.server,
                position=config.initial_camera_position,
                look_at=config.initial_camera_look_at,
            )
            self.robot_urdf = urdf_class(
                self.server,
                Path(robot_file_path),
                root_node_name="/robot_mesh",
                load_meshes=True,
                load_collision_meshes=False,
            )
            self._urdf_joint_names = tuple(str(name) for name in self.robot_urdf.get_actuated_joint_names())
            self._qpos_indices = self._resolve_urdf_qpos_indices()
            self._hand_renderer = ViserHandObservationRenderer(
                self.server,
                point_size=config.human_keypoint_size,
            )
        except Exception:
            self.server.stop()
            raise
        print(f"Viser server listening on {config.host}:{self.server.get_port()}.")

    @staticmethod
    def _create_server(config: MujocoWebViewerConfig) -> Any:
        """Create a Viser server while tolerating older constructors without host support.

        Args:
            config: Validated execution viewer configuration.

        Returns:
            Started Viser server instance.
        """
        viser, _ = _load_viser_dependencies()
        try:
            return viser.ViserServer(host=config.host, port=config.port)
        except TypeError:
            return viser.ViserServer(port=config.port)

    def _resolve_urdf_qpos_indices(self) -> np.ndarray:
        """Map URDF actuated-joint order back to backend command order.

        Args:
            None.

        Returns:
            Integer indices that reorder backend qpos into ViserUrdf order.
        """
        qpos_by_name = {name: index for index, name in enumerate(self.actuated_joint_names)}
        missing = [name for name in self._urdf_joint_names if name not in qpos_by_name]
        if missing:
            raise ValueError("URDF actuated joints are missing from backend joint order: " + ", ".join(missing))
        return np.asarray([qpos_by_name[name] for name in self._urdf_joint_names], dtype=int)

    def update_qpos(self, qpos: Sequence[float]) -> None:
        """Publish one backend qpos vector through the URDF visualizer.

        Args:
            qpos: Backend joint positions in execution command order.

        Returns:
            None.
        """
        values = np.asarray(qpos, dtype=float)
        expected_shape = (len(self.actuated_joint_names),)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"qpos must be finite and have shape {expected_shape}.")
        self.robot_urdf.update_cfg(values[self._qpos_indices])

    def update_observation(self, observation: RetargetingHandObservation) -> None:
        """Publish one canonical human-hand observation beside the robot URDF.

        Args:
            observation: Canonical hand observation produced by the mapping layer.

        Returns:
            None.
        """
        self._hand_renderer.update_observation(observation)

    def hide_observation(self) -> None:
        """Hide the current human hand when no valid observation is available.

        Args:
            None.

        Returns:
            None.
        """
        self._hand_renderer.hide()

    def wait_for_client(self) -> None:
        """Optionally wait until at least one browser has connected.

        Args:
            None.

        Returns:
            None.
        """
        if not self.config.wait_for_client:
            return
        if not hasattr(self.server, "get_clients"):
            return
        print("Waiting for a Viser browser client before starting execution.")
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
        print("Offline execution complete; press Ctrl+C to stop the Viser server.")
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


def create_viser_live_visualizer(
    *,
    robot_file_path: str | Path,
    actuated_joint_names: Sequence[str],
    config: MujocoWebViewerConfig,
) -> ViserLiveVisualizer:
    """Create the concrete passive Viser URDF adapter.

    Args:
        robot_file_path: URDF path for the robot embodiment being commanded.
        actuated_joint_names: Backend qpos names in execution command order.
        config: Validated execution viewer configuration.

    Returns:
        Ready passive Viser live visualizer.
    """
    return ViserLiveVisualizer(
        robot_file_path=robot_file_path,
        actuated_joint_names=actuated_joint_names,
        config=config,
    )


__all__ = ["ViserLiveVisualizer", "create_viser_live_visualizer"]
