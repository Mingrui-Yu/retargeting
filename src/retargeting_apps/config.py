"""Configuration helpers and schemas owned by application composition."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retargeting.config.io import load_config_source, resolve_project_path, to_plain_config_data

__all__ = [
    "MujocoWebViewerConfig",
    "ReplayAppConfig",
    "ViewerConfig",
    "load_mujoco_web_viewer_config",
    "load_replay_app_config",
    "resolve_project_path",
    "to_plain_config_data",
]


@dataclass(frozen=True)
class MujocoWebViewerConfig:
    """Application-owned settings for passive MuJoCo Web visualization."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 9219
    wait_for_client: bool = True
    keep_open_after_completion: bool = False
    camera_distance: float = -1.0
    camera_azimuth: float = 120.0
    camera_elevation: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MujocoWebViewerConfig":
        """Build passive MuJoCo viewer settings from config data.

        Args:
            data: Optional mapping with Web server, lifecycle, and camera settings.

        Returns:
            Typed passive MuJoCo viewer configuration.
        """
        data = {} if data is None else data
        return cls(
            enabled=bool(data.get("enabled", False)),
            host=str(data.get("host", "0.0.0.0")),
            port=int(data.get("port", 9219)),
            wait_for_client=bool(data.get("wait_for_client", True)),
            keep_open_after_completion=bool(data.get("keep_open_after_completion", False)),
            camera_distance=float(data.get("camera_distance", -1.0)),
            camera_azimuth=float(data.get("camera_azimuth", 120.0)),
            camera_elevation=float(data.get("camera_elevation", 20.0)),
        )

    def validate(self) -> None:
        """Validate Web server and camera values before optional imports occur.

        Args:
            None.

        Returns:
            None.
        """
        if not self.host.strip():
            raise ValueError("MuJoCo Web viewer host must not be empty.")
        if not 0 <= self.port <= 65535:
            raise ValueError(f"MuJoCo Web viewer port must be in [0, 65535], got {self.port}.")
        camera_values = (self.camera_distance, self.camera_azimuth, self.camera_elevation)
        if not all(math.isfinite(value) for value in camera_values):
            raise ValueError("MuJoCo Web viewer camera settings must be finite.")


@dataclass(frozen=True)
class ViewerConfig:
    fps: float = 30.0
    port: int = 8080
    no_robot_mesh: bool = False
    trail_length: int = 120
    human_keypoint_size: float = 0.018
    initial_camera_position: tuple[float, float, float] = (1.5, 1.5, 1.2)
    initial_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.45)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ViewerConfig":
        """Build replay viewer settings from config data.

        Args:
            data: Optional mapping with viewer rendering and camera settings.

        Returns:
            Typed replay viewer config.
        """
        data = {} if data is None else data
        return cls(
            fps=float(data.get("fps", 30.0)),
            port=int(data.get("port", 8080)),
            no_robot_mesh=bool(data.get("no_robot_mesh", False)),
            trail_length=int(data.get("trail_length", 120)),
            human_keypoint_size=float(data.get("human_keypoint_size", 0.018)),
            initial_camera_position=tuple(
                float(value) for value in data.get("initial_camera_position", (1.5, 1.5, 1.2))
            ),
            initial_camera_look_at=tuple(
                float(value) for value in data.get("initial_camera_look_at", (0.0, 0.0, 0.45))
            ),
        )


@dataclass(frozen=True)
class ReplayAppConfig:
    run_name: str | None
    runtime_root: str
    viewer: ViewerConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayAppConfig":
        """Build replay app settings from config data.

        Args:
            data: Mapping with runtime artifact and viewer settings.

        Returns:
            Typed replay app config.
        """
        return cls(
            run_name=None if data.get("run_name") is None else str(data["run_name"]),
            runtime_root=str(data.get("runtime_root", "outputs")),
            viewer=ViewerConfig.from_dict(data.get("viewer")),
        )


def load_replay_app_config(path: str | Path | Mapping[str, Any] | ReplayAppConfig) -> ReplayAppConfig:
    """Load replay app settings.

    Args:
        path: YAML path, composed mapping, or typed config.

    Returns:
        Typed replay app config.
    """
    if isinstance(path, ReplayAppConfig):
        return path
    return ReplayAppConfig.from_dict(load_config_source(path))


def load_mujoco_web_viewer_config(
    source: str | Path | Mapping[str, Any] | MujocoWebViewerConfig | None,
) -> MujocoWebViewerConfig:
    """Load and validate passive MuJoCo Web viewer settings.

    Args:
        source: Config path, composed mapping, typed config, or None for defaults.

    Returns:
        Validated passive MuJoCo viewer configuration.
    """
    if isinstance(source, MujocoWebViewerConfig):
        config = source
    elif source is None:
        config = MujocoWebViewerConfig()
    else:
        config = MujocoWebViewerConfig.from_dict(load_config_source(source))
    config.validate()
    return config
