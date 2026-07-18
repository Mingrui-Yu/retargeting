"""Backend-aware passive visualizer composition for execution apps."""

from __future__ import annotations

from typing import Any, Protocol

from retargeting.core.types import RetargetingHandObservation
from retargeting.config import load_retargeting_profile_config, load_robot_config
from retargeting_apps.config import load_execution_viewer_config
from retargeting_apps.visualization.execution.mjviser import create_mujoco_web_visualizer
from retargeting_apps.visualization.execution.viser import create_viser_live_visualizer
from teleoperation.config import load_execution_backend_config
from teleoperation.types import ExecutionStepResult


DEFAULT_VIEWER_TYPE_BY_BACKEND = {
    "mujoco": "mjviser",
    "kinematic": "viser",
}


class ExecutionVisualizer(Protocol):
    """Shared lifecycle for app-owned passive execution visualizers."""

    def update_observation(self, observation: RetargetingHandObservation) -> None:
        """Publish one valid canonical human-hand observation.

        Args:
            observation: Canonical mapped hand observation.

        Returns:
            None.
        """
        ...

    def hide_observation(self) -> None:
        """Hide human-hand nodes when the current source has no observation.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def wait_for_client(self) -> None:
        """Wait for a client when the visualizer is configured to block startup.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def wait_after_completion(self) -> None:
        """Keep the final frame visible when configured for offline playback.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def close(self) -> None:
        """Release viewer-owned resources.

        Args:
            None.

        Returns:
            None.
        """
        ...


def _resolve_visualizer_type(requested_type: str, backend_name: str) -> str:
    """Resolve automatic viewer selection against the configured backend.

    Args:
        requested_type: Configured viewer type, usually ``auto``.
        backend_name: Execution backend name from the composed app config.

    Returns:
        Concrete visualizer type.
    """
    if requested_type != "auto":
        return requested_type
    try:
        return DEFAULT_VIEWER_TYPE_BY_BACKEND[backend_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend for automatic viewer selection: {backend_name!r}.") from exc


def _attach_observation_observers(flow: Any, visualizer: ExecutionVisualizer) -> None:
    """Bind source-frame observation updates shared by every execution viewer.

    Args:
        flow: Execution flow publishing immutable source-frame results.
        visualizer: Viewer receiving mapped human-hand observations.

    Returns:
        None.
    """
    def update_hand(result: ExecutionStepResult) -> None:
        """Publish a valid observation or hide stale human-hand geometry.

        Args:
            result: Completed execution source-frame result.

        Returns:
            None.
        """
        frame = result.retargeted_frame
        if frame is None:
            visualizer.hide_observation()
        else:
            visualizer.update_observation(frame.observation)

    flow.add_step_observer(update_hand)
    flow.add_reset_observer(lambda _: visualizer.hide_observation())


def _attach_mjviser(config_data: dict[str, Any], flow: Any) -> ExecutionVisualizer:
    """Create and attach a passive mjviser adapter to a MuJoCo-backed flow.

    Args:
        config_data: Plain composed execution app config.
        flow: Execution flow whose backend owns MuJoCo model/data.

    Returns:
        Attached mjviser visualizer.
    """
    viewer_config = load_execution_viewer_config(config_data.get("viewer"))
    model = getattr(flow.backend, "model", None)
    data = getattr(flow.backend, "data", None)
    if model is None or data is None:
        raise TypeError("viewer.type=mjviser requires a MuJoCo backend exposing model and data.")
    visualizer = create_mujoco_web_visualizer(model, data, viewer_config)
    visualizer.update()
    flow.add_command_observer(lambda result: visualizer.update())
    flow.add_reset_observer(lambda qpos: visualizer.update())
    _attach_observation_observers(flow, visualizer)
    return visualizer


def _attach_viser(config_data: dict[str, Any], flow: Any) -> ExecutionVisualizer:
    """Create and attach a passive Viser URDF adapter to an execution flow.

    Args:
        config_data: Plain composed execution app config.
        flow: Execution flow publishing backend qpos states.

    Returns:
        Attached Viser visualizer.
    """
    viewer_config = load_execution_viewer_config(config_data.get("viewer"))
    profile_config = load_retargeting_profile_config(config_data["profile"])
    robot_config = load_robot_config(profile_config.robot)
    visualizer = create_viser_live_visualizer(
        robot_file_path=robot_config.robot_file_path,
        actuated_joint_names=robot_config.actuated_joints,
        config=viewer_config,
    )
    visualizer.update_qpos(flow.backend.get_joint_pos())
    flow.add_command_observer(lambda result: visualizer.update_qpos(result.actual_qpos))
    flow.add_reset_observer(lambda qpos: visualizer.update_qpos(qpos))
    _attach_observation_observers(flow, visualizer)
    return visualizer


def create_optional_execution_visualizer(config_data: dict[str, Any], flow: Any) -> ExecutionVisualizer | None:
    """Create, attach, and optionally block on the configured execution visualizer.

    Args:
        config_data: Plain composed execution app config.
        flow: Execution flow that emits passive observer events.

    Returns:
        Attached visualizer, or None when visualization is disabled.
    """
    viewer_config = load_execution_viewer_config(config_data.get("viewer"))
    if not viewer_config.enabled:
        return None
    backend_config = load_execution_backend_config(config_data.get("backend"))
    visualizer_type = _resolve_visualizer_type(viewer_config.type, backend_config.name)
    if visualizer_type == "mjviser":
        visualizer = _attach_mjviser(config_data, flow)
    elif visualizer_type == "viser":
        visualizer = _attach_viser(config_data, flow)
    else:
        raise ValueError(f"Unsupported execution viewer type: {visualizer_type!r}.")
    if viewer_config.wait_for_client:
        visualizer.wait_for_client()
    return visualizer


__all__ = ["DEFAULT_VIEWER_TYPE_BY_BACKEND", "ExecutionVisualizer", "create_optional_execution_visualizer"]
