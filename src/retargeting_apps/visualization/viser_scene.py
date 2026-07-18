"""Shared Viser scene helpers for live and replay visualization."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from mr_utils.utils_calc import transformPositions
from mr_utils.utils_mano import MANO_LINE_PAIRS, MANO_POINTS_COLORS
from scipy.spatial.transform import Rotation as sciR

from retargeting.core.types import RetargetingHandObservation


HUMAN_COLOR = (40, 180, 255)
_WRIST_AXES_LENGTH = 0.06
_WRIST_AXES_RADIUS = 0.004


def configure_initial_camera(
    server: Any,
    position: tuple[float, float, float],
    look_at: tuple[float, float, float],
) -> None:
    """Configure Viser's initial and reset-view camera pose.

    Args:
        server: Viser server whose initial camera should be configured.
        position: Three-dimensional camera position in Viser world coordinates.
        look_at: Three-dimensional point that the camera initially targets.

    Returns:
        None.
    """
    # Viser applies this pose to new clients and uses it for the Reset View action.
    server.initial_camera.position = position
    server.initial_camera.look_at = look_at


def rotation_matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to Viser's scalar-first quaternion order.

    Args:
        rotation_matrix: Three-by-three rotation matrix.

    Returns:
        Quaternion ordered as ``wxyz``.
    """
    xyzw = sciR.from_matrix(rotation_matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _hand_line_segments(keypoints_world: np.ndarray) -> np.ndarray:
    """Build MANO skeleton line segments from world-frame keypoints.

    Args:
        keypoints_world: World-frame hand keypoints with shape ``(21, 3)``.

    Returns:
        Skeleton endpoints with shape ``(n_connections, 2, 3)``.
    """
    return np.asarray([[keypoints_world[start], keypoints_world[end]] for start, end in MANO_LINE_PAIRS])


class ViserHandObservationRenderer:
    """Render one current human-hand observation into a shared Viser scene."""

    def __init__(
        self,
        server: Any,
        *,
        point_size: float,
        root_node_name: str = "/current/human",
    ) -> None:
        """Create a lazy renderer that reuses scene handles across frames.

        Args:
            server: Viser server owning the target scene.
            point_size: Rendered diameter of each human keypoint.
            root_node_name: Scene path under which hand nodes are created.

        Returns:
            None.
        """
        self.server = server
        self.scene = server.scene
        self.point_size = float(point_size)
        if not np.isfinite(self.point_size) or self.point_size <= 0.0:
            raise ValueError("point_size must be positive and finite.")
        self.root_node_name = root_node_name.rstrip("/")
        self._keypoint_handle: Any | None = None
        self._skeleton_handle: Any | list[Any] | None = None
        self._wrist_handle: Any | None = None

    def _atomic(self) -> Any:
        """Return an atomic Viser update context when supported.

        Args:
            None.

        Returns:
            Server atomic context, or a no-op context for older Viser versions.
        """
        atomic = getattr(self.server, "atomic", None)
        return atomic() if callable(atomic) else nullcontext()

    @staticmethod
    def _validate_world_data(
        keypoints_world: np.ndarray,
        wrist_pose_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and validate one world-frame hand pose.

        Args:
            keypoints_world: World-frame hand keypoints.
            wrist_pose_world: World-frame wrist homogeneous transform.

        Returns:
            Validated keypoint and wrist-pose arrays.
        """
        keypoints = np.asarray(keypoints_world, dtype=float)
        wrist_pose = np.asarray(wrist_pose_world, dtype=float)
        if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
            raise ValueError("keypoints_world must be finite and have shape (21, 3).")
        if wrist_pose.shape != (4, 4) or not np.isfinite(wrist_pose).all():
            raise ValueError("wrist_pose_world must be finite and have shape (4, 4).")
        return keypoints, wrist_pose

    def _create_skeleton(self, segments: np.ndarray) -> Any | list[Any] | None:
        """Create the preferred line-segment skeleton or its spline fallback.

        Args:
            segments: Skeleton endpoints with shape ``(n_connections, 2, 3)``.

        Returns:
            One line-segment handle, a list of spline handles, or None.
        """
        name = f"{self.root_node_name}/skeleton"
        if hasattr(self.scene, "add_line_segments"):
            colors = np.tile(np.asarray(HUMAN_COLOR, dtype=np.uint8), (segments.shape[0], 2, 1))
            return self.scene.add_line_segments(name, points=segments, colors=colors, line_width=2.0)
        if not hasattr(self.scene, "add_spline_catmull_rom"):
            return None
        return [
            self.scene.add_spline_catmull_rom(
                f"{name}/{index}",
                points=segment,
                color=HUMAN_COLOR,
                line_width=2.0,
            )
            for index, segment in enumerate(segments)
        ]

    def _update_skeleton(self, segments: np.ndarray) -> None:
        """Create or update the current MANO skeleton handles.

        Args:
            segments: Skeleton endpoints with shape ``(n_connections, 2, 3)``.

        Returns:
            None.
        """
        if self._skeleton_handle is None:
            self._skeleton_handle = self._create_skeleton(segments)
            return
        if isinstance(self._skeleton_handle, list):
            for handle, segment in zip(self._skeleton_handle, segments):
                handle.points = segment
                handle.visible = True
            return
        self._skeleton_handle.points = segments
        self._skeleton_handle.visible = True

    def update_observation(self, observation: RetargetingHandObservation) -> None:
        """Render one canonical wrist-frame observation in robot-world coordinates.

        Args:
            observation: Canonical hand observation produced by the mapping layer.

        Returns:
            None.
        """
        keypoints_world = transformPositions(
            observation.keypoints_wrist,
            target_frame_pose_inv=observation.wrist_pose_world,
        )
        self.update_world(keypoints_world, observation.wrist_pose_world)

    def update_world(self, keypoints_world: np.ndarray, wrist_pose_world: np.ndarray) -> None:
        """Render one hand whose keypoints and wrist pose are already in world coordinates.

        Args:
            keypoints_world: World-frame hand keypoints with shape ``(21, 3)``.
            wrist_pose_world: World-frame wrist homogeneous transform.

        Returns:
            None.
        """
        keypoints, wrist_pose = self._validate_world_data(keypoints_world, wrist_pose_world)
        segments = _hand_line_segments(keypoints)
        with self._atomic():
            if self._keypoint_handle is None:
                self._keypoint_handle = self.scene.add_point_cloud(
                    f"{self.root_node_name}/keypoints",
                    points=keypoints,
                    colors=np.asarray(MANO_POINTS_COLORS, dtype=np.uint8),
                    point_size=self.point_size,
                )
            else:
                self._keypoint_handle.points = keypoints
                self._keypoint_handle.visible = True
            self._update_skeleton(segments)
            if self._wrist_handle is None and hasattr(self.scene, "add_frame"):
                self._wrist_handle = self.scene.add_frame(
                    f"{self.root_node_name}/wrist",
                    wxyz=rotation_matrix_to_wxyz(wrist_pose[:3, :3]),
                    position=wrist_pose[:3, 3],
                    axes_length=_WRIST_AXES_LENGTH,
                    axes_radius=_WRIST_AXES_RADIUS,
                )
            elif self._wrist_handle is not None:
                self._wrist_handle.wxyz = rotation_matrix_to_wxyz(wrist_pose[:3, :3])
                self._wrist_handle.position = wrist_pose[:3, 3]
                self._wrist_handle.visible = True

    def hide(self) -> None:
        """Hide all current human-hand nodes without destroying their handles.

        Args:
            None.

        Returns:
            None.
        """
        handles = [self._keypoint_handle, self._skeleton_handle, self._wrist_handle]
        with self._atomic():
            for handle in handles:
                if isinstance(handle, list):
                    for child in handle:
                        child.visible = False
                elif handle is not None:
                    handle.visible = False


__all__ = ["ViserHandObservationRenderer", "configure_initial_camera"]
