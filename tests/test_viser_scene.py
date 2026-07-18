from __future__ import annotations

import numpy as np
from mr_utils.utils_mano import MANO_LINE_PAIRS

from retargeting.core.types import RetargetingHandObservation


class _FakeHandle:
    """Mutable Viser scene-node handle used by shared renderer tests."""

    def __init__(self, **properties) -> None:
        """Store initial scene-node properties.

        Args:
            **properties: Initial handle properties.

        Returns:
            None.
        """
        self.visible = True
        for name, value in properties.items():
            setattr(self, name, value)


class _FakeScene:
    """Viser scene fake supporting persistent hand geometry handles."""

    def __init__(self) -> None:
        """Initialize scene creation records.

        Args:
            None.

        Returns:
            None.
        """
        self.point_clouds = []
        self.line_segments = []
        self.frames = []

    def add_point_cloud(self, name: str, **properties) -> _FakeHandle:
        """Create and record one point-cloud handle.

        Args:
            name: Scene node name.
            **properties: Initial point-cloud properties.

        Returns:
            Mutable fake handle.
        """
        handle = _FakeHandle(name=name, **properties)
        self.point_clouds.append(handle)
        return handle

    def add_line_segments(self, name: str, **properties) -> _FakeHandle:
        """Create and record one line-segment handle.

        Args:
            name: Scene node name.
            **properties: Initial line-segment properties.

        Returns:
            Mutable fake handle.
        """
        handle = _FakeHandle(name=name, **properties)
        self.line_segments.append(handle)
        return handle

    def add_frame(self, name: str, **properties) -> _FakeHandle:
        """Create and record one coordinate-frame handle.

        Args:
            name: Scene node name.
            **properties: Initial frame properties.

        Returns:
            Mutable fake handle.
        """
        handle = _FakeHandle(name=name, **properties)
        self.frames.append(handle)
        return handle


class _FakeAtomic:
    """Count one atomic server update context."""

    def __init__(self, server) -> None:
        """Store the server whose atomic count is updated.

        Args:
            server: Fake Viser server.

        Returns:
            None.
        """
        self.server = server

    def __enter__(self):
        """Enter and count the atomic context.

        Args:
            None.

        Returns:
            This context manager.
        """
        self.server.atomic_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit without suppressing exceptions.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception value.
            traceback: Optional exception traceback.

        Returns:
            None.
        """
        del exc_type, exc_value, traceback


class _FakeServer:
    """Minimal Viser server exposing one scene and atomic updates."""

    def __init__(self, scene) -> None:
        """Store the target scene.

        Args:
            scene: Fake Viser scene.

        Returns:
            None.
        """
        self.scene = scene
        self.atomic_count = 0

    def atomic(self) -> _FakeAtomic:
        """Return one fake atomic update context.

        Args:
            None.

        Returns:
            Atomic context manager.
        """
        return _FakeAtomic(self)


def test_hand_renderer_transforms_observation_and_reuses_scene_handles():
    """Verify canonical observations update persistent world-frame geometry.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting_apps.visualization.viser_scene import ViserHandObservationRenderer

    scene = _FakeScene()
    server = _FakeServer(scene)
    renderer = ViserHandObservationRenderer(server, point_size=0.012)
    keypoints_wrist = np.arange(63, dtype=float).reshape(21, 3) * 0.001
    wrist_pose_world = np.eye(4)
    wrist_pose_world[:3, 3] = [0.4, -0.2, 0.1]
    observation = RetargetingHandObservation(
        keypoints_wrist=keypoints_wrist,
        wrist_pose_world=wrist_pose_world,
    )

    renderer.update_observation(observation)

    expected_world = keypoints_wrist + wrist_pose_world[:3, 3]
    assert len(scene.point_clouds) == len(scene.line_segments) == len(scene.frames) == 1
    np.testing.assert_allclose(scene.point_clouds[0].points, expected_world)
    start, end = MANO_LINE_PAIRS[0]
    np.testing.assert_allclose(scene.line_segments[0].points[0], [expected_world[start], expected_world[end]])
    np.testing.assert_allclose(scene.frames[0].position, wrist_pose_world[:3, 3])
    assert scene.point_clouds[0].point_size == 0.012

    renderer.update_world(expected_world + 1.0, wrist_pose_world)
    assert len(scene.point_clouds) == len(scene.line_segments) == len(scene.frames) == 1
    np.testing.assert_allclose(scene.point_clouds[0].points, expected_world + 1.0)

    renderer.hide()
    assert scene.point_clouds[0].visible is False
    assert scene.line_segments[0].visible is False
    assert scene.frames[0].visible is False
    assert server.atomic_count == 3


def test_hand_renderer_uses_spline_fallback_without_line_segment_support():
    """Verify older Viser scenes retain one persistent spline per MANO edge.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting_apps.visualization.viser_scene import ViserHandObservationRenderer

    class SplineScene:
        """Scene fake that deliberately omits add_line_segments."""

        def __init__(self) -> None:
            """Initialize spline and common node records.

            Args:
                None.

            Returns:
                None.
            """
            self.point_clouds = []
            self.splines = []
            self.frames = []

        def add_point_cloud(self, name: str, **properties) -> _FakeHandle:
            """Create one point-cloud handle.

            Args:
                name: Scene node name.
                **properties: Initial point-cloud properties.

            Returns:
                Mutable fake handle.
            """
            handle = _FakeHandle(name=name, **properties)
            self.point_clouds.append(handle)
            return handle

        def add_spline_catmull_rom(self, name: str, **properties) -> _FakeHandle:
            """Create one spline handle.

            Args:
                name: Scene node name.
                **properties: Initial spline properties.

            Returns:
                Mutable fake handle.
            """
            handle = _FakeHandle(name=name, **properties)
            self.splines.append(handle)
            return handle

        def add_frame(self, name: str, **properties) -> _FakeHandle:
            """Create one coordinate-frame handle.

            Args:
                name: Scene node name.
                **properties: Initial frame properties.

            Returns:
                Mutable fake handle.
            """
            handle = _FakeHandle(name=name, **properties)
            self.frames.append(handle)
            return handle

    scene = SplineScene()
    renderer = ViserHandObservationRenderer(_FakeServer(scene), point_size=0.01)
    keypoints_world = np.arange(63, dtype=float).reshape(21, 3)

    renderer.update_world(keypoints_world, np.eye(4))
    renderer.update_world(keypoints_world + 1.0, np.eye(4))

    assert len(scene.splines) == len(MANO_LINE_PAIRS)
    start, end = MANO_LINE_PAIRS[-1]
    np.testing.assert_allclose(
        scene.splines[-1].points,
        [keypoints_world[start] + 1.0, keypoints_world[end] + 1.0],
    )


def test_hand_renderer_rejects_invalid_point_size():
    """Verify invalid point sizes fail before any scene nodes are created.

    Args:
        None.

    Returns:
        None.
    """
    import pytest

    from retargeting_apps.visualization.viser_scene import ViserHandObservationRenderer

    with pytest.raises(ValueError, match="point_size"):
        ViserHandObservationRenderer(_FakeServer(_FakeScene()), point_size=0.0)
