from types import SimpleNamespace


def test_configure_initial_camera_sets_initial_and_reset_view_pose():
    """Verify replay configures the Viser pose used by new clients and Reset View.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting_apps.visualization.viser_replay import configure_initial_camera

    initial_camera = SimpleNamespace(position=None, look_at=None)
    server = SimpleNamespace(initial_camera=initial_camera)

    configure_initial_camera(server, position=(1.5, 1.5, 1.2), look_at=(0.0, 0.0, 0.45))

    assert initial_camera.position == (1.5, 1.5, 1.2)
    assert initial_camera.look_at == (0.0, 0.0, 0.45)
