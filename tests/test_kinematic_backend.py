from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from teleoperation.backends import KinematicRobotBackend


def test_kinematic_backend_realizes_commands_and_detaches_state() -> None:
    """An ideal backend publishes exact commands without sharing mutable buffers.

    Args:
        None.

    Returns:
        None.
    """
    initial_qpos = np.array([0.1, -0.2])
    backend = KinematicRobotBackend(initial_qpos=initial_qpos, control_period=0.05)
    initial_qpos[:] = 9.0

    command = np.array([0.3, -0.4])
    result = backend.execute(command)
    command[:] = 8.0

    assert backend.control_period == 0.05
    np.testing.assert_allclose(backend.get_target_joint_pos(), [0.3, -0.4])
    np.testing.assert_allclose(backend.get_joint_pos(), [0.3, -0.4])
    np.testing.assert_allclose(result.command_qpos, result.actual_qpos)
    assert dict(result.diagnostics) == {}
    with pytest.raises(ValueError):
        result.actual_qpos[0] = 1.0


def test_kinematic_backend_reset_synchronizes_target_and_actual_state() -> None:
    """Reset restores either an explicit configuration or the initial state.

    Args:
        None.

    Returns:
        None.
    """
    backend = KinematicRobotBackend(initial_qpos=[0.1, -0.2], control_period=0.05)
    backend.execute(np.array([0.3, -0.4]))

    backend.reset(np.array([-0.5, 0.6]))

    np.testing.assert_allclose(backend.get_target_joint_pos(), [-0.5, 0.6])
    np.testing.assert_allclose(backend.get_joint_pos(), [-0.5, 0.6])

    backend.reset()

    np.testing.assert_allclose(backend.get_target_joint_pos(), [0.1, -0.2])
    np.testing.assert_allclose(backend.get_joint_pos(), [0.1, -0.2])


@pytest.mark.parametrize(
    "initial_qpos",
    ([], [[0.0]], [0.0, np.nan], [0.0, np.inf]),
)
def test_kinematic_backend_rejects_invalid_initial_state(initial_qpos) -> None:
    """Construction rejects empty, non-vector, and non-finite robot state.

    Args:
        initial_qpos: Invalid initial state selected by pytest.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="initial_qpos"):
        KinematicRobotBackend(initial_qpos=initial_qpos, control_period=0.05)


@pytest.mark.parametrize("control_period", (True, 0.0, -0.1, np.nan, np.inf))
def test_kinematic_backend_rejects_invalid_control_period(control_period) -> None:
    """Construction requires a positive finite non-boolean command period.

    Args:
        control_period: Invalid command period selected by pytest.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="control_period"):
        KinematicRobotBackend(initial_qpos=[0.0, 0.0], control_period=control_period)


@pytest.mark.parametrize("qpos", ([0.0], [[0.0, 0.0]], [0.0, np.nan], [0.0, np.inf]))
def test_kinematic_backend_rejects_invalid_commands_without_changing_state(qpos) -> None:
    """Invalid execute and reset requests leave the previous ideal state intact.

    Args:
        qpos: Invalid command selected by pytest.

    Returns:
        None.
    """
    backend = KinematicRobotBackend(initial_qpos=[0.1, -0.2], control_period=0.05)

    with pytest.raises(ValueError, match="qpos"):
        backend.execute(np.asarray(qpos))
    with pytest.raises(ValueError, match="qpos"):
        backend.reset(np.asarray(qpos))

    np.testing.assert_allclose(backend.get_target_joint_pos(), [0.1, -0.2])
    np.testing.assert_allclose(backend.get_joint_pos(), [0.1, -0.2])


def test_kinematic_backend_has_no_model_hardware_or_viewer_dependency() -> None:
    """Keep ideal joint-state execution independent of optional runtime systems.

    Args:
        None.

    Returns:
        None.
    """
    source = Path("src/teleoperation/backends/kinematic.py").read_text(encoding="utf-8").lower()

    for forbidden in ("pinocchio", "mujoco", "rclpy", "viser", "open3d", "cv2"):
        assert forbidden not in source
