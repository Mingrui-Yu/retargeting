from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("mujoco")


def _build_backend(ctrlrange_policy: str = "clip"):
    """Create the portable Panda+Leap backend for one headless test.

    Args:
        ctrlrange_policy: Actuator control-range behavior under test.

    Returns:
        Loaded robot config and headless MuJoCo backend.
    """
    from retargeting.config import load_robot_config
    from teleoperation.backends.mujoco import MujocoRobotBackend
    from teleoperation.config import load_mujoco_robot_binding_config

    robot_source = "configs/robots/panda_leap_paxini.yaml"
    robot_config = load_robot_config(robot_source)
    simulator_binding = load_mujoco_robot_binding_config(robot_source, robot_config=robot_config)
    backend = MujocoRobotBackend(
        model_path=simulator_binding.simulation_file_path,
        joint_names=robot_config.actuated_joints,
        initial_qpos=robot_config.initial_qpos,
        config={
            "command_hz": 20.0,
            "physics_timestep": 0.002,
            "realtime": False,
            "ctrlrange_policy": ctrlrange_policy,
        },
    )
    return robot_config, backend


def test_mujoco_backend_import_and_execution_are_headless():
    """Verify the backend loads and steps without optional viewer dependencies.

    Args:
        None.

    Returns:
        None.
    """
    robot_config, backend = _build_backend()

    assert Path(backend.model_path).is_file()
    assert backend.physics_steps_per_command == 25
    assert backend.control_period == pytest.approx(0.05)
    np.testing.assert_allclose(backend.get_joint_pos(), robot_config.initial_qpos)

    target = np.asarray(robot_config.initial_qpos, dtype=float)
    target[0] += 0.01
    applied = backend.ctrl_joint_pos(target)
    backend.step()

    np.testing.assert_allclose(applied, target)
    assert backend.data.time == pytest.approx(0.05)
    assert np.isfinite(backend.get_joint_pos()).all()
    assert backend.get_diagnostics()["simulation_time"] == pytest.approx(0.05)


def test_mujoco_backend_maps_compiled_qpos_by_joint_name():
    """Verify configured command order does not rely on compiled qpos order.

    Args:
        None.

    Returns:
        None.
    """
    robot_config, backend = _build_backend()

    assert backend.joint_names == robot_config.actuated_joints
    assert backend._qpos_addresses.tolist()[7:13] == [8, 7, 9, 10, 12, 11]
    probe = np.asarray(robot_config.initial_qpos, dtype=float) + np.linspace(
        -0.001, 0.001, len(robot_config.actuated_joints)
    )
    backend.reset(probe)

    np.testing.assert_allclose(backend.get_joint_pos(), probe)


def test_mujoco_backend_applies_configured_ctrlrange_policy():
    """Verify commands are clipped or rejected before reaching MuJoCo controls.

    Args:
        None.

    Returns:
        None.
    """
    robot_config, clipping_backend = _build_backend("clip")
    target = np.asarray(robot_config.initial_qpos, dtype=float)
    target[3] = -10.0
    applied = clipping_backend.ctrl_joint_pos(target)

    assert applied[3] == pytest.approx(-3.0718)

    _, error_backend = _build_backend("error")
    with pytest.raises(ValueError, match="panda_joint4"):
        error_backend.ctrl_joint_pos(target)
