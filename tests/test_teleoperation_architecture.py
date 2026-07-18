from pathlib import Path
import numpy as np


def test_profile_separates_objective_runtime_from_command_limits():
    """The profile exposes distinct algorithm and teleoperation config domains."""
    from retargeting.config import load_retargeting_profile_config
    from teleoperation.config import load_teleoperation_command_config

    profile_source = "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
    profile = load_retargeting_profile_config(profile_source)
    command_config = load_teleoperation_command_config(profile_source)

    assert profile.retargeting.arm_dof == 7
    assert len(profile.retargeting.joint_position_weights) == 23
    assert len(profile.retargeting.joint_velocity_weights) == 23
    assert len(command_config.max_joint_speed) == 23


def test_output_filter_is_independent_from_retargeting_solver_state():
    """Output smoothing remains a stateful teleoperation concern."""
    from teleoperation.config import load_teleoperation_mode_config
    from teleoperation.output import QposOutputFilter

    output_filter = QposOutputFilter(
        initial_qpos=np.array([0.0, 0.0]),
        mode_config=load_teleoperation_mode_config("configs/teleoperation_modes/real_world.yaml"),
    )

    np.testing.assert_allclose(output_filter.apply(np.array([1.0, -1.0])), np.array([0.3, -0.3]))
    np.testing.assert_allclose(output_filter.previous_qpos, np.array([0.3, -0.3]))


def test_avp_mapper_reset_clears_relative_wrist_alignment():
    """Reset the mapper without retaining robot or sensor wrist origins.

    Args:
        None.

    Returns:
        None.
    """
    from teleoperation.observation_mapping import AvpRelativeWristMapper

    mapper = AvpRelativeWristMapper.__new__(AvpRelativeWristMapper)
    mapper._robot_initial_wrist_pose = np.eye(4)
    mapper._sensor_initial_wrist_pose = np.eye(4)

    mapper.reset()

    assert mapper._robot_initial_wrist_pose is None
    assert mapper._sensor_initial_wrist_pose is None


def test_retargeting_core_has_no_detector_or_output_filter_dependency():
    """The pure solver boundary must not reach into teleoperation adapters."""
    source = (Path("src/retargeting/core") / "retargeter.py").read_text(encoding="utf-8")

    assert "retargeting.inputs.avp" not in source
    assert "retargeting.inputs.rgb" not in source
    assert "teleoperation.output" not in source
    assert "cv2" not in source


def test_optimizer_registry_is_owned_by_retargeting_core():
    """The canonical optimizer import path lives below the algorithm boundary."""
    from retargeting.core.optimizers import VectorWristJointOptimizer, get_optimizer_class

    assert get_optimizer_class("VectorWristJointOptimizer") is VectorWristJointOptimizer


def test_callback_solver_factory_is_owned_by_retargeting_core():
    """The callback solver factory is exposed from the retargeting core."""
    from retargeting.core.solvers import create_callback_solver
    from retargeting.core.solvers.callback import create_callback_solver as callback_create_callback_solver

    assert callback_create_callback_solver is create_callback_solver


def test_mujoco_backend_remains_headless_and_viewer_independent():
    """Protect the physics backend from optional GUI and vision imports.

    Args:
        None.

    Returns:
        None.
    """
    source = Path("src/teleoperation/backends/mujoco.py").read_text(encoding="utf-8")

    assert "mujoco.viewer" not in source
    assert "open3d" not in source
    assert "cv2" not in source


def test_flat_runtime_has_one_flow_owner_and_no_obsolete_controller_modules():
    """Protect the final controller and directory ownership boundaries.

    Args:
        None.

    Returns:
        None.
    """
    teleoperation_root = Path("src/teleoperation")
    flow_source = (teleoperation_root / "flow.py").read_text(encoding="utf-8")
    teleop_exe_source = Path("src/retargeting_apps/apps/teleop_exe.py").read_text(encoding="utf-8")
    ros_callback_source = Path(
        "ws_ros2/src/retargeting_benchmark/src/robot_teleoperation_ros.py"
    ).read_text(encoding="utf-8")

    assert flow_source.count("class BatchRetargetFlow:") == 1
    assert flow_source.count("class ExecutionFlow:") == 1
    assert "Mujoco" not in flow_source
    assert "self.backend.execute(" in flow_source
    assert "self.input.read()" in flow_source
    assert "self.observation_mapper.initialize(" in flow_source
    assert "self.retargeter" in flow_source
    for removed_path in (
        teleoperation_root / "batch.py",
        teleoperation_root / "session.py",
        teleoperation_root / "mujoco_runtime.py",
        teleoperation_root / "avp_alignment.py",
        teleoperation_root / "timing.py",
        teleoperation_root / "inputs" / "adapter.py",
        Path("src/retargeting_apps/pipelines"),
        Path("src/retargeting_apps/apps/mujoco_online_simulation.py"),
        Path("src/retargeting_apps/apps/mujoco_offline_simulation.py"),
    ):
        assert not removed_path.exists()
    assert "flow.run()" in teleop_exe_source
    assert "while True" not in teleop_exe_source
    assert "for frame" not in teleop_exe_source
    assert "decode_rgb_sample" in ros_callback_source
    assert "self.flow.step(sample)" in ros_callback_source


def test_ros_command_and_real_robot_backends_expose_shared_atomic_contract():
    """Keep ROS callback and hardware adapters compatible with ExecutionFlow.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting_ros.backends import RosCommandBackend
    from retargeting_ros.real_robot import RobotReal

    commands: list[np.ndarray] = []

    def execute(qpos: np.ndarray) -> np.ndarray:
        """Record one callback command and return a measured state.

        Args:
            qpos: Requested callback target.

        Returns:
            Deterministic measured state.
        """
        commands.append(qpos.copy())
        return qpos + 0.01

    backend = RosCommandBackend(
        initial_qpos=np.zeros(2),
        control_period=0.05,
        execute_callback=execute,
    )

    result = backend.execute(np.array([0.2, -0.3]))

    np.testing.assert_allclose(commands[-1], [0.2, -0.3])
    np.testing.assert_allclose(result.command_qpos, [0.2, -0.3])
    np.testing.assert_allclose(result.actual_qpos, [0.21, -0.29])
    for method_name in ("reset", "get_joint_pos", "get_target_joint_pos", "execute"):
        assert hasattr(RobotReal, method_name)
    assert hasattr(RobotReal, "control_period")
