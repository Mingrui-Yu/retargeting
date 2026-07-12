from pathlib import Path

import numpy as np


def test_profile_separates_objective_runtime_from_command_limits():
    """The profile exposes distinct algorithm and teleoperation config domains."""
    from retargeting.config import load_retargeting_profile_config

    profile = load_retargeting_profile_config(
        "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
    )

    assert profile.retargeting.arm_dof == 7
    assert len(profile.retargeting.joint_position_weights) == 23
    assert len(profile.retargeting.joint_velocity_weights) == 23
    assert len(profile.teleoperation.max_joint_speed) == 23


def test_output_filter_is_independent_from_retargeting_solver_state():
    """Output smoothing remains a stateful teleoperation concern."""
    from retargeting.config import load_teleoperation_mode_config
    from teleoperation.output import QposOutputFilter

    output_filter = QposOutputFilter(
        initial_qpos=np.array([0.0, 0.0]),
        mode_config=load_teleoperation_mode_config("configs/teleoperation_modes/real_world.yaml"),
    )

    np.testing.assert_allclose(output_filter.apply(np.array([1.0, -1.0])), np.array([0.3, -0.3]))
    np.testing.assert_allclose(output_filter.previous_qpos, np.array([0.3, -0.3]))


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
