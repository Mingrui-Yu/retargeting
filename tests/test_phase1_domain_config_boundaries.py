from __future__ import annotations

from dataclasses import fields
from pathlib import Path


def test_canonical_hand_contract_is_owned_by_retargeting_core():
    """Require canonical hand contracts to live at the core boundary.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.core import HandInput, HandObservation
    from retargeting.core.types import HandInput as TypesHandInput
    from retargeting.core.types import HandObservation as TypesHandObservation

    assert HandObservation is TypesHandObservation
    assert HandInput is TypesHandInput


def test_core_robot_and_profile_configs_exclude_execution_policy():
    """Keep simulator models and command limits outside pure core dataclasses.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.config.core import RetargetingProfileConfig, RobotConfig

    assert "simulation_model" not in {field.name for field in fields(RobotConfig)}
    assert "teleoperation" not in {field.name for field in fields(RetargetingProfileConfig)}
    assert not hasattr(RobotConfig, "simulation_file_path")


def test_profile_yaml_loads_equivalent_core_and_command_configs():
    """Split one unchanged profile mapping into core and command ownership.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.config import load_config_data, load_retargeting_profile_config, load_robot_config
    from teleoperation.config import load_teleoperation_command_config

    profile_data = load_config_data("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    profile_config = load_retargeting_profile_config(profile_data)
    robot_config = load_robot_config(profile_config.robot)
    command_config = load_teleoperation_command_config(profile_data["teleoperation"], robot_config=robot_config)

    assert profile_config.retargeting.arm_dof == 7
    assert len(profile_config.retargeting.joint_position_weights) == 23
    assert len(profile_config.retargeting.joint_velocity_weights) == 23
    assert command_config.max_joint_speed == tuple(
        float(value) for value in profile_data["teleoperation"]["max_joint_speed"]
    )
    assert command_config.max_joint_speed[:7] == (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
    assert command_config.max_joint_speed[7:] == (1.0,) * 16


def test_robot_yaml_loads_equivalent_core_and_mujoco_binding():
    """Split one unchanged robot mapping into kinematic and simulator ownership.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.config import load_config_data, load_robot_config
    from teleoperation.config import load_mujoco_robot_binding_config

    robot_data = load_config_data("configs/robots/panda_leap_paxini.yaml")
    robot_config = load_robot_config(robot_data)
    binding_config = load_mujoco_robot_binding_config(robot_data, robot_config=robot_config)

    assert robot_config.model.type == "urdf"
    assert binding_config.robot_name == robot_config.name == "panda_leap_paxini"
    assert binding_config.model.type == "mjcf"
    assert binding_config.model.path == robot_data["simulation_model"]["path"]
    assert Path(binding_config.simulation_file_path).is_file()


def test_hydra_composition_keeps_phase0_profile_and_robot_values():
    """Verify the ownership split does not change composed YAML values.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.config import load_robot_config
    from retargeting_apps.main import compose_hydra_base_config
    from teleoperation.config import load_mujoco_robot_binding_config, load_teleoperation_command_config

    config = compose_hydra_base_config(["app=mujoco_online_simulation"])
    robot_config = load_robot_config(config["profile"]["robot"])
    command_config = load_teleoperation_command_config(config["profile"]["teleoperation"], robot_config=robot_config)
    binding_config = load_mujoco_robot_binding_config(config["profile"]["robot"], robot_config=robot_config)

    assert config["profile"]["teleoperation"]["max_joint_speed"] == list(command_config.max_joint_speed)
    assert binding_config.model.path == "assets/robots/panda_leap_paxini/mjcf/panda_leap_paxini.xml"


def test_artifact_metadata_preserves_execution_fields_during_split():
    """Keep existing metadata content while typed ownership is separated.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.config import load_config_data, load_retargeting_profile_config, load_robot_config
    from retargeting_apps.pipelines.offline_retargeting import (
        retargeting_profile_config_to_metadata_dict,
        robot_config_to_metadata_dict,
    )

    profile_source = "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
    profile_data = load_config_data(profile_source)
    profile_config = load_retargeting_profile_config(profile_data)
    robot_data = load_config_data(profile_config.robot)
    robot_config = load_robot_config(robot_data)

    profile_metadata = retargeting_profile_config_to_metadata_dict(profile_config, profile_source)
    robot_metadata = robot_config_to_metadata_dict(robot_config, profile_config.robot)

    assert profile_metadata["teleoperation"] == profile_data["teleoperation"]
    assert robot_metadata["simulation_model"] == robot_data["simulation_model"]
