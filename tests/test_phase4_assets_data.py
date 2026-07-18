import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml


def test_asset_resolver_handles_current_robot_asset_paths():
    from retargeting.config import resolve_asset_path

    for robot_asset_path in [
        "assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf",
        "assets/robots/panda_shadow/urdf/panda_shadow.urdf",
    ]:
        assert resolve_asset_path(robot_asset_path).is_file()

    for removed_legacy_path in [
        Path("assets/panda_leap_paxini.urdf"),
        Path("assets/panda_shadow.urdf"),
    ]:
        assert not removed_legacy_path.exists()


def test_robot_configs_use_phase4_asset_paths():
    from retargeting.config import load_robot_config

    for config_path in Path("configs/robots").glob("*.yaml"):
        robot_config = load_robot_config(config_path)

        assert robot_config.model.path.startswith("assets/robots/")
        assert not robot_config.model.path_is_symlink
        assert Path(robot_config.robot_file_path).is_file()


def test_retargeting_profiles_carry_robot_method_parameters():
    from retargeting.config import load_retargeting_config, load_retargeting_profile_config, load_robot_config
    from teleoperation.config import load_teleoperation_command_config

    method_config = load_retargeting_config("configs/retargeting_methods/vector_wrist_joint.yaml")
    assert method_config.type == "VECTOR_WRIST_JOINT"

    for config_path in Path("configs/retargeting_profiles").glob("*.yaml"):
        profile_config = load_retargeting_profile_config(config_path)
        robot_config = load_robot_config(profile_config.robot)
        retargeting_runtime_config = profile_config.retargeting
        teleoperation_command_config = load_teleoperation_command_config(config_path, robot_config=robot_config)
        qpos_size = len(robot_config.initial_qpos)

        assert profile_config.method == "configs/retargeting_methods/vector_wrist_joint.yaml"
        assert profile_config.objective.pinch_transition_threshold == 0.1
        assert profile_config.objective.weights.world_thumb == 10.0
        assert 0 < retargeting_runtime_config.arm_dof <= qpos_size
        assert len(retargeting_runtime_config.joint_position_weights) == qpos_size
        assert len(retargeting_runtime_config.joint_velocity_weights) == qpos_size
        assert len(teleoperation_command_config.max_joint_speed) == qpos_size
        assert len(profile_config.target.link_pairs) == 3 * len(robot_config.benchmark.fingertips)


def test_detection_source_configs_carry_detector_world_calibration():
    from teleoperation.config import load_detection_source_config

    for config_path in Path("configs/detection_sources").glob("*.yaml"):
        detection_source_config = load_detection_source_config(config_path)

        assert detection_source_config.input_device in {"rgb", "avp"}
        assert len(detection_source_config.rotation_euler_xyz_deg) == 3
        assert len(detection_source_config.translation) == 3


def test_teleoperation_mode_configs_carry_runtime_flags():
    from teleoperation.config import load_teleoperation_mode_config

    for config_path in Path("configs/teleoperation_modes").glob("*.yaml"):
        mode_config = load_teleoperation_mode_config(config_path)

        assert mode_config.name == config_path.stem
        assert 0.0 <= mode_config.output.smoothing_alpha <= 1.0


def test_robot_meshes_are_shared_components_under_assets_meshes():
    for component_dir in [
        Path("assets/meshes/panda/meshes"),
        Path("assets/meshes/leap_hand/meshes"),
        Path("assets/meshes/shadow_hand/meshes"),
    ]:
        assert component_dir.is_dir()

    for legacy_root_dir in [
        Path("assets/panda"),
        Path("assets/leap_hand"),
    ]:
        assert not legacy_root_dir.exists()


def test_unused_root_level_scene_xml_is_removed():
    assert not Path("assets/scene.xml").exists()


def test_panda_shadow_component_symlinks_target_shared_meshes():
    expected_targets = {
        Path("assets/robots/panda_shadow/urdf/panda"): Path("assets/meshes/panda"),
        Path("assets/robots/panda_shadow/urdf/shadow_hand"): Path("assets/meshes/shadow_hand"),
    }

    for link_path, target_path in expected_targets.items():
        assert link_path.is_symlink()
        assert link_path.resolve() == target_path.resolve()


def test_panda_leap_paxini_portable_bundle_is_self_contained():
    """Validate the public bundle manifest and every description resource path.

    Args:
        None.

    Returns:
        None.
    """
    assets_dir = Path("assets/robots/panda_leap_paxini")
    manifest_path = assets_dir / "manifest.yaml"
    urdf_path = assets_dir / "urdf/panda_leap_paxini.urdf"
    mjcf_path = assets_dir / "mjcf/panda_leap_paxini.xml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["robot"] == "panda_leap_paxini"
    assert manifest["format"] == "portable_robot_assets"
    assert manifest["entrypoints"] == {
        "urdf": "urdf/panda_leap_paxini.urdf",
        "mjcf": "mjcf/panda_leap_paxini.xml",
    }
    assert manifest["meshes"] == "meshes"
    assert not [path for path in assets_dir.rglob("*") if path.is_symlink()]

    # Both public descriptions must use only relative files contained by this bundle.
    path_attributes = ("filename", "file", "meshdir", "texturedir")
    referenced_urdf_meshes = set()
    for xml_path in (urdf_path, mjcf_path):
        for element in ET.parse(xml_path).getroot().iter():
            for attribute in path_attributes:
                value = element.get(attribute)
                if not value:
                    continue
                assert not Path(value).is_absolute()
                assert not value.startswith("package://")
                resolved_path = (xml_path.parent / value).resolve()
                resolved_path.relative_to(assets_dir.resolve())
                assert resolved_path.is_file()
                if xml_path == urdf_path and element.tag == "mesh" and attribute == "filename":
                    referenced_urdf_meshes.add(resolved_path)

    bundle_meshes = {path.resolve() for path in (assets_dir / "meshes").rglob("*") if path.is_file()}
    assert referenced_urdf_meshes == bundle_meshes


def test_panda_leap_paxini_portable_mjcf_compiles_headlessly():
    """Compile the portable MJCF without opening a MuJoCo viewer.

    Args:
        None.

    Returns:
        None.
    """
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(
        str(Path("assets/robots/panda_leap_paxini/mjcf/panda_leap_paxini.xml"))
    )

    assert (model.nbody, model.njnt, model.ngeom, model.nu, model.nsensor, model.nkey) == (44, 23, 66, 23, 0, 1)


def test_replay_app_requires_a_saved_artifact_not_raw_input_data():
    from retargeting_apps.main import compose_hydra_base_config

    config = compose_hydra_base_config(["app=replay"])
    assert config["run_name"] is None
    assert "data" not in config
    assert "detection_source" not in config


def test_gitignore_covers_phase4_outputs():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "outputs/",
    ]:
        assert pattern in gitignore
