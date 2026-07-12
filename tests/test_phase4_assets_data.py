from pathlib import Path


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

    method_config = load_retargeting_config("configs/retargeting_methods/vector_wrist_joint.yaml")
    assert method_config.type == "VECTOR_WRIST_JOINT"

    for config_path in Path("configs/retargeting_profiles").glob("*.yaml"):
        profile_config = load_retargeting_profile_config(config_path)
        robot_config = load_robot_config(profile_config.robot)
        retargeting_runtime_config = profile_config.retargeting
        teleoperation_command_config = profile_config.teleoperation
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
    from retargeting.config import load_detection_source_config

    for config_path in Path("configs/detection_sources").glob("*.yaml"):
        detection_source_config = load_detection_source_config(config_path)

        assert detection_source_config.input_device in {"rgb", "avp"}
        assert len(detection_source_config.rotation_euler_xyz_deg) == 3
        assert len(detection_source_config.translation) == 3


def test_teleoperation_mode_configs_carry_runtime_flags():
    from retargeting.config import load_teleoperation_mode_config

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


def test_robot_asset_component_symlinks_target_shared_meshes():
    expected_targets = {
        Path("assets/robots/panda_leap_paxini/urdf/panda"): Path("assets/meshes/panda"),
        Path("assets/robots/panda_leap_paxini/urdf/leap_hand"): Path("assets/meshes/leap_hand"),
        Path("assets/robots/panda_shadow/urdf/panda"): Path("assets/meshes/panda"),
        Path("assets/robots/panda_shadow/urdf/shadow_hand"): Path("assets/meshes/shadow_hand"),
    }

    for link_path, target_path in expected_targets.items():
        assert link_path.is_symlink()
        assert link_path.resolve() == target_path.resolve()


def test_replay_app_uses_promoted_fixture_not_data_output_tree():
    from retargeting.config import load_replay_app_config

    app_config = load_replay_app_config("configs/apps/replay_avp.yaml")

    assert app_config.data == "tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz"
    assert app_config.detection_source == "configs/detection_sources/avp.yaml"
    assert Path(app_config.data).is_file()
    assert not app_config.data.startswith("data/")


def test_gitignore_covers_phase4_outputs():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "outputs/",
    ]:
        assert pattern in gitignore
