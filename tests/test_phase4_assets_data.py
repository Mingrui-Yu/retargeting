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
    assert Path(app_config.data).is_file()
    assert not app_config.data.startswith("data/")


def test_gitignore_covers_phase4_outputs():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "outputs/",
    ]:
        assert pattern in gitignore
