from pathlib import Path

import pytest


def test_robot_configs_load_and_validate_asset_paths():
    from retargeting.config import load_robot_config

    for config_path in [
        "configs/robots/panda_leap_paxini.yaml",
        "configs/robots/panda_shadow.yaml",
    ]:
        config = load_robot_config(config_path)

        link_path = config.model.resolved_link_path()
        if config.model.path_is_symlink:
            assert link_path.is_symlink()
        else:
            assert link_path.is_file()
        assert config.model.resolved_path().is_file()
        assert Path(config.robot_file_path).exists()
        assert len(config.actuated_joints) == len(config.initial_qpos)
        assert config.wrist_frame_name in config.visual_frame_names
        assert config.benchmark.wrist_link_name in config.visual_frame_names
        assert len(config.benchmark.fingertip_link_names) in {4, 5}
        assert config.benchmark.thumb_fingertip.link_name in config.benchmark.fingertip_link_names
        for fingertip_link_name in config.benchmark.fingertip_link_names:
            assert fingertip_link_name in config.visual_frame_names


def test_robot_config_loads_from_composed_mapping():
    from retargeting.config import load_config_data, load_robot_config

    data = load_config_data("configs/robots/panda_leap_paxini.yaml")
    config = load_robot_config(data)

    assert config.name == "panda_leap_paxini"
    assert len(config.actuated_joints) == len(config.initial_qpos)
    assert config.benchmark.human_tip_indices == (4, 8, 12, 16)


def test_robot_benchmark_uses_configured_fingertip_metadata():
    import numpy as np

    from retargeting.config import load_robot_config
    from retargeting.evaluation.robot_metrics import RobotBenchmark

    class FakeRobotModel:
        def __init__(self, poses):
            self.poses = poses

        def get_frame_pose(self, name):
            return self.poses[name]

    class FakeRobotAdaptor:
        def __init__(self, robot_model):
            self.robot_model = robot_model

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    poses = {}
    positions = {
        "wrist": np.array([0.0, 0.0, 0.0]),
        "thumb_tip_center": np.array([0.0, 0.0, 0.0]),
        "finger1_tip_center": np.array([1.0, 0.0, 0.0]),
        "finger2_tip_center": np.array([0.0, 1.0, 0.0]),
        "finger3_tip_center": np.array([0.0, 0.0, 1.0]),
    }
    for frame_name, position in positions.items():
        pose = np.eye(4)
        pose[:3, 3] = position
        poses[frame_name] = pose

    benchmark = RobotBenchmark(
        robot_adaptor=FakeRobotAdaptor(FakeRobotModel(poses)),
        benchmark_config=robot_config.benchmark,
    )
    target_pos = np.zeros((21, 3))
    for fingertip in robot_config.benchmark.fingertips:
        target_pos[fingertip.human_tip_index] = positions[fingertip.link_name]
        target_pos[fingertip.human_direction_base_index] = positions[fingertip.link_name] - np.array([1.0, 0.0, 0.0])

    np.testing.assert_allclose(benchmark.position_error(None, target_pos, 1), np.zeros(4))
    np.testing.assert_allclose(benchmark.orientation_error(None, target_pos, 1), np.zeros(4))
    np.testing.assert_allclose(benchmark.relative_position_error(None, target_pos, 1), np.zeros(3))
    np.testing.assert_allclose(benchmark.relative_position_to_wrist_error(None, target_pos, 1), np.zeros(4))


def test_retargeting_method_and_profile_configs_load_vector_wrist_joint_targets():
    from retargeting.config import ABLATION_OPTION_DESCRIPTIONS, load_retargeting_config, load_retargeting_profile_config

    config = load_retargeting_config("configs/retargeting_methods/vector_wrist_joint.yaml")
    profile = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")

    assert config.type == "VECTOR_WRIST_JOINT"
    assert config.ablation_option == 0
    assert config.ablation_description == ABLATION_OPTION_DESCRIPTIONS[0]
    assert config.optimizer_class == "VectorWristJointOptimizerV2"
    assert config.optimizer_params["huber_delta"] == 0.02
    assert profile.robot == "configs/robots/panda_leap_paxini.yaml"
    assert profile.method == "configs/retargeting_methods/vector_wrist_joint.yaml"
    assert profile.target.wrist_link_name == "wrist"
    assert profile.objective.weights.world_thumb == 10.0


def test_retargeting_config_rejects_unknown_ablation_option():
    from retargeting.config import load_config_data, load_retargeting_config

    config_data = load_config_data("configs/retargeting_methods/vector_wrist_joint.yaml")
    config_data["ablation_option"] = 99

    with pytest.raises(ValueError, match="Unsupported ablation_option"):
        load_retargeting_config(config_data)


def test_detection_source_configs_load_detector_calibration():
    from retargeting.config import load_detection_source_config

    avp_config = load_detection_source_config("configs/detection_sources/avp.yaml")
    rgb_config = load_detection_source_config("configs/detection_sources/rgb.yaml")

    assert avp_config.input_device == "avp"
    assert avp_config.rotation_euler_xyz_deg == (0.0, 0.0, 180.0)
    assert avp_config.translation == (0.7, 0.2, -1.0)
    assert avp_config.use_relative_wrist_alignment is True
    assert rgb_config.input_device == "rgb"
    assert rgb_config.translation == (0.4, 0.0, 0.0)
    assert rgb_config.use_relative_wrist_alignment is False


def test_teleoperation_mode_configs_load_runtime_flags():
    from retargeting.config import load_teleoperation_mode_config

    simulation_config = load_teleoperation_mode_config("configs/teleoperation_modes/simulation.yaml")
    real_world_config = load_teleoperation_mode_config("configs/teleoperation_modes/real_world.yaml")
    virtual_hardware_config = load_teleoperation_mode_config("configs/teleoperation_modes/virtual_hardware.yaml")

    assert simulation_config.name == "simulation"
    assert simulation_config.robot_control.use_hardware is False
    assert simulation_config.output.smooth_output_qpos is False
    assert real_world_config.robot_control.use_hardware is True
    assert real_world_config.robot_control.use_virtual_hardware is False
    assert real_world_config.output.smooth_output_qpos is True
    assert virtual_hardware_config.robot_control.use_hardware is True
    assert virtual_hardware_config.robot_control.use_virtual_hardware is True
    assert virtual_hardware_config.output.smooth_output_qpos is False


def test_default_teleoperation_mode_is_simulation():
    from retargeting.config import load_teleoperation_mode_config

    config = load_teleoperation_mode_config(None)

    assert config.name == "simulation"
    assert config.robot_control.use_hardware is False
    assert config.output.smoothing_alpha == 0.3


def test_retargeting_config_accepts_vector_wrist_joint_class():
    from retargeting.config import load_config_data, load_retargeting_config
    from retargeting.core.optimizers import VectorWristJointOptimizer, get_optimizer_class

    config_data = load_config_data("configs/retargeting_methods/vector_wrist_joint.yaml")
    config_data["optimizer"]["class"] = "VectorWristJointOptimizer"

    config = load_retargeting_config(config_data)

    assert config.optimizer_class == "VectorWristJointOptimizer"
    assert get_optimizer_class(config.optimizer_class) is VectorWristJointOptimizer


def test_solver_configs_load_backend_specific_params():
    from retargeting.config import load_solver_config

    nlopt_config = load_solver_config("configs/solvers/nlopt_slsqp.yaml")
    scipy_config = load_solver_config("configs/solvers/scipy_slsqp.yaml")

    assert nlopt_config.name == "nlopt_slsqp"
    assert nlopt_config.params["ftol_abs"] == 0.00001
    assert nlopt_config.params["maxtime"] == -1
    assert scipy_config.name == "scipy_slsqp"
    assert scipy_config.params["ftol"] == 0.00001
    assert scipy_config.params["maxtime"] == -1


def test_offline_retarget_config_accepts_post_action_overrides():
    from retargeting.main import compose_hydra_base_config

    config = compose_hydra_base_config(
        [
            "app=offline_retarget",
            "post.benchmark.enabled=true",
            "post.benchmark.plot=false",
            "post.visualize.enabled=true",
            "post.visualize.viewer.port=9321",
            "post.visualize.viewer.human_keypoint_size=0.012",
        ]
    )

    assert config["post"]["benchmark"]["enabled"] is True
    assert config["post"]["benchmark"]["plot"] is False
    assert config["post"]["visualize"]["enabled"] is True
    assert config["post"]["visualize"]["viewer"]["port"] == 9321
    assert config["post"]["visualize"]["viewer"]["human_keypoint_size"] == 0.012
    assert config["teleoperation_mode"]["name"] == "simulation"


def test_replay_app_config_loads_defaults():
    from retargeting.main import compose_hydra_base_config

    config = compose_hydra_base_config(["app=replay"])

    assert config["app"]["id"] == "replay"
    assert config["run_name"] is None
    assert "data" not in config
    assert "profile" not in config
    assert "detection_source" not in config
    assert isinstance(config["viewer"]["port"], int)
    assert config["viewer"]["port"] > 0
    assert config["viewer"]["human_keypoint_size"] == 0.005


def test_base_config_selects_each_whitelisted_app():
    """Verify that each app selection composes only its own configuration.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.main import compose_hydra_base_config

    offline_config = compose_hydra_base_config(["app=offline_retarget"])
    replay_config = compose_hydra_base_config(["app=replay"])
    benchmark_config = compose_hydra_base_config(["app=benchmark"])

    assert offline_config["app"]["id"] == "offline_retarget"
    assert "profile" in offline_config
    assert replay_config["app"]["id"] == "replay"
    assert "viewer" in replay_config
    assert "profile" not in replay_config
    assert benchmark_config["app"]["id"] == "benchmark"
    assert "profile" not in benchmark_config


def test_main_dispatcher_rejects_unknown_app_id():
    """Verify the dispatcher accepts only explicitly registered app identifiers.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.main import resolve_app_runner

    with pytest.raises(ValueError, match="Unsupported app.id"):
        resolve_app_runner({"app": {"id": "not_a_real_app"}})


def test_replay_hydra_style_mapping_resolves_runtime_options():
    from retargeting.apps.viser_retargeting_visualize import resolve_replay_options_from_config

    config = {
        "run_name": "example",
        "runtime_root": "/tmp/runtime-root",
        "viewer": {
            "fps": 24.0,
            "port": 8090,
            "no_robot_mesh": True,
            "trail_length": 10,
            "human_keypoint_size": 0.012,
        },
    }

    options = resolve_replay_options_from_config(config)

    assert options["result"] == "/tmp/runtime-root/example/retargeting"
    assert options["port"] == 8090
    assert options["no_robot_mesh"] is True
    assert options["human_keypoint_size"] == 0.012


def test_replay_config_requires_a_saved_artifact():
    """Verify replay refuses to retarget raw input data.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.apps.viser_retargeting_visualize import resolve_replay_options_from_config

    with pytest.raises(ValueError, match="requires run_name"):
        resolve_replay_options_from_config({"viewer": {}})
