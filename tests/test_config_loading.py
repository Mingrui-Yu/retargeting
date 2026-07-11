from pathlib import Path


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
    assert config.hand_type == "leap"
    assert len(config.actuated_joints) == len(config.initial_qpos)
    assert config.benchmark.human_tip_indices == (4, 8, 12, 16)


def test_robot_benchmark_uses_configured_fingertip_metadata():
    import numpy as np

    from retargeting.config import load_robot_config
    from retargeting.robot_benchmark import RobotBenchmark

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


def test_retargeting_config_loads_vector_wrist_joint_targets():
    from retargeting.config import load_retargeting_config

    config = load_retargeting_config("configs/retargeting/vector_wrist_joint.yaml")

    assert config.type == "VECTOR_WRIST_JOINT"
    assert config.optimizer_class == "VectorWristJointOptimizerV2"
    assert config.optimizer_params["huber_delta"] == 0.02
    assert config.targets_for("leap").wrist_link_name == "wrist"
    assert config.targets_for("shadow").wrist_link_name == "ee_link"
    if config.joint_limit_overrides:
        assert config.joint_limit_overrides[0].indices == (9, 10, 13, 14, 17, 18)


def test_retargeting_config_accepts_legacy_vector_wrist_joint_class():
    from retargeting.config import load_config_data, load_retargeting_config
    from retargeting.retarget_optimizer import VectorWristJointOptimizer
    from retargeting.robot_teleoperation import get_optimizer_class

    config_data = load_config_data("configs/retargeting/vector_wrist_joint.yaml")
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
    from retargeting.offline_retarget import compose_hydra_offline_retarget_config

    config = compose_hydra_offline_retarget_config(
        [
            "post.benchmark.enabled=true",
            "post.benchmark.plot=false",
            "post.visualize.enabled=true",
            "post.visualize.viewer.port=9321",
        ]
    )

    assert config["post"]["benchmark"]["enabled"] is True
    assert config["post"]["benchmark"]["plot"] is False
    assert config["post"]["visualize"]["enabled"] is True
    assert config["post"]["visualize"]["viewer"]["port"] == 9321


def test_replay_app_config_loads_defaults():
    from retargeting.config import load_replay_app_config

    config = load_replay_app_config("configs/apps/replay_avp.yaml")

    assert config.data == "tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz"
    assert config.result is None
    assert config.robot == "configs/robots/panda_leap_paxini.yaml"
    assert config.retargeting == "configs/retargeting/vector_wrist_joint.yaml"
    assert isinstance(config.viewer.port, int)
    assert config.viewer.port > 0


def test_replay_hydra_style_mapping_resolves_runtime_options():
    from retargeting.config import load_config_data
    from retargeting.viser_retargeting_visualize import resolve_replay_options_from_config

    config = {
        "data": "tests/fixtures/avp_short_replay.npz",
        "hand_type": "leap",
        "start": 1,
        "end": 3,
        "stride": 2,
        "viewer": {
            "fps": 24.0,
            "port": 8090,
            "no_robot_mesh": True,
            "trail_length": 10,
        },
        "robot": load_config_data("configs/robots/panda_leap_paxini.yaml"),
        "retargeting": load_config_data("configs/retargeting/vector_wrist_joint.yaml"),
    }

    options = resolve_replay_options_from_config(config)

    assert options["data"] == "tests/fixtures/avp_short_replay.npz"
    assert options["hand_type"] == "leap"
    assert options["start"] == 1
    assert options["end"] == 3
    assert options["stride"] == 2
    assert options["port"] == 8090
    assert options["no_robot_mesh"] is True
    assert options["robot_config"].name == "panda_leap_paxini"
    assert options["retargeting_config"].type == "VECTOR_WRIST_JOINT"
