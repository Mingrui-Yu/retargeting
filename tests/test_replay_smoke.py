from pathlib import Path

import pytest


# Phase-0 regression tests for the current research-code layout.
#
# These tests intentionally exercise only headless paths: no ROS, no RViz, no
# camera, no hardware, and no MuJoCo/Open3D viewer. They protect the existing
# repo-relative assets, the promoted AVP replay fixture, Pinocchio model loading,
# RobotAdaptor joint mapping, and one optimizer call before larger refactors.
FIXTURE = Path("tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz")


def _np():
    return pytest.importorskip("numpy")


def _load_fixture():
    np = _np()
    return np.load(FIXTURE)


def _stream_frame(data, frame_idx=0):
    return {
        key.removeprefix("stream_"): value[frame_idx]
        for key, value in data.items()
        if key.startswith("stream_")
    }


def test_phase4_asset_layout_is_available():
    # Phase 4 keeps robot model paths under assets/robots. Root-level URDF
    # symlinks were removed after legacy entrypoints moved to current paths.
    for urdf_path in [
        Path("assets/robots/panda_leap_paxini/urdf/panda_leap_paxini.urdf"),
        Path("assets/robots/panda_shadow/urdf/panda_shadow.urdf"),
    ]:
        assert urdf_path.is_file()

    for removed_legacy_urdf_link in [
        Path("assets/panda_leap_paxini.urdf"),
        Path("assets/panda_shadow.urdf"),
    ]:
        assert not removed_legacy_urdf_link.exists()


def test_avp_teleop_replay_fixture_shape_and_expected_qpos():
    # The fixture is a promoted offline AVP teleoperation replay.
    # It pins the stream tensor shapes and the first recorded retargeted qpos.
    np = _np()
    data = _load_fixture()

    assert data["stream_right_wrist"].shape == (760, 1, 4, 4)
    assert data["stream_right_fingers"].shape == (760, 25, 4, 4)
    assert data["retarget_qpos"].shape == (760, 23)
    assert np.isfinite(data["retarget_qpos"]).all()

    expected_first_qpos = np.array(
        [
            0.00875867996364832,
            -0.7382485713556766,
            -0.07347981631755829,
            -2.3194164472804544,
            0.04033493250608444,
            1.601985483565998,
            2.1532108026729104,
            -0.0005586452898569405,
            0.11768739268183707,
            0.11215394333004951,
            0.10800079226493835,
            -0.0007274928502738476,
            0.11521222040057182,
            0.11009211063385009,
            0.1064139787852764,
            -0.0007787310751155019,
            0.11381804168224334,
            0.10910807147622108,
            0.10574997931718826,
            0.03051988035440445,
            0.005901265423744917,
            0.09606269717216491,
            0.09836528234183788,
        ]
    )
    np.testing.assert_allclose(data["retarget_qpos"][0], expected_first_qpos, rtol=0, atol=1e-12)


def test_avp_common_decoder_is_headless():
    # Exercise only the shared AVP decoder for an already-recorded stream frame.
    # Importing and decoding archived data must not require the live client.
    _np()
    pytest.importorskip("scipy")

    from teleoperation.inputs.avp import decode_avp_sample

    data = _load_fixture()
    stream = _stream_frame(data, frame_idx=0)
    sample = decode_avp_sample(stream, source_index=0)

    assert sample.keypoints_wrist.shape == (21, 3)
    assert sample.wrist_pose_sensor.shape == (4, 4)
    assert sample.source_index == 0


def test_robot_pinocchio_loads_current_urdf_headless():
    # Verify that the configured Panda+Leap URDF loads in Pinocchio and exposes
    # the frame names required by the current retargeting objective.
    pytest.importorskip("pinocchio")

    from retargeting.config import load_robot_config
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    robot = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)

    assert robot.dof > 0
    for frame_name in [
        "wrist",
        "thumb_tip_center",
        "finger1_tip_center",
        "finger2_tip_center",
        "finger3_tip_center",
        "thumb_tip_center_lower",
    ]:
        assert frame_name in robot.frame_names


def test_robot_adaptor_forward_backward_qpos_round_trip():
    # Protect the current actuated-joint mapping. Later config migration should
    # preserve this round-trip behavior for Panda+Leap.
    np = _np()
    pytest.importorskip("pinocchio")

    from retargeting.config import load_robot_config
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    adaptor = RobotAdaptor(
        robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )

    qpos_doa = np.linspace(-0.01, 0.01, adaptor.doa)
    qpos_dof = adaptor.forward_qpos(qpos_doa)

    assert qpos_dof.shape == (robot_model.dof,)
    np.testing.assert_allclose(adaptor.backward_qpos(qpos_dof), qpos_doa)

    jacobian = np.zeros((2, 6, robot_model.dof))
    assert adaptor.backward_jacobian(jacobian).shape == (2, 6, adaptor.doa)


def test_panda_leap_profile_joint_limit_override_reaches_optimizer_and_solver():
    """Verify configured Leap lower bounds are applied to optimizer and solver state.

    Args:
        None.

    Returns:
        None.
    """
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("nlopt")
    pytest.importorskip("torch")

    from retargeting.config import load_retargeting_config, load_retargeting_profile_config, load_robot_config
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio
    from retargeting.core.retargeter import Retargeter

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    profile_config = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    method_config = load_retargeting_config(profile_config.method)
    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    adaptor = RobotAdaptor(robot_model, actuated_joints_name=list(robot_config.actuated_joints))
    retargeter = Retargeter(adaptor, robot_config, profile_config, method_config)

    expected_indices = np.asarray([9, 10, 13, 14, 17, 18])
    assert profile_config.joint_limit_overrides[0].indices == tuple(expected_indices)
    assert profile_config.joint_limit_overrides[0].lower == 0.0
    np.testing.assert_allclose(retargeter.optimizer.joint_limits[expected_indices, 0], 0.0)
    np.testing.assert_allclose(retargeter.optimizer.opt._state.lower_bounds[expected_indices], -0.001)


def test_vector_wrist_joint_optimizer_single_frame_smoke():
    # Minimal optimizer smoke test: construct the current VectorWristJoint
    # objective and run one retarget() call. This checks that the optimization
    # chain is executable, not that the generated pose is semantically good.
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("nlopt")
    pytest.importorskip("torch")

    from retargeting.core.optimizers import VectorWristJointOptimizer
    from retargeting.config import load_retargeting_config, load_retargeting_profile_config, load_robot_config, load_solver_config
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    profile_config = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    retargeting_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config("configs/solvers/nlopt_slsqp.yaml")
    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    adaptor = RobotAdaptor(
        robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )

    target_config = profile_config.target
    target_link_pairs = target_config.link_pairs
    optimizer = VectorWristJointOptimizer(
        robot_adaptor=adaptor,
        targets={
            "origin_links_name": [pair[0] for pair in target_link_pairs],
            "task_links_name": [pair[1] for pair in target_link_pairs],
            "wrist_link_name": target_config.wrist_link_name,
        },
        params={**retargeting_config.optimizer_params, "solver_params": {**solver_config.params, "maxtime": 0.01}},
        joint_limit_overrides=[
            {
                "indices": list(override.indices),
                "lower": override.lower,
                "upper": override.upper,
            }
            for override in profile_config.joint_limit_overrides
        ],
    )

    qpos_init = _load_fixture()["retarget_qpos"][0].copy()
    ref_values = {
        "links_vec": np.zeros((len(target_link_pairs), 3)),
        "wrist_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "qpos_doa": qpos_init.copy(),
        "qpos_doa_last": qpos_init.copy(),
        "weights": {
            "links_vec": np.ones(len(target_link_pairs)),
            "wrist_rot": 0.0,
            "joint_pos": np.zeros(adaptor.doa),
            "joint_vel": np.zeros(adaptor.doa),
        },
    }

    qpos = optimizer.retarget(ref_values)

    assert qpos.shape == (adaptor.doa,)
    assert np.isfinite(qpos).all()


def test_vector_wrist_joint_optimizer_scipy_slsqp_single_frame_smoke():
    # The scipy backend should run through the same retarget() API while using
    # scipy's SLSQP adapter instead of nlopt.
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("scipy")
    pytest.importorskip("torch")

    from retargeting.config import load_retargeting_config, load_retargeting_profile_config, load_robot_config, load_solver_config
    from retargeting.core.optimizers import VectorWristJointOptimizer
    from retargeting.core.kinematics.adaptor import RobotAdaptor
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    profile_config = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    retargeting_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config("configs/solvers/scipy_slsqp.yaml")
    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    adaptor = RobotAdaptor(
        robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )

    target_config = profile_config.target
    target_link_pairs = target_config.link_pairs
    optimizer = VectorWristJointOptimizer(
        robot_adaptor=adaptor,
        targets={
            "origin_links_name": [pair[0] for pair in target_link_pairs],
            "task_links_name": [pair[1] for pair in target_link_pairs],
            "wrist_link_name": target_config.wrist_link_name,
        },
        params={**retargeting_config.optimizer_params, "solver_params": {**solver_config.params, "maxtime": 0.01}},
        joint_limit_overrides=[
            {
                "indices": list(override.indices),
                "lower": override.lower,
                "upper": override.upper,
            }
            for override in profile_config.joint_limit_overrides
        ],
        solver="scipy_slsqp",
    )

    qpos_init = _load_fixture()["retarget_qpos"][0].copy()
    ref_values = {
        "links_vec": np.zeros((len(target_link_pairs), 3)),
        "wrist_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "qpos_doa": qpos_init.copy(),
        "qpos_doa_last": qpos_init.copy(),
        "weights": {
            "links_vec": np.ones(len(target_link_pairs)),
            "wrist_rot": 0.0,
            "joint_pos": np.zeros(adaptor.doa),
            "joint_vel": np.zeros(adaptor.doa),
        },
    }

    qpos = optimizer.retarget(ref_values)

    assert qpos.shape == (adaptor.doa,)
    assert np.isfinite(qpos).all()
