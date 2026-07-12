from pathlib import Path

import pytest


FIXTURE = Path("tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz")


def _np():
    return pytest.importorskip("numpy")


def test_offline_replay_rebuilds_stream_frames_without_avp_stream():
    _np()
    pytest.importorskip("scipy")

    from retargeting.offline_replay import load_offline_replay
    from retargeting.avp_detector import parse_avp_stream_frame

    replay = load_offline_replay(FIXTURE)

    assert replay.n_frames == 760
    assert replay.retarget_qpos.shape == (760, 23)
    assert replay.streams[0]["right_wrist"].shape == (1, 4, 4)

    num_box, hand_kps, _, wrist_pose = parse_avp_stream_frame(replay.streams[0])

    assert num_box == 1
    assert hand_kps.shape == (21, 3)
    assert wrist_pose.shape == (4, 4)


def test_retarget_replay_recomputes_human_and_robot_geometry_headless():
    np = _np()
    pytest.importorskip("pinocchio")
    pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    pytest.importorskip("scipy")

    from retargeting.retargeting_replay import build_retarget_replay_frames

    context, frames = build_retarget_replay_frames(
        data_file=str(FIXTURE),
        retargeting_profile_config_path="configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml",
        start=0,
        end=1,
        stride=1,
    )

    assert context.robot_name == "panda_leap_paxini"
    assert len(frames) == 2

    frame = frames[0]
    assert frame.frame_idx == 0
    assert frame.hand_keypoints_wrist.shape == (21, 3)
    assert frame.hand_keypoints_world.shape == (21, 3)
    assert frame.wrist_pose_world.shape == (4, 4)
    assert frame.qpos.shape == (23,)
    assert frame.err is not None
    assert "optimization_time" in frame.err
    assert np.isfinite(frame.hand_keypoints_world).all()
    assert np.isfinite(frame.qpos).all()

    for frame_name in [
        "wrist",
        "thumb_tip_center",
        "finger1_tip_center",
        "finger2_tip_center",
        "finger3_tip_center",
    ]:
        assert frame_name in frame.robot_frame_poses
        assert frame.robot_frame_poses[frame_name].shape == (4, 4)
        assert np.isfinite(frame.robot_frame_poses[frame_name]).all()
