from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from retargeting.config import (
    RobotBenchmarkConfig,
    RetargetingConfig,
    RobotConfig,
    default_robot_config_path,
    load_retargeting_config,
    load_robot_config,
    to_plain_config_data,
)
from retargeting.offline_replay import OfflineReplay, iter_frame_indices, load_offline_replay
from retargeting.robot_adaptor import RobotAdaptor
from retargeting.robot_pinocchio import RobotPinocchio
from retargeting.trajectory_result import RetargetingRunMetadata, RetargetingTrajectory
from scipy.spatial.transform import Rotation as sciR
from retargeting.utils.utils_calc import posRotMat2Isometry3d, transformPositions
from retargeting.vision_pro_detector import parse_vision_pro_stream_frame


LEAP_ACTUATED_JOINTS_NAME = [f"panda_joint{i + 1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
LEAP_VISUAL_FRAME_NAMES = [
    "wrist",
    "thumb_tip_center",
    "finger1_tip_center",
    "finger2_tip_center",
    "finger3_tip_center",
    "thumb_tip_center_lower",
    "finger1_tip_center_lower",
    "finger2_tip_center_lower",
    "finger3_tip_center_lower",
]

SHADOW_ACTUATED_JOINTS_NAME = [f"panda_joint{i + 1}" for i in range(7)] + [
    "WRJ2",
    "WRJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
]
SHADOW_VISUAL_FRAME_NAMES = [
    "ee_link",
    "thtip",
    "fftip",
    "mftip",
    "rftip",
    "lftip",
    "thdistal",
    "ffdistal",
    "mfdistal",
    "rfdistal",
    "lfdistal",
]


@dataclass
class RobotReplayContext:
    hand_type: str
    robot_file_path: str
    robot_model: RobotPinocchio
    robot_adaptor: RobotAdaptor
    actuated_joints_name: List[str]
    init_joint_pos: np.ndarray
    visual_frame_names: List[str]
    wrist_frame_name: str
    human_hand_scale: float
    benchmark_config: RobotBenchmarkConfig


@dataclass
class RetargetReplayFrame:
    frame_idx: int
    hand_keypoints_wrist: np.ndarray
    hand_keypoints_world: np.ndarray
    wrist_pose_world: np.ndarray
    qpos: Optional[np.ndarray]
    robot_frame_poses: Dict[str, np.ndarray]
    err: Optional[Dict[str, object]] = None


def get_default_init_joint_pos(hand_type: str) -> np.ndarray:
    return np.asarray(load_robot_config(default_robot_config_path(hand_type)).initial_qpos, dtype=float)


def robot_config_to_metadata_dict(robot_config: RobotConfig) -> dict:
    """Convert a robot config object to metadata-safe plain data.

    Args:
        robot_config: Typed robot config.

    Returns:
        Plain dictionary that can be embedded in metadata.yaml.
    """
    return to_plain_config_data(asdict(robot_config))


def retargeting_config_to_metadata_dict(retargeting_config: RetargetingConfig) -> dict:
    """Convert a retargeting config object to metadata-safe plain data.

    Args:
        retargeting_config: Typed retargeting config.

    Returns:
        Plain dictionary that can be embedded in metadata.yaml.
    """
    return to_plain_config_data(
        {
            "type": retargeting_config.type,
            "setting_id": retargeting_config.setting_id,
            "ablation_option": retargeting_config.ablation_option,
            "optimizer": {
                "class": retargeting_config.optimizer_class,
                "params": retargeting_config.optimizer_params,
            },
            "targets": {
                hand_type: asdict(target) for hand_type, target in retargeting_config.targets.items()
            },
            "joint_limit_overrides": [
                asdict(override) for override in retargeting_config.joint_limit_overrides
            ],
        }
    )


def pose_from_avp_world_to_robot_world(pose_in_avp_world: np.ndarray) -> np.ndarray:
    transform = posRotMat2Isometry3d(
        pos=[0, 0, 0], rot_mat=sciR.from_euler("xyz", [0, 0, 180], degrees=True).as_matrix()
    )
    pose_in_world = transform @ pose_in_avp_world
    pose_in_world[:3, 3] += [0.7, 0.2, -1.0]
    return pose_in_world


def create_robot_replay_context_from_config(robot_config: RobotConfig) -> RobotReplayContext:
    robot_model = RobotPinocchio(robot_file_path=robot_config.robot_file_path, robot_file_type=robot_config.model.type)
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )
    return RobotReplayContext(
        hand_type=robot_config.hand_type,
        robot_file_path=robot_config.robot_file_path,
        robot_model=robot_model,
        robot_adaptor=robot_adaptor,
        actuated_joints_name=list(robot_config.actuated_joints),
        init_joint_pos=np.asarray(robot_config.initial_qpos, dtype=float),
        visual_frame_names=list(robot_config.visual_frame_names),
        wrist_frame_name=robot_config.wrist_frame_name,
        human_hand_scale=robot_config.human_hand_scale,
        benchmark_config=robot_config.benchmark,
    )


def create_robot_replay_context_from_metadata(metadata: RetargetingRunMetadata) -> RobotReplayContext:
    """Create a replay context from saved trajectory metadata.

    Args:
        metadata: Metadata loaded from a saved retargeting artifact.

    Returns:
        Robot replay context reconstructed from the embedded robot config.
    """
    return create_robot_replay_context_from_config(load_robot_config(metadata.robot_config))


def create_robot_replay_context(hand_type: str = "leap") -> RobotReplayContext:
    return create_robot_replay_context_from_config(load_robot_config(default_robot_config_path(hand_type)))


def compute_robot_frame_poses(
    context: RobotReplayContext,
    qpos: Optional[np.ndarray],
    frame_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    if qpos is None:
        return {}
    frame_names = context.visual_frame_names if frame_names is None else frame_names
    qpos_dof = context.robot_adaptor.forward_qpos(qpos)
    context.robot_model.compute_forward_kinematics(qpos_dof)
    return {frame_name: context.robot_model.get_frame_pose(frame_name).copy() for frame_name in frame_names}


def get_initial_alignment_poses(
    replay: OfflineReplay,
    context: RobotReplayContext,
    first_frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    init_robot_wrist_pose = context.robot_model.get_frame_pose(
        context.wrist_frame_name,
        qpos=context.robot_adaptor.forward_qpos(context.init_joint_pos),
    )
    _, _, _, wrist_pose_in_avp_world = parse_vision_pro_stream_frame(replay.streams[first_frame_idx])
    init_avp_wrist_pose = pose_from_avp_world_to_robot_world(wrist_pose_in_avp_world)
    return init_robot_wrist_pose, init_avp_wrist_pose


def create_retargeter(context: RobotReplayContext, retargeting_config: Optional[RetargetingConfig] = None):
    from retargeting.robot_teleoperation import RobotTeleoperation

    return RobotTeleoperation(
        hand_type=context.hand_type,
        robot_adaptor=context.robot_adaptor,
        robot_control=None,
        qpos_init=context.init_joint_pos.copy(),
        input_device="vision_pro",
        mujoco_vis=False,
        use_real_hardware=False,
        retargeting_config=retargeting_config,
        human_hand_scale=context.human_hand_scale,
        benchmark_config=context.benchmark_config,
    )


def frames_to_trajectory(frames: List[RetargetReplayFrame]) -> RetargetingTrajectory:
    """Convert in-memory replay frames to a persistent trajectory representation.

    Args:
        frames: Retargeted replay frames with qpos and geometry.

    Returns:
        Compact trajectory arrays suitable for saving to result.npz.
    """
    frames = [frame for frame in frames if frame.qpos is not None]
    if not frames:
        raise ValueError("No frames with retargeted qpos are available.")

    robot_frame_names = sorted({name for frame in frames for name in frame.robot_frame_poses})
    error_names = sorted({name for frame in frames if frame.err is not None for name in frame.err})
    robot_frame_poses = {
        frame_name: np.stack([frame.robot_frame_poses[frame_name] for frame in frames], axis=0)
        for frame_name in robot_frame_names
    }
    errors = {}
    for error_name in error_names:
        values = [frame.err[error_name] for frame in frames if frame.err is not None and error_name in frame.err]
        if len(values) == len(frames):
            errors[error_name] = np.asarray(values)

    return RetargetingTrajectory(
        frame_indices=np.asarray([frame.frame_idx for frame in frames], dtype=np.int64),
        retarget_qpos=np.stack([frame.qpos for frame in frames], axis=0),
        hand_keypoints_wrist=np.stack([frame.hand_keypoints_wrist for frame in frames], axis=0),
        hand_keypoints_world=np.stack([frame.hand_keypoints_world for frame in frames], axis=0),
        wrist_pose_world=np.stack([frame.wrist_pose_world for frame in frames], axis=0),
        robot_frame_poses=robot_frame_poses,
        errors=errors,
    )


def trajectory_to_replay_frames(
    context: RobotReplayContext,
    trajectory: RetargetingTrajectory,
) -> List[RetargetReplayFrame]:
    """Convert a saved trajectory representation to viewer replay frames.

    Args:
        context: Robot replay context used to compute missing robot frame poses.
        trajectory: Loaded or freshly generated retargeting trajectory.

    Returns:
        Replay frames suitable for viser rendering.
    """
    trajectory.validate()
    frames: List[RetargetReplayFrame] = []
    for idx, frame_idx in enumerate(trajectory.frame_indices):
        robot_frame_poses = {
            frame_name: poses[idx].copy() for frame_name, poses in trajectory.robot_frame_poses.items()
        }
        if not robot_frame_poses:
            robot_frame_poses = compute_robot_frame_poses(context, trajectory.retarget_qpos[idx])
        errors = {name: values[idx] for name, values in trajectory.errors.items()} or None
        frames.append(
            RetargetReplayFrame(
                frame_idx=int(frame_idx),
                hand_keypoints_wrist=trajectory.hand_keypoints_wrist[idx].copy(),
                hand_keypoints_world=trajectory.hand_keypoints_world[idx].copy(),
                wrist_pose_world=trajectory.wrist_pose_world[idx].copy(),
                qpos=trajectory.retarget_qpos[idx].copy(),
                robot_frame_poses=robot_frame_poses,
                err=errors,
            )
        )
    return frames


def run_offline_retargeting(
    data_file: str,
    hand_type: str = "leap",
    start: int = 0,
    end: int = -1,
    stride: int = 1,
    robot_config: RobotConfig | None = None,
    retargeting_config: RetargetingConfig | None = None,
    robot_config_path: str | Path | None = None,
    retargeting_config_path: str | Path | None = None,
) -> tuple[RobotReplayContext, RetargetingTrajectory, RetargetingRunMetadata]:
    """Run offline retargeting and return a persistent trajectory artifact in memory.

    Args:
        data_file: Offline AVP replay file.
        hand_type: Default hand type used when no robot config is provided.
        start: First input frame index.
        end: Last input frame index, inclusive; negative means the final frame.
        stride: Frame stride.
        robot_config: Optional already-loaded robot config.
        retargeting_config: Optional already-loaded retargeting config.
        robot_config_path: Optional robot config path.
        retargeting_config_path: Optional retargeting config path.

    Returns:
        Tuple of replay context, trajectory arrays, and run metadata.
    """
    replay = load_offline_replay(data_file)
    frame_indices = list(iter_frame_indices(replay.n_frames, start=start, end=end, stride=stride))
    if not frame_indices:
        raise ValueError("No replay frames selected.")

    if robot_config is None:
        robot_config = (
            load_robot_config(robot_config_path)
            if robot_config_path is not None
            else load_robot_config(default_robot_config_path(hand_type))
        )
    if retargeting_config is None:
        retargeting_config = (
            load_retargeting_config(retargeting_config_path) if retargeting_config_path is not None else None
        )

    context = create_robot_replay_context_from_config(robot_config)
    init_robot_wrist_pose, init_avp_wrist_pose = get_initial_alignment_poses(replay, context, frame_indices[0])

    retargeter = create_retargeter(context, retargeting_config=retargeting_config)
    retargeter.set_robot_init_wrist_pose(init_robot_wrist_pose)
    retargeter.set_avp_init_wrist_pose(init_avp_wrist_pose)

    frames: List[RetargetReplayFrame] = []
    for frame_idx in frame_indices:
        hand_kps_wrist, wrist_pose_world, qpos, err = retargeter.vision_pro_retarget(stream=replay.streams[frame_idx])
        if hand_kps_wrist is None:
            continue
        hand_kps_world = transformPositions(hand_kps_wrist, target_frame_pose_inv=wrist_pose_world)

        frames.append(
            RetargetReplayFrame(
                frame_idx=frame_idx,
                hand_keypoints_wrist=hand_kps_wrist,
                hand_keypoints_world=hand_kps_world,
                wrist_pose_world=wrist_pose_world,
                qpos=qpos,
                robot_frame_poses=compute_robot_frame_poses(context, qpos),
                err=err,
            )
        )

    trajectory = frames_to_trajectory(frames)
    metadata = RetargetingRunMetadata(
        source_data=str(data_file),
        robot_config=robot_config_to_metadata_dict(robot_config),
        retargeting_config=None
        if retargeting_config is None
        else retargeting_config_to_metadata_dict(retargeting_config),
        start=start,
        end=end,
        stride=stride,
        num_frames=trajectory.n_frames,
        qpos_dim=trajectory.qpos_dim,
    )
    return context, trajectory, metadata


def build_retarget_replay_frames(
    data_file: str,
    hand_type: str = "leap",
    start: int = 0,
    end: int = -1,
    stride: int = 1,
    robot_config: RobotConfig | None = None,
    retargeting_config: RetargetingConfig | None = None,
    robot_config_path: str | Path | None = None,
    retargeting_config_path: str | Path | None = None,
) -> tuple[RobotReplayContext, List[RetargetReplayFrame]]:
    """Build viewer replay frames by running offline retargeting.

    Args:
        data_file: Offline AVP replay file.
        hand_type: Default hand type used when no robot config is provided.
        start: First input frame index.
        end: Last input frame index, inclusive; negative means the final frame.
        stride: Frame stride.
        robot_config: Optional already-loaded robot config.
        retargeting_config: Optional already-loaded retargeting config.
        robot_config_path: Optional robot config path.
        retargeting_config_path: Optional retargeting config path.

    Returns:
        Tuple of robot replay context and replay frames suitable for visualization.
    """
    context, trajectory, _ = run_offline_retargeting(
        data_file=data_file,
        hand_type=hand_type,
        start=start,
        end=end,
        stride=stride,
        robot_config=robot_config,
        retargeting_config=retargeting_config,
        robot_config_path=robot_config_path,
        retargeting_config_path=retargeting_config_path,
    )
    frames = trajectory_to_replay_frames(context, trajectory)
    return context, frames
