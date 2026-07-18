from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from retargeting.config import (
    RobotBenchmarkConfig,
    RetargetingConfig,
    RetargetingProfileConfig,
    RobotConfig,
    RetargetingRuntimeConfig,
    SolverConfig,
    default_solver_config,
    load_config_data,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)
from retargeting_apps.config import to_plain_config_data
from retargeting_apps.composition import build_batch_retarget_flow
from retargeting_apps.artifacts.trajectory import RetargetingRunMetadata, RetargetingTrajectory
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.core.kinematics.pinocchio_model import RobotPinocchio
from mr_utils.utils_calc import transformPositions
from teleoperation.config import (
    DetectionSourceConfig,
    TeleoperationModeConfig,
    load_detection_source_config,
    load_teleoperation_mode_config,
)


DEFAULT_RETARGETING_PROFILE_CONFIG_PATH = "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
DEFAULT_DETECTION_SOURCE_CONFIG_PATH = "configs/inputs/avp.yaml"

@dataclass
class RobotReplayContext:
    robot_config: RobotConfig
    robot_name: str
    robot_file_path: str
    robot_model: RobotPinocchio
    robot_adaptor: RobotAdaptor
    actuated_joints_name: List[str]
    init_joint_pos: np.ndarray
    visual_frame_names: List[str]
    wrist_frame_name: str
    human_hand_scale: float
    benchmark_config: RobotBenchmarkConfig
    retargeting_runtime_config: RetargetingRuntimeConfig
    detection_source_config: DetectionSourceConfig
    retargeting_profile_config: RetargetingProfileConfig


@dataclass
class RetargetReplayFrame:
    frame_idx: int
    hand_keypoints_wrist: np.ndarray
    hand_keypoints_world: np.ndarray
    wrist_pose_world: np.ndarray
    qpos: Optional[np.ndarray]
    robot_frame_poses: Dict[str, np.ndarray]
    err: Optional[Dict[str, object]] = None


def _load_metadata_source(source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Load optional source data used to preserve artifact compatibility.

    Args:
        source: YAML path, composed config mapping, or None.

    Returns:
        Plain source mapping, or an empty mapping when no source is available.
    """
    if source is None:
        return {}
    data = (
        to_plain_config_data(source)
        if isinstance(source, Mapping) or hasattr(source, "items")
        else load_config_data(source)
    )
    if not isinstance(data, dict):
        raise ValueError("Expected metadata config source to be a mapping.")
    return data


def robot_config_to_metadata_dict(
    robot_config: RobotConfig,
    source: str | Path | Mapping[str, Any] | None = None,
) -> dict:
    """Convert a robot config object to metadata-safe plain data.

    Args:
        robot_config: Typed robot config.
        source: Optional original robot config containing execution-layer fields.

    Returns:
        Plain dictionary that can be embedded in metadata.yaml.
    """
    data = to_plain_config_data(asdict(robot_config))
    source_data = _load_metadata_source(source)
    if "simulation_model" in source_data:
        data["simulation_model"] = source_data["simulation_model"]
    return data


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
        }
    )


def retargeting_profile_config_to_metadata_dict(
    profile_config: RetargetingProfileConfig,
    source: str | Path | Mapping[str, Any] | None = None,
) -> dict:
    """Convert a retargeting profile config object to metadata-safe plain data.

    Args:
        profile_config: Typed robot-method retargeting profile config.
        source: Optional original profile config containing execution-layer fields.

    Returns:
        Plain dictionary that can be embedded in metadata.yaml.
    """
    data = to_plain_config_data(asdict(profile_config))
    source_data = _load_metadata_source(source)
    if "teleoperation" in source_data:
        data["teleoperation"] = source_data["teleoperation"]
    return data


def detection_source_config_to_metadata_dict(detection_source_config: DetectionSourceConfig) -> dict:
    """Convert a detection source config object to metadata-safe plain data.

    Args:
        detection_source_config: Typed detector source calibration config.

    Returns:
        Plain dictionary that can be embedded in metadata.yaml.
    """
    return to_plain_config_data(asdict(detection_source_config))


def create_robot_replay_context_from_config(
    robot_config: RobotConfig,
    retargeting_profile_config: RetargetingProfileConfig,
    detection_source_config: DetectionSourceConfig,
) -> RobotReplayContext:
    robot_model = RobotPinocchio(robot_file_path=robot_config.robot_file_path, robot_file_type=robot_config.model.type)
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )
    return RobotReplayContext(
        robot_config=robot_config,
        robot_name=robot_config.name,
        robot_file_path=robot_config.robot_file_path,
        robot_model=robot_model,
        robot_adaptor=robot_adaptor,
        actuated_joints_name=list(robot_config.actuated_joints),
        init_joint_pos=np.asarray(robot_config.initial_qpos, dtype=float),
        visual_frame_names=list(robot_config.visual_frame_names),
        wrist_frame_name=robot_config.wrist_frame_name,
        human_hand_scale=robot_config.human_hand_scale,
        benchmark_config=robot_config.benchmark,
        retargeting_runtime_config=retargeting_profile_config.retargeting,
        detection_source_config=detection_source_config,
        retargeting_profile_config=retargeting_profile_config,
    )


def create_robot_replay_context_from_metadata(metadata: RetargetingRunMetadata) -> RobotReplayContext:
    """Create a replay context from saved trajectory metadata.

    Args:
        metadata: Metadata loaded from a saved retargeting artifact.

    Returns:
        Robot replay context reconstructed from the embedded robot config.
    """
    robot_config = load_robot_config(metadata.robot_config)
    profile_source = metadata.extra.get("retargeting_profile_config")
    if profile_source is None:
        raise ValueError("Saved artifact metadata is missing retargeting_profile_config.")
    profile_config = load_retargeting_profile_config(profile_source)
    detection_source = metadata.extra.get("detection_source_config")
    if detection_source is None:
        raise ValueError("Saved artifact metadata is missing detection_source_config.")
    detection_source_config = load_detection_source_config(detection_source)
    return create_robot_replay_context_from_config(robot_config, profile_config, detection_source_config)


def create_robot_replay_context(
    profile_config_path: str | Path = DEFAULT_RETARGETING_PROFILE_CONFIG_PATH,
    detection_source_config_path: str | Path = DEFAULT_DETECTION_SOURCE_CONFIG_PATH,
) -> RobotReplayContext:
    profile_config = load_retargeting_profile_config(profile_config_path)
    detection_source_config = load_detection_source_config(detection_source_config_path)
    return create_robot_replay_context_from_config(
        load_robot_config(profile_config.robot),
        profile_config,
        detection_source_config,
    )


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
    start: int = 0,
    end: int = -1,
    stride: int = 1,
    robot_config: RobotConfig | None = None,
    retargeting_config: RetargetingConfig | None = None,
    retargeting_profile_config: RetargetingProfileConfig | None = None,
    detection_source_config: DetectionSourceConfig | None = None,
    teleoperation_mode_config: TeleoperationModeConfig | None = None,
    solver_config: SolverConfig | None = None,
    robot_config_path: str | Path | None = None,
    retargeting_config_path: str | Path | None = None,
    retargeting_profile_config_path: str | Path | Mapping[str, Any] | None = None,
    detection_source_config_path: str | Path | None = None,
    teleoperation_mode_config_path: str | Path | None = None,
    solver_config_path: str | Path | None = None,
) -> tuple[RobotReplayContext, RetargetingTrajectory, RetargetingRunMetadata]:
    """Run offline retargeting and return a persistent trajectory artifact in memory.

    Args:
        data_file: Offline AVP replay file.
        start: First input frame index.
        end: Last input frame index, inclusive; negative means the final frame.
        stride: Frame stride.
        robot_config: Optional already-loaded robot config.
        retargeting_config: Optional already-loaded retargeting method config.
        retargeting_profile_config: Optional already-loaded robot-method profile config.
        detection_source_config: Optional already-loaded detection source config.
        teleoperation_mode_config: Optional already-loaded teleoperation runtime mode config.
        solver_config: Optional already-loaded solver config.
        robot_config_path: Optional robot config path.
        retargeting_config_path: Optional retargeting method config path.
        retargeting_profile_config_path: Optional retargeting profile config source.
        detection_source_config_path: Optional detection source config path.
        teleoperation_mode_config_path: Optional teleoperation runtime mode config path.
        solver_config_path: Optional solver config path.

    Returns:
        Tuple of replay context, trajectory arrays, and run metadata.
    """
    profile_metadata_source = retargeting_profile_config_path
    if retargeting_profile_config is None:
        profile_metadata_source = (
            retargeting_profile_config_path
            if retargeting_profile_config_path is not None
            else DEFAULT_RETARGETING_PROFILE_CONFIG_PATH
        )
        retargeting_profile_config = load_retargeting_profile_config(
            profile_metadata_source
        )
    if detection_source_config is None:
        detection_source_config = load_detection_source_config(
            detection_source_config_path
            if detection_source_config_path is not None
            else DEFAULT_DETECTION_SOURCE_CONFIG_PATH
        )
    if detection_source_config.input_device != "avp":
        raise ValueError("Offline replay currently requires an avp detection source.")
    if robot_config is None:
        robot_config = (
            load_robot_config(robot_config_path)
            if robot_config_path is not None
            else load_robot_config(retargeting_profile_config.robot)
        )
    robot_metadata_source = robot_config_path if robot_config_path is not None else retargeting_profile_config.robot
    if retargeting_config is None:
        retargeting_config = (
            load_retargeting_config(retargeting_config_path)
            if retargeting_config_path is not None
            else load_retargeting_config(retargeting_profile_config.method)
        )
    if teleoperation_mode_config is None:
        teleoperation_mode_config = load_teleoperation_mode_config(teleoperation_mode_config_path)
    if solver_config is None:
        solver_config = (
            load_solver_config(solver_config_path)
            if solver_config_path is not None
            else default_solver_config()
        )

    retargeting_profile_config.validate(robot_config)
    context = create_robot_replay_context_from_config(robot_config, retargeting_profile_config, detection_source_config)
    flow = build_batch_retarget_flow(
        data_file=data_file,
        start=start,
        end=end,
        stride=stride,
        robot_config=context.robot_config,
        profile_config=context.retargeting_profile_config,
        method_config=retargeting_config,
        detection_config=context.detection_source_config,
        mode_config=teleoperation_mode_config,
        solver_config=solver_config,
        robot_adaptor=context.robot_adaptor,
        evaluate=True,
    )

    frames: List[RetargetReplayFrame] = []
    for record in flow.run():
        if record.source_index is None:
            raise ValueError("Offline AVP batch results require a source frame index.")
        observation = record.observation
        hand_kps_wrist = observation.keypoints_wrist
        wrist_pose_world = observation.wrist_pose_world
        hand_kps_world = transformPositions(hand_kps_wrist, target_frame_pose_inv=wrist_pose_world)
        frame_qpos = np.asarray(record.retargeted_qpos, dtype=float).copy()

        frames.append(
            RetargetReplayFrame(
                frame_idx=record.source_index,
                hand_keypoints_wrist=hand_kps_wrist,
                hand_keypoints_world=hand_kps_world,
                wrist_pose_world=wrist_pose_world,
                qpos=frame_qpos,
                robot_frame_poses=compute_robot_frame_poses(context, frame_qpos),
                err=dict(record.diagnostics),
            )
        )

    trajectory = frames_to_trajectory(frames)
    metadata = RetargetingRunMetadata(
        source_data=str(data_file),
        robot_config=robot_config_to_metadata_dict(robot_config, robot_metadata_source),
        retargeting_config=None
        if retargeting_config is None
        else retargeting_config_to_metadata_dict(retargeting_config),
        start=start,
        end=end,
        stride=stride,
        num_frames=trajectory.n_frames,
        qpos_dim=trajectory.qpos_dim,
        extra={
            "retargeting_profile_config": retargeting_profile_config_to_metadata_dict(
                retargeting_profile_config,
                profile_metadata_source,
            ),
            "detection_source_config": detection_source_config_to_metadata_dict(detection_source_config),
        },
    )
    return context, trajectory, metadata
