"""Configuration-driven live retargeting into headless MuJoCo simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from retargeting.backends.mujoco import MujocoRobotBackend
from retargeting.config import (
    load_detection_source_config,
    load_mujoco_simulation_config,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
    load_teleoperation_mode_config,
    to_plain_config_data,
)
from retargeting.core.kinematics import RobotAdaptor, RobotPinocchio
from retargeting.inputs.avp import parse_avp_stream_frame
from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
from teleoperation.output import QposCommandLimiter
from teleoperation.session import TeleoperationSession


def build_mujoco_runtime(config: Any) -> MujocoTeleoperationRuntime:
    """Build the online session, command policy, and headless backend.

    Args:
        config: Composed MuJoCo simulation app configuration.

    Returns:
        Runtime ready to process individual observations or sensor frames.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected MuJoCo simulation config to be a mapping.")
    profile_config = load_retargeting_profile_config(config_data["profile"])
    robot_config = load_robot_config(profile_config.robot)
    if robot_config.simulation_model is None:
        raise ValueError(f"Robot config {robot_config.name} does not define a MuJoCo simulation model.")
    detection_source_config = load_detection_source_config(config_data["detection_source"])
    if detection_source_config.input_device != "avp":
        raise ValueError("The headless MuJoCo runtime currently supports the AVP input source only.")
    method_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config(config_data.get("solver"))
    teleoperation_mode_config = load_teleoperation_mode_config(config_data.get("teleoperation_mode"))
    simulator_config = load_mujoco_simulation_config(config_data.get("simulator"))

    robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
    robot_adaptor = RobotAdaptor(robot_model, list(robot_config.actuated_joints))
    session = TeleoperationSession(
        robot_adaptor=robot_adaptor,
        robot_config=robot_config,
        profile_config=profile_config,
        method_config=method_config,
        detection_source_config=detection_source_config,
        teleoperation_mode_config=teleoperation_mode_config,
        solver_config=solver_config,
        evaluate=bool(config_data.get("evaluate", False)),
    )
    backend = MujocoRobotBackend(
        model_path=robot_config.simulation_file_path,
        joint_names=robot_config.actuated_joints,
        initial_qpos=robot_config.initial_qpos,
        config=simulator_config,
    )
    lower, upper = backend.joint_ctrlrange
    command_limiter = QposCommandLimiter(
        initial_qpos=backend.get_target_joint_pos(),
        max_joint_speed=np.asarray(profile_config.teleoperation.max_joint_speed, dtype=float),
        command_hz=simulator_config.command_hz,
        lower=lower,
        upper=upper,
        ctrlrange_policy=simulator_config.ctrlrange_policy,
    )
    return MujocoTeleoperationRuntime(
        session=session,
        backend=backend,
        command_limiter=command_limiter,
        realtime=simulator_config.realtime,
    )


def initialize_avp_alignment(runtime: MujocoTeleoperationRuntime, sensor_data: Any) -> bool:
    """Initialize relative wrist alignment from the first valid AVP frame.

    Args:
        runtime: Online runtime whose session and backend require alignment.
        sensor_data: Raw AVP stream frame.

    Returns:
        True when a valid wrist pose initialized the alignment.
    """
    _, _, _, detection_wrist_pose = parse_avp_stream_frame(sensor_data)
    if detection_wrist_pose is None:
        return False
    session = runtime.session
    robot_qpos = session.robot_adaptor.forward_qpos(runtime.backend.get_joint_pos())
    robot_wrist_pose = session.robot_model.get_frame_pose(session.robot_config.wrist_frame_name, qpos=robot_qpos)
    session.set_robot_init_wrist_pose(robot_wrist_pose)
    session.set_detection_source_init_wrist_pose(
        session.pose_from_detection_world_to_robot_world(detection_wrist_pose)
    )
    return True


def run_mujoco_simulation_from_config(config: Any, argv: list[str] | None = None) -> dict[str, float]:
    """Run live AVP retargeting directly into headless MuJoCo at 20 Hz.

    Args:
        config: Composed MuJoCo simulation app configuration.
        argv: CLI arguments accepted for the common app-runner interface.

    Returns:
        Diagnostics from the last completed online frame.
    """
    del argv
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected MuJoCo simulation config to be a mapping.")
    runtime = build_mujoco_runtime(config_data)
    avp_ip = str(config_data["avp_ip"])
    max_frames_value = config_data.get("max_frames")
    max_frames = None if max_frames_value is None else int(max_frames_value)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")
    log_every_frames = int(config_data.get("log_every_frames", 20))
    if log_every_frames <= 0:
        raise ValueError("log_every_frames must be positive.")

    detector = runtime.session.detector
    detector.connect(avp_ip=avp_ip)
    alignment_initialized = False
    last_diagnostics: dict[str, float] = {}
    try:
        while max_frames is None or runtime.frame_count < max_frames:
            sensor_data = detector.get_raw_stream()
            if not alignment_initialized:
                alignment_initialized = initialize_avp_alignment(runtime, sensor_data)
                if not alignment_initialized:
                    result = runtime.step_hold()
                else:
                    result = runtime.step_sensor_data(sensor_data)
            else:
                result = runtime.step_sensor_data(sensor_data)
            last_diagnostics = result.diagnostics
            if runtime.frame_count % log_every_frames == 0:
                print(
                    f"frame={runtime.frame_count} sim_time={last_diagnostics.get('simulation_time', 0.0):.3f} "
                    f"tracking_max={last_diagnostics.get('tracking_error_max', 0.0):.6f} "
                    f"overrun={last_diagnostics.get('runtime_overrun', 0.0):.6f}"
                )
    except KeyboardInterrupt:
        pass
    return last_diagnostics
