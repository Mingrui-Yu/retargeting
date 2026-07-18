"""Application-pipeline composition for the shared MuJoCo teleoperation runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from retargeting.config import (
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)
from retargeting.core.kinematics import RobotAdaptor, RobotPinocchio
from retargeting_apps.config import to_plain_config_data
from teleoperation.backends.mujoco import MujocoRobotBackend
from teleoperation.config import (
    load_detection_source_config,
    load_mujoco_robot_binding_config,
    load_mujoco_simulation_config,
    load_teleoperation_command_config,
    load_teleoperation_mode_config,
)
from teleoperation.mujoco_runtime import MujocoTeleoperationRuntime
from teleoperation.output import QposCommandLimiter
from teleoperation.session import TeleoperationSession


def build_mujoco_runtime(config: Any) -> MujocoTeleoperationRuntime:
    """Build the shared session, command policy, and headless MuJoCo backend.

    Args:
        config: Composed online or offline MuJoCo application configuration.

    Returns:
        Runtime ready to process individual observations or sensor frames.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected MuJoCo simulation config to be a mapping.")
    profile_source = config_data["profile"]
    profile_config = load_retargeting_profile_config(profile_source)
    robot_config = load_robot_config(profile_config.robot)
    simulator_binding = load_mujoco_robot_binding_config(profile_config.robot, robot_config=robot_config)
    command_config = load_teleoperation_command_config(profile_source, robot_config=robot_config)
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
        model_path=simulator_binding.simulation_file_path,
        joint_names=robot_config.actuated_joints,
        initial_qpos=robot_config.initial_qpos,
        config=simulator_config,
    )
    lower, upper = backend.joint_ctrlrange
    command_limiter = QposCommandLimiter(
        initial_qpos=backend.get_target_joint_pos(),
        max_joint_speed=np.asarray(command_config.max_joint_speed, dtype=float),
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
        startup_move_frames=simulator_config.startup_move_frames,
    )
