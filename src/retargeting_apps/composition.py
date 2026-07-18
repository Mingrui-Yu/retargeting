"""Pure construction helpers for complete teleoperation flow graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from retargeting.config import (
    RetargetingConfig,
    RetargetingProfileConfig,
    RobotConfig,
    SolverConfig,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)
from retargeting.core import Retargeter
from retargeting.core.kinematics import RobotAdaptor, RobotPinocchio
from retargeting.evaluation.robot_metrics import RobotBenchmark
from retargeting_apps.config import resolve_project_path, to_plain_config_data
from teleoperation.backends.mujoco import MujocoRobotBackend
from teleoperation.backends.kinematic import KinematicRobotBackend
from teleoperation.config import (
    DetectionSourceConfig,
    ExecutionBackendConfig,
    TeleoperationModeConfig,
    load_execution_backend_config,
    load_detection_source_config,
    load_mujoco_robot_binding_config,
    load_teleoperation_command_config,
    load_teleoperation_mode_config,
)
from teleoperation.flow import BatchRetargetFlow, ExecutionFlow
from teleoperation.inputs.avp import AvpOfflineInput, AvpOnlineInput
from teleoperation.observation_mapping import AvpRelativeWristMapper, StaticCalibrationMapper
from teleoperation.output import QposCommandLimiter, QposOutputFilter


def _build_mapper(
    detection_config: DetectionSourceConfig,
    robot_config: RobotConfig,
    robot_adaptor: RobotAdaptor,
    robot_model: RobotPinocchio,
):
    """Build the sensor-specific sample-to-core mapping strategy.

    Args:
        detection_config: Sensor selection and world calibration.
        robot_config: Target robot embodiment configuration.
        robot_adaptor: Target robot joint-order adapter.
        robot_model: Target robot kinematics model.

    Returns:
        Mapper matching the selected sensor and alignment semantics.
    """
    if detection_config.input_device == "avp":
        return AvpRelativeWristMapper(
            config=detection_config,
            human_hand_scale=robot_config.human_hand_scale,
            robot_adaptor=robot_adaptor,
            robot_model=robot_model,
            wrist_frame_name=robot_config.wrist_frame_name,
        )
    return StaticCalibrationMapper(detection_config, robot_config.human_hand_scale)


def _build_retargeting_components(
    *,
    robot_config: RobotConfig,
    profile_config: RetargetingProfileConfig,
    method_config: RetargetingConfig,
    detection_config: DetectionSourceConfig,
    mode_config: TeleoperationModeConfig,
    solver_config: SolverConfig,
    evaluate: bool,
    robot_adaptor: RobotAdaptor | None = None,
    robot_model: RobotPinocchio | None = None,
) -> tuple[RobotAdaptor, Retargeter, QposOutputFilter, Any, RobotBenchmark | None]:
    """Build components shared by command execution and offline batch flows.

    Args:
        robot_config: Target robot embodiment configuration.
        profile_config: Robot-method objective profile.
        method_config: Retargeting optimizer selection.
        detection_config: Sensor selection and calibration.
        mode_config: Output filtering configuration.
        solver_config: Numerical solver backend configuration.
        evaluate: Whether to compute raw robot benchmark metrics.
        robot_adaptor: Optional existing joint-order adapter.
        robot_model: Optional existing kinematics model paired with the adapter.

    Returns:
        Adapter, solver, output filter, mapper, and optional evaluator.
    """
    if robot_adaptor is None:
        robot_model = RobotPinocchio(robot_config.robot_file_path, robot_config.model.type)
        robot_adaptor = RobotAdaptor(robot_model, list(robot_config.actuated_joints))
    elif robot_model is None:
        robot_model = robot_adaptor.robot_model
    retargeter = Retargeter(
        robot_adaptor=robot_adaptor,
        robot_config=robot_config,
        profile_config=profile_config,
        method_config=method_config,
        solver_config=solver_config,
    )
    output_filter = QposOutputFilter(retargeter.qpos_init, mode_config)
    mapper = _build_mapper(detection_config, robot_config, robot_adaptor, robot_model)
    evaluator = RobotBenchmark(robot_adaptor, robot_config.benchmark) if evaluate else None
    return robot_adaptor, retargeter, output_filter, mapper, evaluator


def _legacy_value(config_data: dict[str, Any], input_data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read an input-runtime value while preserving old root-level app fields.

    Args:
        config_data: Complete composed app config.
        input_data: Nested input config, usually from the new ``input`` key.
        key: Runtime field name to resolve.
        default: Value returned when neither config location defines the key.

    Returns:
        Resolved value from nested input config, root config, or the default.
    """
    return config_data.get(key, input_data.get(key, default))


def _resolve_input_config(config_data: dict[str, Any]) -> tuple[DetectionSourceConfig, dict[str, Any]]:
    """Resolve calibration and runtime input settings from new or legacy keys.

    Args:
        config_data: Complete composed app config.

    Returns:
        Typed detection/input calibration and plain runtime input mapping.
    """
    input_source = config_data.get("input")
    if input_source is None:
        raise ValueError("Execution config requires an input mapping.")
    detection_config = load_detection_source_config(input_source)
    input_data = input_source if isinstance(input_source, dict) else {}
    if detection_config.input_device != "avp":
        raise ValueError("Execution apps currently require an avp input source.")
    return detection_config, input_data


def _resolve_backend_config(config_data: dict[str, Any]) -> ExecutionBackendConfig:
    """Resolve backend selection from the canonical backend key.

    Args:
        config_data: Complete composed app config.

    Returns:
        Validated execution backend config.
    """
    if "backend" not in config_data:
        raise ValueError("Execution config requires a backend mapping.")
    return load_execution_backend_config(config_data["backend"])


def _build_backend(
    *,
    backend_config: ExecutionBackendConfig,
    profile_robot_source: Any,
    robot_config: RobotConfig,
) -> Any:
    """Build one configured robot backend for an execution flow.

    Args:
        backend_config: Backend selection and timing settings.
        profile_robot_source: Robot config source used to load simulator bindings.
        robot_config: Loaded robot embodiment config.

    Returns:
        Backend implementing the shared robot command protocol.
    """
    if backend_config.name == "kinematic":
        return KinematicRobotBackend(
            initial_qpos=robot_config.initial_qpos,
            control_period=backend_config.control_period,
        )
    simulator_binding = load_mujoco_robot_binding_config(profile_robot_source, robot_config=robot_config)
    return MujocoRobotBackend(
        model_path=simulator_binding.simulation_file_path,
        joint_names=robot_config.actuated_joints,
        initial_qpos=robot_config.initial_qpos,
        config=backend_config.to_mujoco_simulation_config(),
    )


def _backend_ctrlrange(backend: Any, qpos_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return actuator command bounds when a backend exposes them.

    Args:
        backend: Configured robot backend.
        qpos_size: Number of commanded robot qpos values.

    Returns:
        Lower and upper qpos bounds for command limiting.
    """
    if hasattr(backend, "joint_ctrlrange"):
        return backend.joint_ctrlrange
    return np.full(qpos_size, -np.inf, dtype=float), np.full(qpos_size, np.inf, dtype=float)


def build_execution_flow(config: Any) -> ExecutionFlow:
    """Build a complete live or archived backend-neutral execution flow.

    Args:
        config: Composed online or offline execution application configuration.

    Returns:
        Top-level backend-neutral execution flow ready for ``run`` or ``step``.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected execution config to be a mapping.")
    profile_source = config_data["profile"]
    profile_config = load_retargeting_profile_config(profile_source)
    robot_config = load_robot_config(profile_config.robot)
    detection_config, input_data = _resolve_input_config(config_data)
    method_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config(config_data.get("solver"))
    mode_config = load_teleoperation_mode_config(config_data.get("teleoperation_mode"))
    if mode_config.pipeline.use_relative_wrist_alignment is not None:
        detection_config = replace(
            detection_config,
            use_relative_wrist_alignment=mode_config.pipeline.use_relative_wrist_alignment,
        )
    backend_config = _resolve_backend_config(config_data)
    command_config = load_teleoperation_command_config(profile_source, robot_config=robot_config)
    _, retargeter, output_filter, mapper, evaluator = _build_retargeting_components(
        robot_config=robot_config,
        profile_config=profile_config,
        method_config=method_config,
        detection_config=detection_config,
        mode_config=mode_config,
        solver_config=solver_config,
        evaluate=bool(config_data.get("evaluate", False)),
    )
    backend = _build_backend(
        backend_config=backend_config,
        profile_robot_source=profile_config.robot,
        robot_config=robot_config,
    )
    lower, upper = _backend_ctrlrange(backend, len(robot_config.initial_qpos))
    ctrlrange_policy = mode_config.pipeline.ctrlrange_policy or backend_config.ctrlrange_policy
    command_policy = QposCommandLimiter(
        initial_qpos=backend.get_target_joint_pos(),
        max_joint_speed=np.asarray(command_config.max_joint_speed, dtype=float),
        command_hz=backend_config.command_hz,
        lower=lower,
        upper=upper,
        ctrlrange_policy=ctrlrange_policy,
    )

    loop = _legacy_value(config_data, input_data, "loop", False)
    if not isinstance(loop, bool):
        raise ValueError("loop must be a boolean.")
    input_mode = str(_legacy_value(config_data, input_data, "mode", "offline" if "data" in config_data else "online"))
    if input_mode not in {"online", "offline"}:
        raise ValueError(f"input.mode must be 'online' or 'offline', got {input_mode!r}.")
    if input_mode == "offline":
        data_file = _legacy_value(config_data, input_data, "data")
        if data_file is None:
            raise ValueError("Offline input requires input.data.")
        source_hz = float(_legacy_value(config_data, input_data, "source_hz", backend_config.command_hz))
        if source_hz <= 0:
            raise ValueError(f"source_hz must be positive, got {source_hz}.")
        if abs(1.0 / source_hz - backend_config.control_period) > 1e-12:
            raise ValueError(
                "Offline human source_hz must match the backend command rate until timestamp resampling is supported: "
                f"source_hz={source_hz}, command_hz={backend_config.command_hz}."
            )
        hand_input = AvpOfflineInput(
            resolve_project_path(str(data_file)),
            start=int(_legacy_value(config_data, input_data, "start", 0)),
            end=int(_legacy_value(config_data, input_data, "end", -1)),
        )
        max_frames = None
    else:
        avp_ip = _legacy_value(config_data, input_data, "avp_ip")
        if avp_ip is None:
            raise ValueError("Online input requires input.avp_ip.")
        hand_input = AvpOnlineInput(str(avp_ip))
        max_frames_value = _legacy_value(config_data, input_data, "max_frames")
        max_frames = None if max_frames_value is None else int(max_frames_value)
    return ExecutionFlow(
        input=hand_input,
        observation_mapper=mapper,
        retargeter=retargeter,
        output_filter=output_filter,
        evaluator=evaluator,
        command_policy=command_policy,
        backend=backend,
        realtime=backend_config.realtime if mode_config.pipeline.realtime is None else mode_config.pipeline.realtime,
        startup_move_frames=(
            backend_config.startup_move_frames
            if mode_config.pipeline.startup_move_frames is None
            else mode_config.pipeline.startup_move_frames
        ),
        loop=loop,
        max_frames=max_frames,
    )


def build_batch_retarget_flow(
    *,
    data_file: str,
    start: int,
    end: int,
    stride: int,
    robot_config: RobotConfig,
    profile_config: RetargetingProfileConfig,
    method_config: RetargetingConfig,
    detection_config: DetectionSourceConfig,
    mode_config: TeleoperationModeConfig,
    solver_config: SolverConfig,
    robot_adaptor: RobotAdaptor,
    evaluate: bool = True,
) -> BatchRetargetFlow:
    """Build a complete finite artifact-retargeting flow.

    Args:
        data_file: Archived raw AVP NPZ path.
        start: First selected source frame.
        end: Last selected source frame, inclusive.
        stride: Positive source-frame selection stride.
        robot_config: Target robot embodiment configuration.
        profile_config: Robot-method objective profile.
        method_config: Retargeting optimizer selection.
        detection_config: AVP calibration configuration.
        mode_config: Output filtering configuration.
        solver_config: Numerical solver backend configuration.
        robot_adaptor: Existing replay-context joint-order adapter.
        evaluate: Whether to compute raw benchmark metrics.

    Returns:
        Backend-free finite batch flow.
    """
    if detection_config.input_device != "avp":
        raise ValueError("Offline batch retargeting currently requires an avp input source.")
    _, retargeter, output_filter, mapper, evaluator = _build_retargeting_components(
        robot_config=robot_config,
        profile_config=profile_config,
        method_config=method_config,
        detection_config=detection_config,
        mode_config=mode_config,
        solver_config=solver_config,
        evaluate=evaluate,
        robot_adaptor=robot_adaptor,
        robot_model=robot_adaptor.robot_model,
    )
    return BatchRetargetFlow(
        input=AvpOfflineInput(data_file, start=start, end=end, stride=stride),
        observation_mapper=mapper,
        retargeter=retargeter,
        output_filter=output_filter,
        initial_robot_qpos=np.asarray(robot_config.initial_qpos, dtype=float),
        evaluator=evaluator,
    )
