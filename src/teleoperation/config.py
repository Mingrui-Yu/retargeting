"""Input, output, robot-control, and simulation configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retargeting.config.core import RobotConfig, RobotModelConfig
from retargeting.config.io import load_config_source


@dataclass(frozen=True)
class TeleoperationCommandConfig:
    """Robot-output limits owned by the teleoperation execution layer."""

    max_joint_speed: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeleoperationCommandConfig":
        """Build output command limits from config data.

        Args:
            data: Mapping with per-joint maximum command speeds.

        Returns:
            Typed teleoperation command configuration.
        """
        values = data.get("max_joint_speed")
        if values is None:
            raise ValueError("Teleoperation command config requires max_joint_speed.")
        return cls(max_joint_speed=tuple(float(item) for item in values))

    def validate(self, qpos_size: int) -> None:
        """Validate command limits against the robot configuration size.

        Args:
            qpos_size: Number of actuated qpos values.

        Returns:
            None.
        """
        if len(self.max_joint_speed) != qpos_size:
            raise ValueError(
                f"max_joint_speed has {len(self.max_joint_speed)} values but robot qpos has {qpos_size}."
            )
        values = np.asarray(self.max_joint_speed, dtype=float)
        if (values <= 0).any() or not np.isfinite(values).all():
            raise ValueError("max_joint_speed must contain positive finite values.")


def load_teleoperation_command_config(
    source: str | Path | Mapping[str, Any] | TeleoperationCommandConfig,
    *,
    robot_config: RobotConfig | None = None,
) -> TeleoperationCommandConfig:
    """Load command policy from a command mapping or an unchanged profile.

    Args:
        source: YAML path, composed mapping, nested command mapping, or typed config.
        robot_config: Optional robot config used to validate qpos dimension.

    Returns:
        Validated teleoperation command config.
    """
    if isinstance(source, TeleoperationCommandConfig):
        config = source
    else:
        data = load_config_source(source)
        command_data = data.get("teleoperation", data)
        if not isinstance(command_data, dict):
            raise ValueError("Expected teleoperation command settings to be a mapping.")
        config = TeleoperationCommandConfig.from_dict(command_data)
    if robot_config is not None:
        config.validate(len(robot_config.initial_qpos))
    return config


@dataclass(frozen=True)
class TeleoperationRobotControlConfig:
    use_hardware: bool = False
    use_virtual_hardware: bool = False
    use_high_freq_interp: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TeleoperationRobotControlConfig":
        """Build robot-control runtime flags from mode config data.

        Args:
            data: Optional mapping with hardware and interpolation flags.

        Returns:
            Typed robot-control runtime config.
        """
        data = {} if data is None else data
        return cls(
            use_hardware=bool(data.get("use_hardware", False)),
            use_virtual_hardware=bool(data.get("use_virtual_hardware", False)),
            use_high_freq_interp=bool(data.get("use_high_freq_interp", False)),
        )


@dataclass(frozen=True)
class TeleoperationOutputConfig:
    smooth_output_qpos: bool = False
    smoothing_alpha: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TeleoperationOutputConfig":
        """Build output filtering config from mode config data.

        Args:
            data: Optional mapping with output smoothing settings.

        Returns:
            Typed output filtering config.
        """
        data = {} if data is None else data
        return cls(
            smooth_output_qpos=bool(data.get("smooth_output_qpos", False)),
            smoothing_alpha=float(data.get("smoothing_alpha", 0.3)),
        )

    def validate(self) -> None:
        """Validate output filtering config.

        Args:
            None.

        Returns:
            None.
        """
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError(f"smoothing_alpha must be in [0, 1], got {self.smoothing_alpha}.")


@dataclass(frozen=True)
class TeleoperationModeConfig:
    name: str
    robot_control: TeleoperationRobotControlConfig
    output: TeleoperationOutputConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeleoperationModeConfig":
        """Build runtime teleoperation mode config from config data.

        Args:
            data: Mapping with mode name, robot-control flags, and output filtering.

        Returns:
            Typed teleoperation runtime mode config.
        """
        return cls(
            name=str(data["name"]),
            robot_control=TeleoperationRobotControlConfig.from_dict(data.get("robot_control")),
            output=TeleoperationOutputConfig.from_dict(data.get("output")),
        )

    def validate(self) -> None:
        """Validate runtime teleoperation mode config.

        Args:
            None.

        Returns:
            None.
        """
        if not self.name:
            raise ValueError("Teleoperation mode name must not be empty.")
        self.output.validate()


@dataclass(frozen=True)
class DetectionSourceConfig:
    name: str
    input_device: str
    rotation_euler_xyz_deg: tuple[float, float, float]
    translation: tuple[float, float, float]
    use_relative_wrist_alignment: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionSourceConfig":
        """Build detector-source calibration metadata from config data.

        Args:
            data: Mapping with detector source and source-to-robot transform.

        Returns:
            Typed detection source config.
        """
        transform_data = data.get("world_to_robot", data)
        return cls(
            name=str(data["name"]),
            input_device=str(data["input_device"]),
            rotation_euler_xyz_deg=tuple(
                float(item) for item in transform_data.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0])
            ),
            translation=tuple(float(item) for item in transform_data.get("translation", [0.0, 0.0, 0.0])),
            use_relative_wrist_alignment=bool(data.get("use_relative_wrist_alignment", False)),
        )

    def validate(self) -> None:
        """Validate detector-source calibration metadata.

        Args:
            None.

        Returns:
            None.
        """
        if self.input_device not in {"rgb", "avp"}:
            raise ValueError(f"Unsupported input_device: {self.input_device}")
        if len(self.rotation_euler_xyz_deg) != 3:
            raise ValueError("rotation_euler_xyz_deg must have exactly 3 values.")
        if len(self.translation) != 3:
            raise ValueError("translation must have exactly 3 values.")


@dataclass(frozen=True)
class MujocoRobotBindingConfig:
    """Associate one core robot embodiment with its MuJoCo model."""

    robot_name: str
    model: RobotModelConfig

    @classmethod
    def from_robot_dict(cls, data: dict[str, Any]) -> "MujocoRobotBindingConfig":
        """Build a simulator binding from an unchanged robot config mapping.

        Args:
            data: Robot mapping containing name and simulation_model fields.

        Returns:
            Typed MuJoCo robot binding.
        """
        simulation_model = data.get("simulation_model")
        if not isinstance(simulation_model, dict):
            raise ValueError(f"Robot config {data.get('name', '<unknown>')} does not define simulation_model.")
        return cls(
            robot_name=str(data["name"]),
            model=RobotModelConfig.from_dict(simulation_model),
        )

    @property
    def simulation_file_path(self) -> str:
        """Return the resolved simulator model path.

        Args:
            None.

        Returns:
            Resolved path to the configured MJCF model.
        """
        return self.model.loader_path()

    def validate(self, robot_config: RobotConfig | None = None) -> None:
        """Validate model type, asset path, and optional robot association.

        Args:
            robot_config: Optional core robot config that must match this binding.

        Returns:
            None.
        """
        self.model.validate()
        if self.model.type != "mjcf":
            raise ValueError(
                f"Robot config {self.robot_name} simulation_model must use type 'mjcf', got {self.model.type!r}."
            )
        if robot_config is not None and self.robot_name != robot_config.name:
            raise ValueError(
                f"Simulator binding robot {self.robot_name!r} does not match core robot {robot_config.name!r}."
            )


def load_mujoco_robot_binding_config(
    source: str | Path | Mapping[str, Any] | MujocoRobotBindingConfig,
    *,
    robot_config: RobotConfig | None = None,
) -> MujocoRobotBindingConfig:
    """Load a MuJoCo binding from a typed config or unchanged robot YAML.

    Args:
        source: Robot YAML path, composed robot mapping, or typed simulator binding.
        robot_config: Optional core robot config used to validate robot identity.

    Returns:
        Validated MuJoCo robot binding config.
    """
    if isinstance(source, MujocoRobotBindingConfig):
        config = source
    else:
        config = MujocoRobotBindingConfig.from_robot_dict(load_config_source(source))
    config.validate(robot_config)
    return config


@dataclass(frozen=True)
class MujocoSimulationConfig:
    """Timing and command behavior for headless MuJoCo execution."""

    command_hz: float = 20.0
    physics_timestep: float = 0.002
    realtime: bool = True
    ctrlrange_policy: str = "clip"
    startup_move_frames: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MujocoSimulationConfig":
        """Build MuJoCo simulation settings from config data.

        Args:
            data: Optional mapping with timing and command-range settings.

        Returns:
            Typed MuJoCo simulation configuration.
        """
        data = {} if data is None else data
        return cls(
            command_hz=float(data.get("command_hz", 20.0)),
            physics_timestep=float(data.get("physics_timestep", 0.002)),
            realtime=bool(data.get("realtime", True)),
            ctrlrange_policy=str(data.get("ctrlrange_policy", "clip")),
            startup_move_frames=data.get("startup_move_frames", 0),
        )

    @property
    def control_period(self) -> float:
        """Return the simulated duration advanced for each execution command.

        Args:
            None.

        Returns:
            Command period in seconds.
        """
        return 1.0 / self.command_hz

    @property
    def physics_steps_per_command(self) -> int:
        """Return the integer number of physics steps per retargeting command.

        Args:
            None.

        Returns:
            Number of MuJoCo physics steps for one command period.
        """
        return int(round(self.control_period / self.physics_timestep))

    def validate(self) -> None:
        """Validate timing ratios and command-range behavior.

        Args:
            None.

        Returns:
            None.
        """
        if self.command_hz <= 0:
            raise ValueError(f"command_hz must be positive, got {self.command_hz}.")
        if self.physics_timestep <= 0:
            raise ValueError(f"physics_timestep must be positive, got {self.physics_timestep}.")
        if self.ctrlrange_policy not in {"clip", "error"}:
            raise ValueError(f"ctrlrange_policy must be 'clip' or 'error', got {self.ctrlrange_policy!r}.")
        if isinstance(self.startup_move_frames, bool) or not isinstance(self.startup_move_frames, int):
            raise ValueError("startup_move_frames must be a non-negative integer.")
        if self.startup_move_frames < 0:
            raise ValueError("startup_move_frames must be a non-negative integer.")
        steps = self.physics_steps_per_command
        if steps <= 0 or not abs(steps * self.physics_timestep - self.control_period) <= 1e-12:
            raise ValueError(
                "command period must be an integer multiple of physics_timestep: "
                f"period={self.control_period}, physics_timestep={self.physics_timestep}."
            )


def default_teleoperation_mode_config() -> TeleoperationModeConfig:
    """Return the default simulation-style teleoperation runtime mode.

    Args:
        None.

    Returns:
        Teleoperation mode config with no hardware execution and no output smoothing.
    """
    return TeleoperationModeConfig(
        name="simulation",
        robot_control=TeleoperationRobotControlConfig(),
        output=TeleoperationOutputConfig(),
    )


def load_detection_source_config(
    path: str | Path | Mapping[str, Any] | DetectionSourceConfig,
) -> DetectionSourceConfig:
    """Load and validate detector calibration settings.

    Args:
        path: YAML path, composed mapping, or typed config.

    Returns:
        Validated detector source config.
    """
    if isinstance(path, DetectionSourceConfig):
        return path
    config = DetectionSourceConfig.from_dict(load_config_source(path))
    config.validate()
    return config


def load_teleoperation_mode_config(
    path: str | Path | Mapping[str, Any] | TeleoperationModeConfig | None,
) -> TeleoperationModeConfig:
    """Load and validate teleoperation mode settings.

    Args:
        path: YAML path, composed mapping, typed config, or None for defaults.

    Returns:
        Validated teleoperation mode config.
    """
    if isinstance(path, TeleoperationModeConfig):
        return path
    config = (
        default_teleoperation_mode_config()
        if path is None
        else TeleoperationModeConfig.from_dict(load_config_source(path))
    )
    config.validate()
    return config


def load_mujoco_simulation_config(
    path: str | Path | Mapping[str, Any] | MujocoSimulationConfig | None,
) -> MujocoSimulationConfig:
    """Load and validate MuJoCo simulation settings.

    Args:
        path: YAML path, composed mapping, typed config, or None for defaults.

    Returns:
        Validated MuJoCo simulation configuration.
    """
    if isinstance(path, MujocoSimulationConfig):
        config = path
    else:
        config = MujocoSimulationConfig.from_dict(None if path is None else load_config_source(path))
    config.validate()
    return config
