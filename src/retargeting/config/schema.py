from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retargeting.config.io import load_config_data, resolve_asset_path, resolve_project_path, to_plain_config_data


@dataclass(frozen=True)
class RobotModelConfig:
    type: str
    path: str
    path_is_symlink: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotModelConfig":
        return cls(
            type=str(data["type"]),
            path=str(data["path"]),
            path_is_symlink=bool(data.get("path_is_symlink", False)),
        )

    def resolved_link_path(self) -> Path:
        return resolve_project_path(self.path)

    def resolved_path(self) -> Path:
        return resolve_asset_path(self.path, follow_symlink=self.path_is_symlink)

    def loader_path(self) -> str:
        return str(self.resolved_path())

    def validate(self) -> None:
        path = self.resolved_link_path()
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"Robot model path does not exist: {path}")
        if self.path_is_symlink:
            if not path.is_symlink():
                raise ValueError(f"Expected robot model path to be a symlink: {path}")
            target = self.resolved_path()
            if not target.exists():
                raise FileNotFoundError(f"Robot model symlink target does not exist: {target}")
        elif not self.resolved_path().exists():
            raise FileNotFoundError(f"Robot model resolved path does not exist: {self.resolved_path()}")


@dataclass(frozen=True)
class RobotBenchmarkFingertipConfig:
    link_name: str
    human_tip_index: int
    human_direction_base_index: int
    robot_direction_axis: str
    is_thumb: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotBenchmarkFingertipConfig":
        """Build one benchmark fingertip mapping from config data.

        Args:
            data: A mapping with the robot fingertip frame, MANO keypoint indices, and direction axis.

        Returns:
            A typed fingertip benchmark mapping.
        """
        return cls(
            link_name=str(data["link_name"]),
            human_tip_index=int(data["human_tip_index"]),
            human_direction_base_index=int(data["human_direction_base_index"]),
            robot_direction_axis=str(data["robot_direction_axis"]),
            is_thumb=bool(data.get("is_thumb", False)),
        )

    def validate(self) -> None:
        """Validate one fingertip benchmark mapping.

        Args:
            None.

        Returns:
            None.
        """
        if self.robot_direction_axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported robot_direction_axis: {self.robot_direction_axis}")
        if self.human_tip_index < 0 or self.human_direction_base_index < 0:
            raise ValueError("Human keypoint indices must be non-negative.")


@dataclass(frozen=True)
class RobotBenchmarkConfig:
    wrist_link_name: str
    fingertips: tuple[RobotBenchmarkFingertipConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotBenchmarkConfig":
        """Build benchmark metadata from robot config data.

        Args:
            data: A mapping with wrist frame and fingertip mappings for benchmark metrics.

        Returns:
            A typed robot benchmark config.
        """
        return cls(
            wrist_link_name=str(data["wrist_link_name"]),
            fingertips=tuple(RobotBenchmarkFingertipConfig.from_dict(item) for item in data["fingertips"]),
        )

    @property
    def fingertip_link_names(self) -> tuple[str, ...]:
        """Return fingertip frame names in benchmark order.

        Args:
            None.

        Returns:
            Ordered robot fingertip frame names.
        """
        return tuple(fingertip.link_name for fingertip in self.fingertips)

    @property
    def human_tip_indices(self) -> tuple[int, ...]:
        """Return MANO fingertip keypoint indices in benchmark order.

        Args:
            None.

        Returns:
            Ordered human fingertip keypoint indices.
        """
        return tuple(fingertip.human_tip_index for fingertip in self.fingertips)

    @property
    def human_direction_base_indices(self) -> tuple[int, ...]:
        """Return MANO base keypoint indices used for fingertip direction vectors.

        Args:
            None.

        Returns:
            Ordered human keypoint indices that define the base of each direction vector.
        """
        return tuple(fingertip.human_direction_base_index for fingertip in self.fingertips)

    @property
    def primary_fingertips(self) -> tuple[RobotBenchmarkFingertipConfig, ...]:
        """Return non-thumb fingertips for thumb-relative metrics.

        Args:
            None.

        Returns:
            Ordered fingertip mappings excluding the thumb.
        """
        return tuple(fingertip for fingertip in self.fingertips if not fingertip.is_thumb)

    @property
    def thumb_fingertip(self) -> RobotBenchmarkFingertipConfig:
        """Return the configured thumb fingertip mapping.

        Args:
            None.

        Returns:
            The single fingertip mapping marked as thumb.
        """
        thumbs = tuple(fingertip for fingertip in self.fingertips if fingertip.is_thumb)
        if len(thumbs) != 1:
            raise ValueError(f"Expected exactly one thumb fingertip, got {len(thumbs)}.")
        return thumbs[0]

    def validate(self, available_frame_names: tuple[str, ...]) -> None:
        """Validate benchmark metadata against the robot visual frames.

        Args:
            available_frame_names: Robot frame names exposed for visualization and metrics.

        Returns:
            None.
        """
        if not self.fingertips:
            raise ValueError("Robot benchmark fingertips must not be empty.")
        self.thumb_fingertip
        frame_names = set(available_frame_names)
        required_frame_names = {self.wrist_link_name, *self.fingertip_link_names}
        missing_frame_names = sorted(required_frame_names - frame_names)
        if missing_frame_names:
            raise ValueError(f"Benchmark frame names are missing from visual_frame_names: {missing_frame_names}")
        for fingertip in self.fingertips:
            fingertip.validate()


def _float_tuple(values: Any, field_name: str) -> tuple[float, ...]:
    """Convert a sequence-like config field to a tuple of floats.

    Args:
        values: Raw sequence loaded from YAML or a composed config mapping.
        field_name: Human-readable field name used in validation errors.

    Returns:
        Tuple of floats with the same ordering as the source sequence.
    """
    if values is None:
        raise ValueError(f"Missing required float sequence: {field_name}")
    return tuple(float(item) for item in values)


@dataclass(frozen=True)
class RetargetingRuntimeConfig:
    """Robot-specific quantities used directly by the retargeting objective."""

    arm_dof: int
    human_wrist_index: int
    joint_position_weights: tuple[float, ...]
    joint_velocity_weights: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetingRuntimeConfig":
        """Build robot-specific objective metadata from config data.

        Args:
            data: Mapping with robot-specific objective weights and indices.

        Returns:
            Typed retargeting runtime configuration.
        """
        return cls(
            arm_dof=int(data["arm_dof"]),
            human_wrist_index=int(data.get("human_wrist_index", 0)),
            joint_position_weights=_float_tuple(data["joint_position_weights"], "joint_position_weights"),
            joint_velocity_weights=_float_tuple(data["joint_velocity_weights"], "joint_velocity_weights"),
        )

    def validate(self, qpos_size: int) -> None:
        """Validate objective metadata against the robot qpos size.

        Args:
            qpos_size: Number of actuated robot joints configured for this robot.

        Returns:
            None.
        """
        if not 0 < self.arm_dof <= qpos_size:
            raise ValueError(f"arm_dof must be in [1, {qpos_size}], got {self.arm_dof}.")
        if self.human_wrist_index < 0:
            raise ValueError("human_wrist_index must be non-negative.")
        for field_name, values in [
            ("joint_position_weights", self.joint_position_weights),
            ("joint_velocity_weights", self.joint_velocity_weights),
        ]:
            if len(values) != qpos_size:
                raise ValueError(f"{field_name} has {len(values)} values but robot qpos has {qpos_size}.")


@dataclass(frozen=True)
class TeleoperationCommandConfig:
    """Robot-output limits owned by the teleoperation execution layer."""

    max_joint_speed: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeleoperationCommandConfig":
        """Build output command limits from config data.

        Args:
            data: Mapping with robot command limits.

        Returns:
            Typed teleoperation command configuration.
        """
        return cls(max_joint_speed=_float_tuple(data["max_joint_speed"], "max_joint_speed"))

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


# Kept as an import-compatible name for callers that used the old schema type.
RobotTeleoperationConfig = RetargetingRuntimeConfig


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
            data: Mapping with detector source name, input device, and source-world to robot-world transform.

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
class RobotConfig:
    name: str
    model: RobotModelConfig
    actuated_joints: tuple[str, ...]
    initial_qpos: tuple[float, ...]
    visual_frame_names: tuple[str, ...]
    wrist_frame_name: str
    human_hand_scale: float
    benchmark: RobotBenchmarkConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotConfig":
        return cls(
            name=str(data["name"]),
            model=RobotModelConfig.from_dict(data["model"]),
            actuated_joints=tuple(str(item) for item in data["actuated_joints"]),
            initial_qpos=tuple(float(item) for item in data["initial_qpos"]),
            visual_frame_names=tuple(str(item) for item in data["visual_frame_names"]),
            wrist_frame_name=str(data["wrist_frame_name"]),
            human_hand_scale=float(data["human_hand_scale"]),
            benchmark=RobotBenchmarkConfig.from_dict(data["benchmark"]),
        )

    @property
    def robot_file_path(self) -> str:
        return self.model.loader_path()

    def validate(self) -> None:
        self.model.validate()
        if len(self.actuated_joints) != len(self.initial_qpos):
            raise ValueError(
                f"Robot config {self.name} has {len(self.actuated_joints)} actuated joints "
                f"but {len(self.initial_qpos)} initial qpos values."
            )
        if self.wrist_frame_name not in self.visual_frame_names:
            raise ValueError(f"wrist_frame_name must be included in visual_frame_names: {self.wrist_frame_name}")
        self.benchmark.validate(self.visual_frame_names)


@dataclass(frozen=True)
class RetargetTargetConfig:
    wrist_link_name: str
    link_pairs: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetTargetConfig":
        link_pairs = tuple(tuple(str(value) for value in pair) for pair in data["link_pairs"])
        return cls(wrist_link_name=str(data["wrist_link_name"]), link_pairs=link_pairs)

    def validate(self) -> None:
        if not self.link_pairs:
            raise ValueError("Retarget target link_pairs must not be empty.")
        for pair in self.link_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected retarget link pair length 2, got: {pair}")


@dataclass(frozen=True)
class JointLimitOverride:
    indices: tuple[int, ...]
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JointLimitOverride":
        return cls(
            indices=tuple(int(item) for item in data["indices"]),
            lower=None if data.get("lower") is None else float(data["lower"]),
            upper=None if data.get("upper") is None else float(data["upper"]),
        )


@dataclass(frozen=True)
class RetargetingObjectiveWeightsConfig:
    world_thumb: float = 10.0
    wrist_fingertip: float = 1.0
    thumb_primary: float = 10.0
    fingertip_orientation: float = 10.0
    wrist_rotation: float = 0.1
    arm_link_vector: float = 10.0
    arm_wrist_rotation: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetargetingObjectiveWeightsConfig":
        """Build objective term weights from config data.

        Args:
            data: Optional mapping with scalar weights for retargeting objective terms.

        Returns:
            Typed objective weight config.
        """
        data = {} if data is None else data
        return cls(
            world_thumb=float(data.get("world_thumb", 10.0)),
            wrist_fingertip=float(data.get("wrist_fingertip", 1.0)),
            thumb_primary=float(data.get("thumb_primary", 10.0)),
            fingertip_orientation=float(data.get("fingertip_orientation", 10.0)),
            wrist_rotation=float(data.get("wrist_rotation", 0.1)),
            arm_link_vector=float(data.get("arm_link_vector", 10.0)),
            arm_wrist_rotation=float(data.get("arm_wrist_rotation", 1.0)),
        )


@dataclass(frozen=True)
class RetargetingObjectiveConfig:
    pinch_transition_threshold: float = 0.1
    pinch_contact_threshold: float = 0.01
    pinch_sigmoid_slope: float = 10.0
    weights: RetargetingObjectiveWeightsConfig = RetargetingObjectiveWeightsConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetargetingObjectiveConfig":
        """Build high-level retargeting objective hyperparameters.

        Args:
            data: Optional mapping with pinch thresholds and objective weights.

        Returns:
            Typed objective config used to build per-frame optimizer references.
        """
        data = {} if data is None else data
        return cls(
            pinch_transition_threshold=float(data.get("pinch_transition_threshold", 0.1)),
            pinch_contact_threshold=float(data.get("pinch_contact_threshold", 0.01)),
            pinch_sigmoid_slope=float(data.get("pinch_sigmoid_slope", 10.0)),
            weights=RetargetingObjectiveWeightsConfig.from_dict(data.get("weights")),
        )

    def validate(self) -> None:
        """Validate objective hyperparameters.

        Args:
            None.

        Returns:
            None.
        """
        if self.pinch_contact_threshold < 0:
            raise ValueError("pinch_contact_threshold must be non-negative.")
        if self.pinch_transition_threshold <= self.pinch_contact_threshold:
            raise ValueError("pinch_transition_threshold must be larger than pinch_contact_threshold.")
        if self.pinch_sigmoid_slope <= 0:
            raise ValueError("pinch_sigmoid_slope must be positive.")


def _coerce_optimizer_param(value: Any) -> Any:
    """Convert optimizer params to floats while preserving backend-specific groups.

    Args:
        value: Raw optimizer param value loaded from YAML or a composed mapping.

    Returns:
        Float scalar for numeric leaves, or a nested dictionary for grouped backend params.
    """
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _coerce_optimizer_param(item) for key, item in value.items()}
    return float(value)


ABLATION_OPTION_DESCRIPTIONS: dict[int, str] = {
    0: "full",
    1: "without pinch",
    2: "actual pinch distance",
    3: "without orientation",
    4: "DexMV orientation",
    5: "replace thumb position with wrist position",
    6: "replace the thumb position term with a wrist position term, remove the fingertip orientation term and the pinch term",
    7: "remove joint position term",
    8: "option 6 plus option 7",
}


@dataclass(frozen=True)
class RetargetingConfig:
    type: str
    setting_id: int
    ablation_option: int
    optimizer_class: str
    optimizer_params: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetingConfig":
        optimizer = data["optimizer"]
        return cls(
            type=str(data["type"]),
            setting_id=int(data.get("setting_id", 0)),
            ablation_option=int(data.get("ablation_option", 0)),
            optimizer_class=str(optimizer["class"]),
            optimizer_params={key: _coerce_optimizer_param(value) for key, value in optimizer["params"].items()},
        )

    @property
    def ablation_description(self) -> str:
        """Return the human-readable description for the configured ablation option.

        Args:
            None.

        Returns:
            Description text for `ablation_option`.
        """
        return ABLATION_OPTION_DESCRIPTIONS[self.ablation_option]

    def validate(self) -> None:
        supported_optimizer_classes = {"VectorWristJointOptimizer", "VectorWristJointOptimizerV2"}
        if self.optimizer_class not in supported_optimizer_classes:
            raise ValueError(f"Unsupported optimizer class in Phase 2: {self.optimizer_class}")
        if self.ablation_option not in ABLATION_OPTION_DESCRIPTIONS:
            supported_options = sorted(ABLATION_OPTION_DESCRIPTIONS)
            raise ValueError(
                f"Unsupported ablation_option: {self.ablation_option}. Supported options: {supported_options}"
            )
        for required_param in ["huber_delta"]:
            if required_param not in self.optimizer_params:
                raise ValueError(f"Missing optimizer param: {required_param}")


@dataclass(frozen=True)
class RetargetingProfileConfig:
    name: str
    robot: str
    method: str
    target: RetargetTargetConfig
    objective: RetargetingObjectiveConfig
    retargeting: RetargetingRuntimeConfig
    teleoperation: TeleoperationCommandConfig
    joint_limit_overrides: tuple[JointLimitOverride, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetingProfileConfig":
        """Build a robot-method retargeting profile from config data.

        Args:
            data: Mapping with robot/method sources and robot-method-specific targets,
                objective weights, retargeting parameters, and teleoperation output limits.

        Returns:
            Typed retargeting profile config.
        """
        retargeting_data = data.get("retargeting")
        teleoperation_data = data.get("teleoperation")
        if retargeting_data is None:
            # Legacy profiles stored algorithm and command settings together.
            retargeting_data = teleoperation_data
        if teleoperation_data is None:
            raise ValueError("Retargeting profile requires a teleoperation command configuration.")
        return cls(
            name=str(data["name"]),
            robot=str(data["robot"]),
            method=str(data["method"]),
            target=RetargetTargetConfig.from_dict(data["target"]),
            objective=RetargetingObjectiveConfig.from_dict(data.get("objective")),
            retargeting=RetargetingRuntimeConfig.from_dict(retargeting_data),
            teleoperation=TeleoperationCommandConfig.from_dict(teleoperation_data),
            joint_limit_overrides=tuple(
                JointLimitOverride.from_dict(item) for item in data.get("joint_limit_overrides", [])
            ),
        )

    def validate(self, robot_config: RobotConfig | None = None) -> None:
        """Validate profile metadata, optionally against a loaded robot config.

        Args:
            robot_config: Optional robot config used to validate qpos-dependent
                teleoperation arrays and target frame names.

        Returns:
            None.
        """
        self.target.validate()
        self.objective.validate()
        for override in self.joint_limit_overrides:
            if not override.indices:
                raise ValueError("joint_limit_overrides indices must not be empty.")
        if robot_config is None:
            return
        self.retargeting.validate(len(robot_config.initial_qpos))
        self.teleoperation.validate(len(robot_config.initial_qpos))
        frame_names = {"world", *robot_config.visual_frame_names}
        target_frame_names = set()
        for origin_link, task_link in self.target.link_pairs:
            target_frame_names.add(origin_link)
            target_frame_names.add(task_link)
        target_frame_names.add(self.target.wrist_link_name)
        missing_frame_names = sorted(target_frame_names - frame_names)
        if missing_frame_names:
            raise ValueError(f"Profile target frame names are missing from robot visual_frame_names: {missing_frame_names}")


@dataclass(frozen=True)
class SolverConfig:
    name: str
    params: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverConfig":
        """Build a solver config from backend-specific config data.

        Args:
            data: Mapping with solver name and numeric backend params.

        Returns:
            Typed solver config.
        """
        return cls(
            name=str(data.get("name", data.get("solver", "nlopt_slsqp"))),
            params={str(key): float(value) for key, value in data.get("params", {}).items()},
        )

    def validate(self) -> None:
        """Validate backend-specific solver parameters.

        Args:
            None.

        Returns:
            None.
        """
        if self.name not in {"nlopt", "nlopt_slsqp", "scipy", "scipy_slsqp"}:
            raise ValueError(f"Unsupported solver: {self.name}")
        if self.name in {"nlopt", "nlopt_slsqp"} and "ftol_abs" not in self.params:
            raise ValueError("Missing solver param for nlopt: ftol_abs")
        if self.name in {"scipy", "scipy_slsqp"} and "ftol" not in self.params:
            raise ValueError("Missing solver param for scipy: ftol")
        if "maxtime" not in self.params:
            raise ValueError("Missing solver param: maxtime")


def default_solver_config() -> SolverConfig:
    """Return the default solver config used by legacy direct retargeting config paths.

    Args:
        None.

    Returns:
        Default NLopt SLSQP solver config.
    """
    return SolverConfig(name="nlopt_slsqp", params={"ftol_abs": 1e-5, "maxtime": -1.0})


def default_teleoperation_mode_config() -> TeleoperationModeConfig:
    """Return the default simulation-style teleoperation runtime mode.

    Args:
        None.

    Returns:
        Teleoperation mode config with no hardware execution and no output smoothing.
    """
    return TeleoperationModeConfig(
        name="simulation",
        robot_control=TeleoperationRobotControlConfig(
            use_hardware=False,
            use_virtual_hardware=False,
            use_high_freq_interp=False,
        ),
        output=TeleoperationOutputConfig(
            smooth_output_qpos=False,
            smoothing_alpha=0.3,
        ),
    )


@dataclass(frozen=True)
class ViewerConfig:
    fps: float = 30.0
    port: int = 8080
    no_robot_mesh: bool = False
    trail_length: int = 120
    human_keypoint_size: float = 0.018
    initial_camera_position: tuple[float, float, float] = (1.5, 1.5, 1.2)
    initial_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.45)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ViewerConfig":
        data = {} if data is None else data
        return cls(
            fps=float(data.get("fps", 30.0)),
            port=int(data.get("port", 8080)),
            no_robot_mesh=bool(data.get("no_robot_mesh", False)),
            trail_length=int(data.get("trail_length", 120)),
            human_keypoint_size=float(data.get("human_keypoint_size", 0.018)),
            initial_camera_position=tuple(float(value) for value in data.get("initial_camera_position", (1.5, 1.5, 1.2))),
            initial_camera_look_at=tuple(float(value) for value in data.get("initial_camera_look_at", (0.0, 0.0, 0.45))),
        )


@dataclass(frozen=True)
class ReplayAppConfig:
    run_name: str | None
    runtime_root: str
    viewer: ViewerConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayAppConfig":
        return cls(
            run_name=None if data.get("run_name") is None else str(data["run_name"]),
            runtime_root=str(data.get("runtime_root", "outputs")),
            viewer=ViewerConfig.from_dict(data.get("viewer")),
        )


def _load_config_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load config data from either a file path or an already-composed mapping.

    Args:
        source: A YAML path, plain mapping, or Hydra/OmegaConf-style mapping.

    Returns:
        A plain Python dictionary suitable for dataclass schema construction.
    """
    if isinstance(source, Mapping) or hasattr(source, "items"):
        data = to_plain_config_data(source)
        if not isinstance(data, dict):
            raise ValueError("Expected mapping config data.")
        return data
    return load_config_data(source)


def load_robot_config(path: str | Path | Mapping[str, Any] | RobotConfig) -> RobotConfig:
    if isinstance(path, RobotConfig):
        return path
    config = RobotConfig.from_dict(_load_config_source(path))
    config.validate()
    return config


def load_retargeting_config(path: str | Path | Mapping[str, Any] | RetargetingConfig) -> RetargetingConfig:
    if isinstance(path, RetargetingConfig):
        return path
    config = RetargetingConfig.from_dict(_load_config_source(path))
    config.validate()
    return config


def load_retargeting_profile_config(
    path: str | Path | Mapping[str, Any] | RetargetingProfileConfig,
) -> RetargetingProfileConfig:
    if isinstance(path, RetargetingProfileConfig):
        return path
    config = RetargetingProfileConfig.from_dict(_load_config_source(path))
    try:
        robot_config = load_robot_config(config.robot)
    except (FileNotFoundError, ValueError, TypeError):
        robot_config = None
    config.validate(robot_config)
    return config


def load_detection_source_config(
    path: str | Path | Mapping[str, Any] | DetectionSourceConfig,
) -> DetectionSourceConfig:
    if isinstance(path, DetectionSourceConfig):
        return path
    config = DetectionSourceConfig.from_dict(_load_config_source(path))
    config.validate()
    return config


def load_teleoperation_mode_config(
    path: str | Path | Mapping[str, Any] | TeleoperationModeConfig | None,
) -> TeleoperationModeConfig:
    if isinstance(path, TeleoperationModeConfig):
        return path
    config = (
        default_teleoperation_mode_config()
        if path is None
        else TeleoperationModeConfig.from_dict(_load_config_source(path))
    )
    config.validate()
    return config


def load_solver_config(path: str | Path | Mapping[str, Any] | SolverConfig | None) -> SolverConfig:
    if isinstance(path, SolverConfig):
        return path
    config = default_solver_config() if path is None else SolverConfig.from_dict(_load_config_source(path))
    config.validate()
    return config


def load_replay_app_config(path: str | Path | Mapping[str, Any] | ReplayAppConfig) -> ReplayAppConfig:
    if isinstance(path, ReplayAppConfig):
        return path
    return ReplayAppConfig.from_dict(_load_config_source(path))
