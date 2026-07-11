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


@dataclass(frozen=True)
class RobotConfig:
    name: str
    hand_type: str
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
            hand_type=str(data["hand_type"]),
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
        if self.hand_type not in {"leap", "shadow"}:
            raise ValueError(f"Unsupported hand_type in robot config: {self.hand_type}")
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


@dataclass(frozen=True)
class RetargetingConfig:
    type: str
    setting_id: int
    ablation_option: int
    optimizer_class: str
    optimizer_params: dict[str, Any]
    targets: dict[str, RetargetTargetConfig]
    joint_limit_overrides: tuple[JointLimitOverride, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetingConfig":
        optimizer = data["optimizer"]
        targets = {
            hand_type: RetargetTargetConfig.from_dict(target_data)
            for hand_type, target_data in data["targets"].items()
        }
        return cls(
            type=str(data["type"]),
            setting_id=int(data.get("setting_id", 0)),
            ablation_option=int(data.get("ablation_option", 0)),
            optimizer_class=str(optimizer["class"]),
            optimizer_params={key: _coerce_optimizer_param(value) for key, value in optimizer["params"].items()},
            targets=targets,
            joint_limit_overrides=tuple(
                JointLimitOverride.from_dict(item) for item in data.get("joint_limit_overrides", [])
            ),
        )

    def targets_for(self, hand_type: str) -> RetargetTargetConfig:
        try:
            return self.targets[hand_type]
        except KeyError as exc:
            raise ValueError(f"No retarget target config for hand_type: {hand_type}") from exc

    def validate(self) -> None:
        if self.optimizer_class != "VectorWristJointOptimizer":
            raise ValueError(f"Unsupported optimizer class in Phase 2: {self.optimizer_class}")
        for required_param in ["huber_delta"]:
            if required_param not in self.optimizer_params:
                raise ValueError(f"Missing optimizer param: {required_param}")
        for target in self.targets.values():
            target.validate()
        for override in self.joint_limit_overrides:
            if not override.indices:
                raise ValueError("joint_limit_overrides indices must not be empty.")


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
    return SolverConfig(name="nlopt_slsqp", params={"ftol_abs": 1e-5, "maxtime": 0.05})


@dataclass(frozen=True)
class ViewerConfig:
    fps: float = 30.0
    port: int = 8080
    no_robot_mesh: bool = False
    trail_length: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ViewerConfig":
        data = {} if data is None else data
        return cls(
            fps=float(data.get("fps", 30.0)),
            port=int(data.get("port", 8080)),
            no_robot_mesh=bool(data.get("no_robot_mesh", False)),
            trail_length=int(data.get("trail_length", 120)),
        )


@dataclass(frozen=True)
class ReplayAppConfig:
    data: str
    result: str | None
    robot: str
    retargeting: str
    solver: str | None
    start: int
    end: int
    stride: int
    viewer: ViewerConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayAppConfig":
        return cls(
            data=str(data["data"]),
            result=None if data.get("result") is None else str(data["result"]),
            robot=str(data["robot"]),
            retargeting=str(data["retargeting"]),
            solver=None if data.get("solver") is None else str(data["solver"]),
            start=int(data.get("start", 0)),
            end=int(data.get("end", -1)),
            stride=int(data.get("stride", 1)),
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
