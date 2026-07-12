from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
import yaml

from retargeting.config import resolve_project_path, to_plain_config_data


RESULT_SCHEMA_VERSION = 1
RESULT_FILE_NAME = "result.npz"
METADATA_FILE_NAME = "metadata.yaml"
ROBOT_FRAME_POSE_PREFIX = "robot_frame_pose__"
ERROR_PREFIX = "err__"


def resolve_runtime_result_dir(run_name: str, runtime_root: str | Path = "outputs") -> Path:
    """Resolve one standard-layout runtime name to its retargeting artifact directory.

    Args:
        run_name: Single runtime directory name created by ``app=offline_retarget``.
        runtime_root: Root directory containing runtime directories.

    Returns:
        Directory containing the runtime's ``result.npz`` and ``metadata.yaml`` files.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty runtime directory name.")
    runtime_path = Path(run_name)
    if runtime_path.name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name, not a path.")
    return resolve_project_path(runtime_root) / run_name / "retargeting"


@dataclass
class RetargetingTrajectory:
    frame_indices: np.ndarray
    retarget_qpos: np.ndarray
    hand_keypoints_wrist: np.ndarray
    hand_keypoints_world: np.ndarray
    wrist_pose_world: np.ndarray
    robot_frame_poses: dict[str, np.ndarray] = field(default_factory=dict)
    errors: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Return the number of stored trajectory frames.

        Args:
            None.

        Returns:
            Number of trajectory frames.
        """
        return int(self.frame_indices.shape[0])

    @property
    def qpos_dim(self) -> int:
        """Return the retargeted qpos dimension.

        Args:
            None.

        Returns:
            Number of actuated qpos values per frame.
        """
        return int(self.retarget_qpos.shape[1])

    def validate(self) -> None:
        """Validate trajectory array shapes.

        Args:
            None.

        Returns:
            None.
        """
        n_frames = self.n_frames
        if self.frame_indices.ndim != 1:
            raise ValueError("frame_indices must be a 1D array.")
        if self.retarget_qpos.ndim != 2 or self.retarget_qpos.shape[0] != n_frames:
            raise ValueError("retarget_qpos must have shape (n_frames, qpos_dim).")
        if self.hand_keypoints_wrist.shape != (n_frames, 21, 3):
            raise ValueError("hand_keypoints_wrist must have shape (n_frames, 21, 3).")
        if self.hand_keypoints_world.shape != (n_frames, 21, 3):
            raise ValueError("hand_keypoints_world must have shape (n_frames, 21, 3).")
        if self.wrist_pose_world.shape != (n_frames, 4, 4):
            raise ValueError("wrist_pose_world must have shape (n_frames, 4, 4).")
        for frame_name, poses in self.robot_frame_poses.items():
            if poses.shape != (n_frames, 4, 4):
                raise ValueError(f"robot_frame_poses[{frame_name!r}] must have shape (n_frames, 4, 4).")
        for error_name, values in self.errors.items():
            if values.shape[0] != n_frames:
                raise ValueError(f"errors[{error_name!r}] first dimension must match n_frames.")


@dataclass
class RetargetingRunMetadata:
    source_data: str
    robot_config: dict[str, Any]
    retargeting_config: dict[str, Any] | None
    start: int
    end: int
    stride: int
    num_frames: int
    qpos_dim: int
    schema_version: int = RESULT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    command: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetargetingRunMetadata":
        """Build metadata from loaded YAML data.

        Args:
            data: Plain mapping loaded from metadata.yaml.

        Returns:
            A typed metadata object.
        """
        if int(data.get("schema_version", 0)) != RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported result schema_version: {data.get('schema_version')}")
        return cls(
            schema_version=int(data["schema_version"]),
            created_at=str(data["created_at"]),
            source_data=str(data["source_data"]),
            robot_config=dict(data["robot_config"]),
            retargeting_config=None if data.get("retargeting_config") is None else dict(data["retargeting_config"]),
            start=int(data["start"]),
            end=int(data["end"]),
            stride=int(data["stride"]),
            num_frames=int(data["num_frames"]),
            qpos_dim=int(data["qpos_dim"]),
            command=None if data.get("command") is None else [str(item) for item in data["command"]],
            extra=dict(data.get("extra", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to YAML-serializable plain data.

        Args:
            None.

        Returns:
            Plain dictionary suitable for yaml.safe_dump().
        """
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source_data": self.source_data,
            "robot_config": _to_yaml_data(self.robot_config),
            "retargeting_config": _to_yaml_data(self.retargeting_config),
            "start": self.start,
            "end": self.end,
            "stride": self.stride,
            "num_frames": self.num_frames,
            "qpos_dim": self.qpos_dim,
            "command": self.command,
            "extra": _to_yaml_data(self.extra),
        }


def _encode_name(name: str) -> str:
    """Encode a logical frame or metric name for use as an npz key suffix.

    Args:
        name: Frame or metric name.

    Returns:
        URL-quoted name that can be embedded in an npz key.
    """
    return quote(name, safe="")


def _to_yaml_data(data: Any) -> Any:
    """Convert nested config data to YAML-safe built-in containers.

    Args:
        data: Nested config data containing mappings, sequences, tuples, or scalars.

    Returns:
        Data using only dict, list, and scalar containers.
    """
    data = to_plain_config_data(data)
    if isinstance(data, dict):
        return {str(key): _to_yaml_data(value) for key, value in data.items()}
    if isinstance(data, tuple) or isinstance(data, list):
        return [_to_yaml_data(item) for item in data]
    return data


def _decode_name(name: str) -> str:
    """Decode a frame or metric name from an npz key suffix.

    Args:
        name: URL-quoted npz key suffix.

    Returns:
        Decoded frame or metric name.
    """
    return unquote(name)


def resolve_result_paths(result_dir_or_file: str | Path) -> tuple[Path, Path, Path]:
    """Resolve a result directory or result.npz path to standard artifact paths.

    Args:
        result_dir_or_file: Output directory, result directory, or direct result.npz path.

    Returns:
        Tuple of (result_dir, result_npz_path, metadata_yaml_path).
    """
    path = Path(result_dir_or_file)
    if path.suffix == ".npz":
        result_file = path
        result_dir = path.parent
    else:
        result_dir = path
        result_file = result_dir / RESULT_FILE_NAME
    return result_dir, result_file, result_dir / METADATA_FILE_NAME


def save_retargeting_trajectory(
    output_dir: str | Path,
    trajectory: RetargetingTrajectory,
    metadata: RetargetingRunMetadata,
) -> Path:
    """Save a retargeting trajectory artifact to disk.

    Args:
        output_dir: Directory where result.npz and metadata.yaml should be written.
        trajectory: Retargeting trajectory arrays to persist.
        metadata: Run metadata associated with the trajectory.

    Returns:
        Path to the created output directory.
    """
    trajectory.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = replace(metadata, num_frames=trajectory.n_frames, qpos_dim=trajectory.qpos_dim)

    arrays: dict[str, np.ndarray] = {
        "frame_idx": np.asarray(trajectory.frame_indices, dtype=np.int64),
        "retarget_qpos": np.asarray(trajectory.retarget_qpos, dtype=float),
        "hand_keypoints_wrist": np.asarray(trajectory.hand_keypoints_wrist, dtype=float),
        "hand_keypoints_world": np.asarray(trajectory.hand_keypoints_world, dtype=float),
        "wrist_pose_world": np.asarray(trajectory.wrist_pose_world, dtype=float),
    }
    for frame_name, poses in trajectory.robot_frame_poses.items():
        arrays[f"{ROBOT_FRAME_POSE_PREFIX}{_encode_name(frame_name)}"] = np.asarray(poses, dtype=float)
    for error_name, values in trajectory.errors.items():
        arrays[f"{ERROR_PREFIX}{_encode_name(error_name)}"] = np.asarray(values)

    np.savez_compressed(output_dir / RESULT_FILE_NAME, **arrays)
    (output_dir / METADATA_FILE_NAME).write_text(
        yaml.safe_dump(metadata.to_dict(), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return output_dir


def load_retargeting_metadata(result_dir_or_file: str | Path) -> RetargetingRunMetadata:
    """Load metadata.yaml from a saved retargeting artifact.

    Args:
        result_dir_or_file: Result directory or direct result.npz path.

    Returns:
        Parsed metadata object.
    """
    _, _, metadata_file = resolve_result_paths(result_dir_or_file)
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping metadata in {metadata_file}.")
    return RetargetingRunMetadata.from_dict(data)


def load_retargeting_trajectory(result_dir_or_file: str | Path) -> tuple[RetargetingTrajectory, RetargetingRunMetadata]:
    """Load a saved retargeting trajectory artifact.

    Args:
        result_dir_or_file: Result directory or direct result.npz path.

    Returns:
        Tuple of loaded trajectory arrays and metadata.
    """
    _, result_file, _ = resolve_result_paths(result_dir_or_file)
    metadata = load_retargeting_metadata(result_dir_or_file)
    with np.load(result_file) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    robot_frame_poses = {
        _decode_name(key.removeprefix(ROBOT_FRAME_POSE_PREFIX)): value
        for key, value in arrays.items()
        if key.startswith(ROBOT_FRAME_POSE_PREFIX)
    }
    errors = {
        _decode_name(key.removeprefix(ERROR_PREFIX)): value
        for key, value in arrays.items()
        if key.startswith(ERROR_PREFIX)
    }
    trajectory = RetargetingTrajectory(
        frame_indices=np.asarray(arrays["frame_idx"], dtype=np.int64),
        retarget_qpos=np.asarray(arrays["retarget_qpos"], dtype=float),
        hand_keypoints_wrist=np.asarray(arrays["hand_keypoints_wrist"], dtype=float),
        hand_keypoints_world=np.asarray(arrays["hand_keypoints_world"], dtype=float),
        wrist_pose_world=np.asarray(arrays["wrist_pose_world"], dtype=float),
        robot_frame_poses=robot_frame_poses,
        errors=errors,
    )
    trajectory.validate()
    if metadata.num_frames != trajectory.n_frames or metadata.qpos_dim != trajectory.qpos_dim:
        raise ValueError("metadata.yaml shape fields do not match result.npz arrays.")
    return trajectory, metadata
