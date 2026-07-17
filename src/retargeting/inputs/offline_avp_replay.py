"""Raw AVP replay npz loader and frame-index utilities."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np


@dataclass
class OfflineReplay:
    streams: list
    arrays: Dict[str, np.ndarray]

    @property
    def n_frames(self) -> int:
        return len(self.streams)

    @property
    def retarget_qpos(self) -> Optional[np.ndarray]:
        return self.arrays.get("retarget_qpos")


@dataclass(frozen=True)
class OfflineHumanTrajectory:
    """Raw per-frame human streams loaded without robot qpos arrays."""

    stream_arrays: Dict[str, np.ndarray]
    source: Path

    @property
    def n_frames(self) -> int:
        """Return the number of raw human frames.

        Args:
            None.

        Returns:
            Number of frames shared by all stream arrays.
        """
        first_stream = next(iter(self.stream_arrays.values()))
        return int(first_stream.shape[0])

    def get_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return one raw human frame in detector-compatible mapping form.

        Args:
            frame_idx: Zero-based source frame index.

        Returns:
            Mapping from raw AVP stream name to copied frame data.
        """
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(f"frame_idx must be in [0, {self.n_frames}), got {frame_idx}.")
        return {name: values[frame_idx].copy() for name, values in self.stream_arrays.items()}


def rebuild_stream_data(data_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    """
    Rebuild the nested `stream` list from flat `stream_*` arrays saved in npz files.
    """
    prefix = "stream_"
    if "stream_left_wrist" not in data_dict:
        raise KeyError("Expected `stream_left_wrist` in replay data.")

    n_steps = data_dict["stream_left_wrist"].shape[0]
    rebuilt = {"stream": [{} for _ in range(n_steps)]}

    for key, value in data_dict.items():
        if key.startswith(prefix):
            stream_key = key[len(prefix) :]
            for step, array in enumerate(value):
                rebuilt["stream"][step][stream_key] = copy.deepcopy(array)
        else:
            rebuilt[key] = value

    return rebuilt


def load_offline_replay(file_name: str | Path) -> OfflineReplay:
    file_name = Path(file_name)
    loaded_data = np.load(file_name)
    arrays = {key: loaded_data[key] for key in loaded_data.files}
    rebuilt = rebuild_stream_data(arrays)
    return OfflineReplay(streams=rebuilt["stream"], arrays=arrays)


def load_offline_human_trajectory(file_name: str | Path) -> OfflineHumanTrajectory:
    """Load only raw human ``stream_*`` arrays from an offline AVP trajectory.

    Args:
        file_name: NPZ file containing frame-aligned raw AVP stream arrays.

    Returns:
        Human trajectory that never loads an existing ``retarget_qpos`` array.
    """
    source = Path(file_name)
    with np.load(source) as loaded:
        stream_keys = [key for key in loaded.files if key.startswith("stream_")]
        if not stream_keys:
            raise ValueError(f"Offline human trajectory has no stream_* arrays: {source}")
        stream_arrays = {
            key.removeprefix("stream_"): np.asarray(loaded[key]).copy() for key in stream_keys
        }
    required_streams = {"right_wrist", "right_fingers"}
    missing_streams = sorted(required_streams - set(stream_arrays))
    if missing_streams:
        raise ValueError(f"Offline AVP trajectory is missing required streams: {missing_streams}")
    frame_counts = {name: values.shape[0] for name, values in stream_arrays.items()}
    unique_frame_counts = set(frame_counts.values())
    if len(unique_frame_counts) != 1:
        raise ValueError(f"Offline human stream arrays have inconsistent frame counts: {frame_counts}")
    if next(iter(unique_frame_counts)) <= 0:
        raise ValueError("Offline human trajectory must contain at least one frame.")
    return OfflineHumanTrajectory(stream_arrays=stream_arrays, source=source)


def normalize_end_index(end: int, n_frames: int) -> int:
    if end < 0:
        return n_frames - 1
    return min(end, n_frames - 1)


def iter_frame_indices(n_frames: int, start: int = 0, end: int = -1, stride: int = 1) -> Iterator[int]:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    if start >= n_frames:
        return

    end = normalize_end_index(end, n_frames)
    for frame_idx in range(start, end + 1, stride):
        yield frame_idx
