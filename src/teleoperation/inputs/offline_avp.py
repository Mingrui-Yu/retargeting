"""Raw AVP trajectory loader and frame-index utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import numpy as np


@dataclass(frozen=True)
class OfflineAvpTrajectory:
    """Raw per-frame AVP streams loaded without robot qpos arrays."""

    stream_arrays: Dict[str, np.ndarray]
    source: Path

    @property
    def n_frames(self) -> int:
        """Return the number of raw AVP frames.

        Args:
            None.

        Returns:
            Number of frames shared by all stream arrays.
        """
        first_stream = next(iter(self.stream_arrays.values()))
        return int(first_stream.shape[0])

    def get_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return one raw AVP frame in detector-compatible mapping form.

        Args:
            frame_idx: Zero-based source frame index.

        Returns:
            Mapping from raw AVP stream name to copied frame data.
        """
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(f"frame_idx must be in [0, {self.n_frames}), got {frame_idx}.")
        return {name: values[frame_idx].copy() for name, values in self.stream_arrays.items()}


def load_offline_avp_trajectory(file_name: str | Path) -> OfflineAvpTrajectory:
    """Load only raw ``stream_*`` arrays from an offline AVP trajectory.

    Args:
        file_name: NPZ file containing frame-aligned raw AVP stream arrays.

    Returns:
        AVP trajectory that never loads an existing ``retarget_qpos`` array.
    """
    source = Path(file_name)
    with np.load(source) as loaded:
        stream_keys = [key for key in loaded.files if key.startswith("stream_")]
        if not stream_keys:
            raise ValueError(f"Offline AVP trajectory has no stream_* arrays: {source}")
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
        raise ValueError(f"Offline AVP stream arrays have inconsistent frame counts: {frame_counts}")
    if next(iter(unique_frame_counts)) <= 0:
        raise ValueError("Offline AVP trajectory must contain at least one frame.")
    return OfflineAvpTrajectory(stream_arrays=stream_arrays, source=source)


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
