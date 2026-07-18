"""Finite archived AVP input with explicit lifecycle and selection semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from teleoperation.inputs.avp.common import decode_avp_sample
from teleoperation.types import SensorHandSample


def _normalize_end_index(end: int, n_frames: int) -> int:
    """Clamp an inclusive selection end index to an archive.

    Args:
        end: Requested inclusive end index; negative selects the final frame.
        n_frames: Number of frames in the archive.

    Returns:
        Inclusive in-range end index.
    """
    return n_frames - 1 if end < 0 else min(end, n_frames - 1)


def select_frame_indices(n_frames: int, start: int = 0, end: int = -1, stride: int = 1) -> tuple[int, ...]:
    """Select finite source indices using inclusive end semantics.

    Args:
        n_frames: Total source frame count.
        start: First selected source index.
        end: Last selected source index, inclusive; negative selects the final frame.
        stride: Positive interval between selected frames.

    Returns:
        Selected source indices in acquisition order.
    """
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    if start >= n_frames:
        return ()
    normalized_end = _normalize_end_index(end, n_frames)
    return tuple(range(start, normalized_end + 1, stride))


class AvpOfflineInput:
    """Read selected raw AVP archive frames through the shared decoder."""

    def __init__(self, file_name: str | Path, start: int = 0, end: int = -1, stride: int = 1) -> None:
        """Load raw stream arrays and configure a finite selection.

        Args:
            file_name: NPZ archive containing frame-aligned ``stream_*`` arrays.
            start: First selected source index.
            end: Last selected source index, inclusive; negative selects the end.
            stride: Positive interval between selected frames.

        Returns:
            None.
        """
        self.source = Path(file_name)
        with np.load(self.source) as loaded:
            stream_keys = [key for key in loaded.files if key.startswith("stream_")]
            if not stream_keys:
                raise ValueError(f"Offline AVP trajectory has no stream_* arrays: {self.source}")
            self._stream_arrays = {
                key.removeprefix("stream_"): np.asarray(loaded[key]).copy() for key in stream_keys
            }
        required_streams = {"right_wrist", "right_fingers"}
        missing_streams = sorted(required_streams - set(self._stream_arrays))
        if missing_streams:
            raise ValueError(f"Offline AVP trajectory is missing required streams: {missing_streams}")
        frame_counts = {name: values.shape[0] for name, values in self._stream_arrays.items()}
        if len(set(frame_counts.values())) != 1:
            raise ValueError(f"Offline AVP stream arrays have inconsistent frame counts: {frame_counts}")
        if next(iter(frame_counts.values())) <= 0:
            raise ValueError("Offline AVP trajectory must contain at least one frame.")
        self.frame_indices = select_frame_indices(self.n_frames, start=start, end=end, stride=stride)
        if not self.frame_indices:
            raise ValueError(
                f"No offline AVP frames selected from {self.source}: start={start}, end={end}, stride={stride}."
            )
        self._cursor = 0
        self._is_open = False

    @property
    def n_frames(self) -> int:
        """Return the full archive frame count before selection.

        Args:
            None.

        Returns:
            Number of frames shared by all raw stream arrays.
        """
        return int(next(iter(self._stream_arrays.values())).shape[0])

    def open(self) -> None:
        """Open the in-memory archive input at the first selected frame.

        Args:
            None.

        Returns:
            None.
        """
        self._cursor = 0
        self._is_open = True

    def read(self) -> SensorHandSample | None:
        """Decode the next selected frame or report finite end-of-stream.

        Args:
            None.

        Returns:
            Decoded AVP sample, or None after the final selected frame.
        """
        if not self._is_open:
            raise RuntimeError("AvpOfflineInput must be opened before read().")
        if self._cursor >= len(self.frame_indices):
            return None
        frame_index = self.frame_indices[self._cursor]
        self._cursor += 1
        raw = {name: values[frame_index].copy() for name, values in self._stream_arrays.items()}
        return decode_avp_sample(raw, source_index=frame_index)

    def reset(self) -> None:
        """Rewind the selected archive interval for a new playback cycle.

        Args:
            None.

        Returns:
            None.
        """
        self._cursor = 0

    def close(self) -> None:
        """Mark the in-memory archive input as closed.

        Args:
            None.

        Returns:
            None.
        """
        self._is_open = False
