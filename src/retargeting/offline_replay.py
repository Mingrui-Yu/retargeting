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
