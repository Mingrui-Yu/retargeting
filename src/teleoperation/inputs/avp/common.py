"""Shared AVP protocol decoding for live and archived acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mr_utils.utils_mano import OPERATOR2MANO_RIGHT
from teleoperation.types import SensorHandSample


def decode_avp_sample(
    raw: Mapping[str, Any] | None,
    *,
    source_index: int | None = None,
    timestamp: float | None = None,
) -> SensorHandSample:
    """Decode one AVP stream frame into sensor-normalized MANO coordinates.

    Args:
        raw: Vision Pro stream mapping, or None for a frame without hand data.
        source_index: Optional zero-based acquisition index.
        timestamp: Optional source timestamp in seconds.

    Returns:
        Sensor sample whose empty hand fields represent a missing detection.
    """
    if raw is None:
        return SensorHandSample(
            keypoints_wrist=None,
            wrist_pose_sensor=None,
            raw=None,
            source_index=source_index,
            timestamp=timestamp,
        )
    missing = sorted({"right_wrist", "right_fingers"} - set(raw))
    if missing:
        raise ValueError(f"AVP frame is missing required streams: {missing}")
    if raw["right_wrist"] is None or raw["right_fingers"] is None:
        return SensorHandSample(
            keypoints_wrist=None,
            wrist_pose_sensor=None,
            raw=raw,
            source_index=source_index,
            timestamp=timestamp,
        )

    wrist_pose = np.asarray(raw["right_wrist"], dtype=float).reshape(4, 4)
    finger_poses = np.asarray(raw["right_fingers"], dtype=float)
    if finger_poses.shape != (25, 4, 4):
        raise ValueError(f"AVP right_fingers must have shape (25, 4, 4), got {finger_poses.shape}.")
    if not np.isfinite(wrist_pose).all() or not np.isfinite(finger_poses).all():
        raise ValueError("AVP wrist and finger transforms must contain only finite values.")

    keypoint_poses_sensor = wrist_pose.reshape(1, 4, 4) @ finger_poses
    wrist_pose_mano = wrist_pose.copy()
    wrist_pose_mano[:3, :3] = wrist_pose_mano[:3, :3] @ OPERATOR2MANO_RIGHT
    keypoint_poses_wrist = np.linalg.inv(wrist_pose_mano).reshape(1, 4, 4) @ keypoint_poses_sensor
    visionpro_to_mediapipe = np.asarray(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
        dtype=int,
    )
    keypoints_wrist = keypoint_poses_wrist[visionpro_to_mediapipe, :3, 3]
    return SensorHandSample(
        keypoints_wrist=keypoints_wrist,
        wrist_pose_sensor=wrist_pose_mano,
        raw=raw,
        source_index=source_index,
        timestamp=timestamp,
    )
