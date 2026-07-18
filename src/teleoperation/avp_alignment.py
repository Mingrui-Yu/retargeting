"""Shared AVP alignment for teleoperation sessions and runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np

from teleoperation.inputs.avp import parse_avp_stream_frame
from teleoperation.session import TeleoperationSession


def initialize_avp_alignment(
    session: TeleoperationSession,
    sensor_data: Any,
    robot_qpos: np.ndarray,
) -> bool:
    """Initialize one session's relative wrist alignment from a raw AVP frame.

    Args:
        session: Teleoperation session whose input adapter requires alignment.
        sensor_data: Raw AVP stream frame.
        robot_qpos: Current robot positions in actuated-joint order.

    Returns:
        True when the frame contains a wrist pose and alignment was initialized.
    """
    _, _, _, detection_wrist_pose = parse_avp_stream_frame(sensor_data)
    if detection_wrist_pose is None:
        return False
    robot_model_qpos = session.robot_adaptor.forward_qpos(np.asarray(robot_qpos, dtype=float))
    robot_wrist_pose = session.robot_model.get_frame_pose(
        session.robot_config.wrist_frame_name,
        qpos=robot_model_qpos,
    )
    session.set_robot_init_wrist_pose(robot_wrist_pose)
    session.set_detection_source_init_wrist_pose(
        session.pose_from_detection_world_to_robot_world(detection_wrist_pose)
    )
    return True
