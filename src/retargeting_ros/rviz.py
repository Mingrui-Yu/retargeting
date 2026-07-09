import time

import numpy as np
from retargeting.utils.utils_calc import isometry3dToPosQuat, transformPositions
from retargeting.utils.utils_mano import MANO_LINE_PAIRS, MANO_POINTS_COLORS
from retargeting_ros.messages import (
    pointPairsToLines,
    pointsToMarker,
    posQuatToRosPose,
    rosPoseToAxisMarker,
)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray
except ModuleNotFoundError as _ROS_IMPORT_ERROR:
    rclpy = None
    Node = object
    JointState = None
    Header = None
    Marker = None
    MarkerArray = None
else:
    _ROS_IMPORT_ERROR = None


def _require_ros_support() -> None:
    if _ROS_IMPORT_ERROR is not None:
        from retargeting_ros._optional import missing_ros_error

        raise missing_ros_error("RvizVisualizer", _ROS_IMPORT_ERROR)


class RvizVisualizer:
    def __init__(self, node: Node):
        _require_ros_support()
        self.hand_keypoints_colors = []

        self.node = node
        self.hand_keypoints_marker_pub = node.create_publisher(Marker, "visualize/hand_keypoints", 10)
        self.hand_connections_marker_pub = node.create_publisher(Marker, "visualize/hand_connections", 10)
        self.wrist_frame_marker_pub = node.create_publisher(MarkerArray, "visualize/wrist_frame", 30)
        self.joint_state_pub = node.create_publisher(JointState, "visualize/robot_joint_states", 10)

        time.sleep(0.1)

    # ------------------------------------
    def publish_hand_detection_results(
        self,
        joint_pos_in_wrist: np.ndarray,
        wrist_pose: np.ndarray,
        frame_id: str,
    ):
        pos, quat = isometry3dToPosQuat(wrist_pose)
        pose = posQuatToRosPose(pos, quat)

        # ----------- publish keypoints -----------
        joint_pos_in_cam = transformPositions(joint_pos_in_wrist, target_frame_pose_inv=wrist_pose)

        self.hand_keypoints_marker_pub.publish(
            pointsToMarker(
                joint_pos_in_cam,
                MANO_POINTS_COLORS,
                frame_id,
                marker_id=3,
            )
        )

        # ----------- publish keypoint connections -----------
        pairs = []
        for connection in MANO_LINE_PAIRS:
            pairs.append(joint_pos_in_cam[connection[0], :])
            pairs.append(joint_pos_in_cam[connection[1], :])
        pairs = np.array(pairs)

        self.hand_connections_marker_pub.publish(pointPairsToLines(pairs, frame_id, marker_id=4))

        # ----------- publish wrist frame ------------
        axis_markers = rosPoseToAxisMarker(pose, frame_id, 0, 1, 2)
        self.wrist_frame_marker_pub.publish(axis_markers)

    def publish_robot_joint_states(self, joints_pos, joints_name):
        assert len(joints_pos) == len(joints_name)

        joint_state_msg = JointState()
        # Set the header for the message (current time)
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = joints_name
        joint_state_msg.position = joints_pos.tolist()

        self.joint_state_pub.publish(joint_state_msg)


if __name__ == "__main__":
    _require_ros_support()
    rclpy.init()
    node = Node("node_name")
    visualizer = RvizVisualizer(node)
