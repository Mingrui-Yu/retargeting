import numpy as np
from scipy.spatial.transform import Rotation as sciR

from retargeting_ros._optional import missing_ros_error


# ------------------------------------
def rosPoseToPosQuat(ros_pose):
    pos = np.zeros((3,))
    quat = np.zeros((4,))
    pos[0] = ros_pose.position.x
    pos[1] = ros_pose.position.y
    pos[2] = ros_pose.position.z
    quat[0] = ros_pose.orientation.x
    quat[1] = ros_pose.orientation.y
    quat[2] = ros_pose.orientation.z
    quat[3] = ros_pose.orientation.w

    return pos, quat


# ------------------------------------
def posQuatToRosPose(pos, quat):
    try:
        from geometry_msgs.msg import Pose, Quaternion
    except ModuleNotFoundError as e:
        raise missing_ros_error("posQuatToRosPose", e) from e

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
    pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    return pose


# ------------------------------------
def rosPoseToAxisMarker(
    ros_pose, parent_frame_id, x_marker_id=0, y_marker_id=1, z_marker_id=2
):
    try:
        from geometry_msgs.msg import Quaternion, Vector3
        from rclpy.clock import Clock
        from visualization_msgs.msg import Marker, MarkerArray
    except ModuleNotFoundError as e:
        raise missing_ros_error("rosPoseToAxisMarker", e) from e

    _, quat = rosPoseToPosQuat(ros_pose)
    rot = sciR.from_quat(quat)

    axis_scale = Vector3(x=0.05, y=0.005, z=0.005)

    x_axis = Marker()
    x_axis.id = x_marker_id
    x_axis.header.stamp = Clock().now().to_msg()
    x_axis.header.frame_id = parent_frame_id
    x_axis.type = Marker.ARROW
    x_axis.action = Marker.ADD
    x_axis.scale = axis_scale
    x_axis.color.a = 1.0
    x_axis.color.r = 1.0
    x_axis.color.g = 0.0
    x_axis.color.b = 0.0
    axis_quat = (rot * sciR.from_quat([0, 0, 0, 1])).as_quat()
    x_axis.pose.position = ros_pose.position
    x_axis.pose.orientation = Quaternion(
        x=axis_quat[0], y=axis_quat[1], z=axis_quat[2], w=axis_quat[3]
    )

    y_axis = Marker()
    y_axis.id = y_marker_id
    y_axis.header.stamp = Clock().now().to_msg()
    y_axis.header.frame_id = parent_frame_id
    y_axis.type = Marker.ARROW
    y_axis.action = Marker.ADD
    y_axis.scale = axis_scale
    y_axis.color.a = 1.0
    y_axis.color.r = 0.0
    y_axis.color.g = 1.0
    y_axis.color.b = 0.0
    axis_quat = (rot * sciR.from_quat([0.0, 0.0, 0.7071, 0.7071])).as_quat()
    y_axis.pose.position = ros_pose.position
    y_axis.pose.orientation = Quaternion(
        x=axis_quat[0], y=axis_quat[1], z=axis_quat[2], w=axis_quat[3]
    )

    z_axis = Marker()
    z_axis.id = z_marker_id
    z_axis.header.stamp = Clock().now().to_msg()
    z_axis.header.frame_id = parent_frame_id
    z_axis.type = Marker.ARROW
    z_axis.action = Marker.ADD
    z_axis.scale = axis_scale
    z_axis.color.a = 1.0
    z_axis.color.r = 0.0
    z_axis.color.g = 0.0
    z_axis.color.b = 1.0
    axis_quat = (rot * sciR.from_quat([0, -0.7071, 0, 0.7071])).as_quat()
    z_axis.pose.position = ros_pose.position
    z_axis.pose.orientation = Quaternion(
        x=axis_quat[0], y=axis_quat[1], z=axis_quat[2], w=axis_quat[3]
    )

    marker_array = MarkerArray()
    marker_array.markers = [x_axis, y_axis, z_axis]
    return marker_array


# ------------------------------------
def pointsToMarker(points, colors, parent_frame_id, marker_id=0):
    try:
        from geometry_msgs.msg import Point, Vector3
        from rclpy.clock import Clock
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker
    except ModuleNotFoundError as e:
        raise missing_ros_error("pointsToMarker", e) from e

    points_marker = Marker()
    points_marker.id = marker_id
    points_marker.header.stamp = Clock().now().to_msg()
    points_marker.header.frame_id = parent_frame_id

    points_marker.type = Marker.POINTS
    points_marker.action = Marker.ADD
    points_marker.scale = Vector3(x=0.01, y=0.01, z=0.01)

    points_marker.pose.orientation.w = 1.0

    points_marker.color.r = 1.0
    points_marker.color.g = 1.0
    points_marker.color.b = 1.0
    points_marker.color.a = 1.0

    for i in range(points.shape[0]):
        points_marker.points.append(
            Point(x=points[i][0], y=points[i][1], z=points[i][2])
        )
        points_marker.colors.append(
            ColorRGBA(
                r=float(colors[i][0]),
                g=float(colors[i][1]),
                b=float(colors[i][2]),
                a=1.0,
            )
        )

    return points_marker


# ------------------------------------
def pointPairsToLines(points, parent_frame_id, marker_id=0):
    try:
        from geometry_msgs.msg import Point
        from rclpy.clock import Clock
        from visualization_msgs.msg import Marker
    except ModuleNotFoundError as e:
        raise missing_ros_error("pointPairsToLines", e) from e

    marker = Marker()
    marker.id = marker_id
    marker.header.stamp = Clock().now().to_msg()
    marker.header.frame_id = parent_frame_id

    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.005

    marker.pose.orientation.w = 1.0

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    for i in range(points.shape[0]):
        marker.points.append(Point(x=points[i][0], y=points[i][1], z=points[i][2]))

    return marker


def stamp_to_seconds(input) -> float:
    return input.sec + input.nanosec / 1e9


def time_to_seconds(input) -> float:
    return input.nanoseconds / 1e9


def seconds_to_stamp(seconds_float):
    try:
        from builtin_interfaces.msg import Time
    except ModuleNotFoundError as e:
        raise missing_ros_error("seconds_to_stamp", e) from e

    sec = int(seconds_float)  # Integer part for seconds
    nanosec = int((seconds_float - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)
