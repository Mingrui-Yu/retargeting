import numpy as np

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

# https://ai.google.dev/static/mediapipe/images/solutions/hand-landmarks.png?hl=zh-cn
MANO_FINGERTIP_INDEX = [4, 8, 12, 16, 20]

MANO_LINE_PAIRS = [
    [0, 1],
    [0, 5],
    [0, 17],
    [5, 17],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20],
]

MANO_POINTS_COLORS = [
    [48, 48, 255],
    [48, 48, 255],
    [180, 229, 255],
    [180, 229, 255],
    [180, 229, 255],
    [48, 48, 255],
    [128, 64, 128],
    [128, 64, 128],
    [128, 64, 128],
    [48, 48, 255],
    [0, 204, 255],
    [0, 204, 255],
    [0, 204, 255],
    [48, 48, 255],
    [48, 255, 48],
    [48, 255, 48],
    [48, 255, 48],
    [48, 48, 255],
    [192, 101, 21],
    [192, 101, 21],
    [192, 101, 21],
]


def estimate_wrist_pose_from_hand_points(
    keypoints_3d: np.ndarray, hand_type: str
) -> np.ndarray:
    """
    Args:
        keypoints_3d: mano representation
        hand_type: 'right' or 'left'
    """
    rot = estimate_frame_from_hand_points(keypoints_3d)
    rot_mano = (
        rot @ OPERATOR2MANO_RIGHT if hand_type == "right" else rot @ OPERATOR2MANO_LEFT
    )

    wrist_pose = np.eye(4)
    wrist_pose[:3, :3] = rot_mano
    wrist_pose[:3, 3] = keypoints_3d[0, :]
    return wrist_pose


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    Args:
        keypoint_3d_array: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    Return:
        orientation: the coordinate frame (orientation only) of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    orientation = np.stack([x, normal, z], axis=1)
    return orientation
