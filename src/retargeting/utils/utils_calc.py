import math
import time

import numpy as np
from scipy.spatial.transform import Rotation as sciR


# ---------------------------------------
def getUniformRandomDouble(lb, ub):
    return lb + (ub - lb) * np.random.rand()


# ---------------------------------------
def getGaussianRandomDouble(mean, sigma):
    return mean + sigma * np.random.randn()


# ---------------------------------------
def twoVecAngle(vec0, vec1):
    return np.arctan2(np.linalg.norm(np.cross(vec0, vec1)), np.dot(vec0, vec1))


# ---------------------------------------
def quatWXYZ2XYZW(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz)
    original_shape = quat_wxyz.shape
    quat_wxyz = quat_wxyz.reshape(-1, 4)

    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    return quat_xyzw.reshape(original_shape)


# ---------------------------------------
def quatXYZW2WXYZ(quat_xyzw):
    quat_xyzw = np.array(quat_xyzw)
    original_shape = quat_xyzw.shape
    quat_xyzw = quat_xyzw.reshape(-1, 4)

    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return quat_wxyz.reshape(original_shape)


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters:
        q (array_like): Quaternion in the form [w, x, y, z].

    Returns:
        R (ndarray): 3x3 rotation matrix.
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)

    # Extract quaternion components
    x, y, z, w = q

    # Compute rotation matrix
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )

    return R


# ---------------------------------------
def posQuat2Isometry3d(pos, quat):
    # quat: [x, y, z, w]
    pos = np.array(pos)
    rot_mat = sciR.from_quat(quat).as_matrix()
    isometry3d = np.block([[rot_mat, pos.reshape(3, 1)], [np.zeros((1, 3)), 1]])
    return isometry3d


# ---------------------------------------
# quat: [x, y, z, w]
def batchPosQuat2Isometry3d(pos, quat):
    pos = np.asarray(pos).reshape(-1, 3)
    quat = np.asarray(quat).reshape(-1, 4)
    rot_mat = sciR.from_quat(quat).as_matrix()

    isometry3d = np.zeros((pos.shape[0], 4, 4))
    isometry3d[:, 0:3, 3] = pos
    isometry3d[:, 0:3, 0:3] = rot_mat
    isometry3d[:, 3, 3] = 1

    return isometry3d


# ---------------------------------------
def batchPosRotVec2Isometry3d(pos, rotvec):
    pos = np.asarray(pos).reshape(-1, 3)
    rotvec = np.asarray(rotvec).reshape(-1, 3)
    rot_mat = sciR.from_rotvec(rotvec).as_matrix()

    isometry3d = np.zeros((pos.shape[0], 4, 4))
    isometry3d[:, 0:3, 3] = pos
    isometry3d[:, 0:3, 0:3] = rot_mat
    isometry3d[:, 3, 3] = 1

    return isometry3d


# ---------------------------------------
def batchIsometry3dInverse(isometry3d):
    isometry3d = np.asarray(isometry3d).reshape(-1, 4, 4)
    pos = isometry3d[:, 0:3, 3].reshape(-1, 3, 1)
    rot_mat = isometry3d[:, 0:3, 0:3]
    rot_mat_inv = np.transpose(rot_mat, [0, 2, 1])
    pos_inv = -np.matmul(rot_mat_inv, pos)

    isometry3d_inv = np.zeros((isometry3d.shape[0], 4, 4))
    isometry3d_inv[:, 0:3, 3] = np.squeeze(pos_inv)
    isometry3d_inv[:, 0:3, 0:3] = rot_mat_inv
    isometry3d_inv[:, 3, 3] = 1
    return isometry3d_inv


# ---------------------------------------
def posRotMat2Isometry3d(pos, rot_mat):
    isometry3d = np.block(
        [[rot_mat, np.asarray(pos).reshape(3, 1)], [np.zeros((1, 3)), 1]]
    )
    return isometry3d


# ---------------------------------------
def posOri2Isometry3d(pos, ori):
    rot_mat = sciR.as_matrix(ori)
    isometry3d = np.block([[rot_mat, pos.reshape(3, 1)], [np.zeros((1, 3)), 1]])
    return isometry3d


# ---------------------------------------
def isometry3dToPosQuat(T):
    if T.shape[0] != 4 or T.shape[1] != 4:  # noqa: PLR2004
        raise NameError("invalid input.")
    pos = T[0:3, 3].reshape(
        -1,
    )
    R = T[0:3, 0:3]
    quat = sciR.from_matrix(R).as_quat()  # quat: [x, y, z, w]
    return pos, quat


# ---------------------------------------
def isometry3dToPosOri(T):
    if T.shape[0] != 4 or T.shape[1] != 4:  # noqa: PLR2004
        raise NameError("invalid input.")
    pos = T[0:3, 3].reshape(
        -1,
    )
    R = T[0:3, 0:3]
    return pos, sciR.from_matrix(R)


# ---------------------------------------
def isometry3dToPosRotVec(T):
    if T.shape[0] != 4 or T.shape[1] != 4:  # noqa: PLR2004
        raise NameError("invalid input.")
    pos = T[0:3, 3].reshape(
        -1,
    )
    R = T[0:3, 0:3]
    return pos, sciR.from_matrix(R).as_rotvec()


# ---------------------------------------
def transformPositions(positions, target_frame_pose=None, target_frame_pose_inv=None):
    """
    input:
        positions: size of [-1, 3]
        target_frame_pose:
            matrix with size of [4, 4]
            target_frame_pose in current frame
        target_frame_pose_inv:
            current frame pose in target frame
    output:
        transformed_pos: size of [-1, 3]
    """

    if (target_frame_pose is None) and (target_frame_pose_inv is None):
        raise NameError("Both target_frame_pose and target_frame_pose_inv are None !")
    elif (target_frame_pose is not None) and (target_frame_pose_inv is not None):
        raise NameError(
            "Both target_frame_pose and target_frame_pose_inv are not None !"
        )

    positions = np.array(positions)
    original_shape = positions.shape
    argument_pos = positions.reshape(-1, 3)
    argument_pos = np.hstack([argument_pos, np.ones((argument_pos.shape[0], 1))])

    if target_frame_pose is not None:
        res = np.dot(np.linalg.inv(target_frame_pose), argument_pos.T)
    elif target_frame_pose_inv is not None:
        res = np.dot(target_frame_pose_inv, argument_pos.T)

    transformed_pos = (res.T[:, 0:3]).reshape(original_shape)
    return transformed_pos


# ---------------------------------------
def transformPoses(poses, target_frame_pose=None, target_frame_pose_inv=None):
    """
    Args:
        poses: shape (..., 4, 4) or (4, 4)
        target_frame_pose: target_frame_pose in current frame, shape (4, 4)
    Retures:
        transformed_poses: shape (..., 4, 4) or (4, 4)
    """
    if (target_frame_pose is None) and (target_frame_pose_inv is None):
        raise NameError("Both target_frame_pose and target_frame_pose_inv are None !")
    elif (target_frame_pose is not None) and (target_frame_pose_inv is not None):
        raise NameError(
            "Both target_frame_pose and target_frame_pose_inv are not None !"
        )

    poses = np.array(poses)
    original_shape = poses.shape
    poses = poses.reshape(-1, 4, 4)

    if target_frame_pose is not None:
        target_frame_pose_inv = np.linalg.inv(target_frame_pose)

    res = np.matmul(np.expand_dims(target_frame_pose_inv, axis=0), poses)

    transformed_poses = res.reshape(original_shape)
    return transformed_poses


# ---------------------------------------
def transformVelocities(
    velocities, target_frame_relative_quat=None, target_frame_relative_quat_inv=None
):
    """
    input:
        velocities:
            shape [-1, 6]
        target_frame_relative_quat:
            target frame's quaternion in current frame [x, y, z, w]
    output:
        transformed_velocities:
            shape [-1, 6]
    """

    if (target_frame_relative_quat is None) and (
        target_frame_relative_quat_inv is None
    ):
        raise NameError(
            "Both target_frame_relative_quat and target_frame_relative_quat_inv are None !"
        )
    elif (target_frame_relative_quat is not None) and (
        target_frame_relative_quat_inv is not None
    ):
        raise NameError(
            "Both target_frame_relative_quat and target_frame_relative_quat_inv are not None !"
        )

    velocities = np.array(velocities)
    original_shape = velocities.shape
    try:
        velocities = velocities.reshape(-1, 6)
    except:
        raise NameError("transformVelocities(): invalid input.")

    if target_frame_relative_quat is not None:
        rot_matrix = sciR.from_quat(target_frame_relative_quat).inv().as_matrix()
    elif target_frame_relative_quat_inv is not None:
        rot_matrix = sciR.from_quat(target_frame_relative_quat_inv).as_matrix()

    rot_operator = np.block(
        [[rot_matrix, np.zeros((3, 3))], [np.zeros((3, 3)), rot_matrix]]
    )

    transformed_velocities = (rot_operator @ velocities.T).T
    return transformed_velocities.reshape(original_shape)


def transformVectors(
    vectors, target_frame_relative_quat=None, target_frame_relative_quat_inv=None
):
    return transformVelocities(
        vectors, target_frame_relative_quat, target_frame_relative_quat_inv
    )


# ---------------------------------------
def diagRotMat(rot_mat):
    return np.block([[rot_mat, np.zeros((3, 3))], [np.zeros((3, 3)), rot_mat]])


# ---------------------------------------
def batchDiagRotMat(rot_mat):
    """
    Does not support non-batch operation.
    """
    diag_rot_mat = np.zeros((rot_mat.shape[0], 6, 6))
    diag_rot_mat[:, 0:3, 0:3] = rot_mat
    diag_rot_mat[:, 3:6, 3:6] = rot_mat
    return diag_rot_mat


# ---------------------------------------
"""
    support batch operation
"""


def skew(a):
    a = a.reshape(-1, 3)
    A = np.zeros((a.shape[0], 3, 3))
    A[:, 0, 1] = -a[:, 2]
    A[:, 0, 2] = a[:, 1]
    A[:, 1, 0] = a[:, 2]
    A[:, 1, 2] = -a[:, 0]
    A[:, 2, 0] = -a[:, 1]
    A[:, 2, 1] = a[:, 0]
    return A


def wrenchTransformationMatrix(a):
    """
    support batch operation
    """
    a = np.asarray(a).reshape(-1, 3)
    M = np.tile(np.eye(6), (a.shape[0], 1, 1))
    M[:, 0:3, 3:6] = -skew(a)
    return np.squeeze(M)


def jacoDeRotVecToAngularVel(rotvec):
    """
    Args:
        rotation vector
    Return:
        jacobian ( angular_velocity = jacobian @ derivative_rotation_vector )
    Support batch operation.
    """
    r = np.asarray(rotvec).reshape(-1, 3, 1)
    r_T = np.transpose(r, [0, 2, 1])

    n = r.shape[0]
    R = sciR.from_rotvec(r.reshape(-1, 3)).as_matrix()
    R_T = np.transpose(R, (0, 2, 1))
    I3 = np.tile(np.eye(3), (n, 1, 1))

    body_jaco = (np.matmul(r, r_T) + np.matmul((R_T - I3), skew(r))) / np.linalg.norm(
        r, axis=1, keepdims=True
    ) ** 2

    # avoid dividing by zero
    zero_index = np.where(np.linalg.norm(r.reshape(-1, 3), axis=1) < 1e-8)
    body_jaco[zero_index, :, :] = np.tile(np.eye(3), (len(zero_index), 1, 1))

    space_jaco = np.matmul(R, body_jaco)

    return np.squeeze(space_jaco)


def quatInv(quat):
    """
    input:
        quat: [w, x, y, z]
    """
    quat = np.array(quat)
    quat_inv = quat.copy()
    quat_inv[1:] = -quat[1:]
    return quat_inv


def partialQuatMultiply(quat):
    """
    Calculate J(q1) = d(q1 * q2) / dq2
    Args:
        q1: [w, x, y, z]
    Return:
        The J
    """
    w, x, y, z = quat

    J = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
    return J


def mappingFromAvelToDquat(quat):
    """
    Function:
        calculate M, where dq/dt = M(q) * avel_in_body_frame
    Input:
        q: [w, x, y, z]
    """
    w, x, y, z = quat

    M = 1.0 / 2.0 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
    return M


def mapping_from_space_avel_to_dquat(quat):
    """
    Function:
        calculate M, where dq/dt = M(q) * avel_in_space_frame
    Input:
        q: [w, x, y, z]
    """
    return (
        mappingFromAvelToDquat(quat) @ sciR.from_quat(quatWXYZ2XYZW(quat)).as_matrix().T
    )


def normalize_angle(angle):
    """
    Return the equivalent angle in (-pi, pi].
    """
    angle = np.rad2deg(angle)
    angle = (angle + 180) % 360 - 180
    return np.deg2rad(angle)


def jacoLeftBCH(rotvec):
    """
    Function:
        J_l(), which satisfies exp((phi + delta_phi)^) = exp((J_l(phi) @ delta_phi)^) @ exp((phi)^).
        Reference: visual SLAM 14 lectures (in Chinese), page 82.
        Support batch operation.
        Equivalent to jacoDeRotVecToAngularVel()
    """
    epsilon = 1e-8  # if the norm of rotvec is less than epsilon, we regard it as a zero rotvec

    r = np.asarray(rotvec).reshape(-1, 3)
    angle = np.linalg.norm(r, axis=1)
    zero_index = np.where(angle < epsilon)  # dealing with zero rotvec
    angle[zero_index] = epsilon

    angle = angle.reshape(-1, 1, 1)
    axis = r.reshape(-1, 3, 1) / angle

    I3 = np.tile(np.eye(3), (r.shape[0], 1, 1))

    J_l = (
        np.sin(angle) / angle * I3
        + (1.0 - np.sin(angle) / angle) * np.matmul(axis, axis.transpose(0, 2, 1))
        + ((1.0 - np.cos(angle)) / angle) * skew(axis)
    )

    J_l[zero_index, ...] = np.tile(
        np.eye(3), (len(zero_index), 1, 1)
    )  # dealing with zero rotvec

    return J_l.squeeze()


def jacoLeftBCHInverse(rotvec):
    """
    Function:
        Analytical inverse of jacoLeftBCH().
        Support batch operation.
    """
    epsilon = 1e-8  # if the norm of rotvec is less than epsilon, we regard it as a zero rotvec

    r = np.asarray(rotvec).reshape(-1, 3)
    angle = np.linalg.norm(r, axis=1)
    zero_index = np.where(angle < epsilon)  # dealing with zero rotvec
    angle[zero_index] = epsilon

    angle = angle.reshape(-1, 1, 1)
    axis = r.reshape(-1, 3, 1) / angle

    I3 = np.tile(np.eye(3), (r.shape[0], 1, 1))

    J_l_inv = (
        angle / 2.0 * 1.0 / np.tan(angle / 2.0) * I3
        + (1.0 - angle / 2.0 * 1.0 / np.tan(angle / 2.0))
        * np.matmul(axis, axis.transpose(0, 2, 1))
        - angle / 2.0 * skew(axis)
    )

    J_l_inv[zero_index, ...] = np.tile(
        np.eye(3), (len(zero_index), 1, 1)
    )  # dealing with zero rotvec

    return J_l_inv.squeeze()


def depth_image_to_points(depth_image, intrinsic_matrix):
    height, width = depth_image.shape
    # Create a meshgrid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    Z = depth_image.flatten()
    # Extract intrinsic matrix components
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    # Backproject pixels to 3D space
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # Stack the 3D points into an (N, 3) array
    points = np.vstack((X, Y, Z)).T
    return points


def points_from_image_to_cam_frame(img_coord, z, intrinsic_matrix):
    """
    Args:
        img_coord: (u, v)
        z: depth, unit: m.
    """
    return NotImplementedError()


def rgbd_to_pointcloud(rgb_image, depth_image, intrinsic_matrix):
    points = depth_image_to_points(depth_image, intrinsic_matrix)
    return np.hstack(
        [
            points,
            rgb_image.reshape(-1, 3).astype(points.dtype) / 255.0,  # (0~255) to (0~1)
        ]
    )


def camera_orientation_opengl_to_common(opengl_quat):
    """
    Args:
        opengl_quat: (w, x, y, z)
    Return:
        common_quat: (w, x, y, z)
    """

    # rotate 180 degree around its x-axis
    return quatXYZW2WXYZ(
        (
            sciR.from_quat(quatWXYZ2XYZW(opengl_quat))
            * sciR.from_euler("XYZ", [np.pi, 0, 0])
        ).as_quat()
    )


# ----------------------------------------------------------------


def test_mapping_from_avel_to_dquat():
    avel = np.array([0.1, -0.1, 0.5])  # avel in world frame
    dt = 0.01
    r1 = sciR.from_euler("xyz", [0.5, 0.6, 0.7])

    r2 = sciR.from_rotvec(avel * dt) * r1

    r1_quat = r1.as_quat()  # (x, y, z, w)
    r2_quat = r2.as_quat()  # (x, y, z, w)

    d_quat = (r2_quat - r1_quat) / dt  # (x, y, z, w)

    d_quat_estimate = mapping_from_space_avel_to_dquat(
        quatXYZW2WXYZ(r1_quat)
    ) @ avel.reshape(-1, 1)
    d_quat_estimate = quatWXYZ2XYZW(d_quat_estimate)

    print("d_quat: ", d_quat)
    print(
        "d_quat_estimate: ",
        d_quat_estimate.reshape(
            -1,
        ),
    )


# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    test_mapping_from_avel_to_dquat()
