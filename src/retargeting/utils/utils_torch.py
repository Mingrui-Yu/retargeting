import torch
import numpy as np
from .pytorch3d.rotation_conversions import *


def quaternion_xyzw2wxyz(q):
    """
    Args:
        q: shape (..., 4)
    """
    return q[:, [3, 0, 1, 2]]


def quaternion_wxyz2xyzw(q):
    """
    Args:
        q: shape (..., 4)
    """
    return q[:, [1, 2, 3, 0]]


def quaternion_angular_error(q1, q2, epsilon=1e-7):
    """
    angular error between two quaternions
    :param q1: (..., 4), torch; quat is (w, x, y, z)
    :param q2: (..., 4), torch
    :param epsilon: a small value to avoid numerial error when calculating the derivative.
    :return: (...,), torch
    """

    # normalize the quaternions
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    # compute the dot product
    dot_product = (
        torch.matmul(q1.unsqueeze(-2), q2.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )

    # compute the absolute value of the dot product
    abs_dot_product = torch.clamp(
        torch.abs(dot_product), min=-1.0 + epsilon, max=1.0 - epsilon
    )

    # compute the error
    error = 2.0 * torch.acos(abs_dot_product)

    return error


def get_random(shape, lower, upper):
    """
    Args:
        shape: a list or tuple
    """
    device = upper.device
    assert shape[1] == lower.shape[0] == upper.shape[0]
    return lower.unsqueeze(0) + torch.rand(shape).float().to(device) * (
        upper - lower
    ).unsqueeze(0)


def transform_points(points, frame_pos, frame_quat):
    """
    Args:
        points: shape (N, 3)
        frame_pos: current frame pos in target frame, shape (N, 3)
        frame_quat: current frame pos in target frame, shape (N, 4), (w, x, y, z)
    """
    return quaternion_apply(frame_quat, points) + frame_pos


def transform_points_inverse(points, frame_pos, frame_quat):
    """
    Args:
        points: shape (N, 3)
        frame_pos: target frame pos in current frame, shape (N, 3)
        frame_quat: target frame quat in current frame, shape (N, 4), (w, x, y, z)
    """
    return quaternion_apply(quaternion_invert(frame_quat), points - frame_pos)


# ----------------------------------------------------------------
def test_quaternion_angular_error():
    from scipy.spatial.transform import Rotation as sciR

    # q1 = sciR.from_euler('xyz', [0.1, -0.3, 0.2]).as_quat()
    # q2 = np.array([0, 1, 0, 0])

    q1 = sciR.from_euler("xyz", np.random.rand(10, 3)).as_quat()
    q2 = sciR.from_euler("xyz", np.random.rand(10, 3)).as_quat()

    # define two quaternions
    q1_torch = torch.tensor(q1[:, [3, 0, 1, 2]], dtype=torch.float32)
    q2_torch = torch.tensor(q2[:, [3, 0, 1, 2]], dtype=torch.float32)

    print("Angle by quaterions (torch):", quaternion_angular_error(q1_torch, q2_torch))

    delta_rot = sciR.from_quat(q2) * sciR.from_quat(q1).inv()

    print(
        "Angle by axis-angle vector (scipy):",
        np.linalg.norm(delta_rot.as_rotvec(), axis=1),
    )


def test_matrix_to_euler_angles():
    from scipy.spatial.transform import Rotation as sciR

    rand_euler = np.random.rand(10, 3)
    print("rand_euler_angles: ", rand_euler)
    matrix = sciR.from_euler("XYZ", rand_euler).as_matrix()
    matrix = torch.from_numpy(matrix)
    euler_angles = matrix_to_euler_angles(matrix, "XYZ")
    print("converted_euler_angles: ", euler_angles)


def test_transform_points():
    points_in_a = torch.rand(10, 3)
    a_quat_in_b = random_quaternions(10)
    a_pos_in_b = torch.rand(10, 3)

    points_in_b = transform_points(points_in_a, a_pos_in_b, a_quat_in_b)

    point_in_a_2 = transform_points_inverse(points_in_b, a_pos_in_b, a_quat_in_b)

    print(torch.norm(point_in_a_2 - points_in_a))


# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    test_transform_points()
