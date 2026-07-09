import numpy as np
import open3d as o3d


def o3d_vis_points(points: np.ndarray):
    """
    Args:
        points: shape (n_points, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coordinate_frame_o3d = create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([pcd, coordinate_frame_o3d])


def o3d_vis_pointcloud_with_color(pointcloud: np.ndarray):
    """
    Args:
        pointcloud: shape (n_points, 6), positions + colors (0~1)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[..., :3])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud[..., 3:])
    coordinate_frame_o3d = create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([pcd, coordinate_frame_o3d])


def create_coordinate_frame(size=1.0):
    # Define the 3D coordinates for the axes
    axes = np.array(
        [
            [0, 0, 0],  # Origin (0, 0, 0)
            [size, 0, 0],  # X axis (size, 0, 0)
            [0, size, 0],  # Y axis (0, size, 0)
            [0, 0, size],  # Z axis (0, 0, size)
        ]
    )

    # Define the color of the axes (RGB)
    colors = [
        [1, 0, 0],  # Red for X axis
        [0, 1, 0],  # Green for Y axis
        [0, 0, 1],  # Blue for Z axis
    ]

    # Create a LineSet object to draw the axes
    lines = [
        [0, 1],  # X axis (from origin to (size, 0, 0))
        [0, 2],  # Y axis (from origin to (0, size, 0))
        [0, 3],  # Z axis (from origin to (0, 0, size))
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(axes)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set the colors for the lines
    line_set.colors = o3d.utility.Vector3dVector([colors[0], colors[1], colors[2]])

    return line_set


def rgbd_to_pointcloud_by_open3d(rgb_image, depth_image, intrinsic_matrix):
    """
    Create a point cloud from an RGB image and a depth image.

    Args:
    - rgb_image: The RGB image as a (H, W, 3) numpy array.
    - depth_image: The depth image as a (H, W) numpy array. unit: m.
    - intrinsic_matrix: Camera intrinsic matrix as a (3, 3) numpy array.

    Returns:
    - point_cloud: shape (n_points, 6), positions + colors (0~1)
    """
    # Convert numpy arrays to Open3D Image objects
    rgb_o3d = o3d.geometry.Image(rgb_image)  # Convert RGB to uint8
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))  # Depth in float32

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
    )

    # Camera intrinsic object from matrix
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        rgb_image.shape[1],
        rgb_image.shape[0],
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )

    # Create the point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    return np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
