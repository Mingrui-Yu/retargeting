import time

import numpy as np
import open3d as o3d
import trimesh


def sample_points_normals_from_mesh(
    mesh: trimesh.Trimesh,
    n_points: int,
    sample_method="",
    normal_method="interpolation",
) -> np.ndarray:
    """
    Sample points and normals of the points from the mesh.
    Returns:
        pointcloud: shape (N, 6), first three for positions and last three for normals.
    """
    if sample_method == "even":
        # the sampled points may be less than desired
        points, face_index = trimesh.sample.sample_surface_even(mesh, count=n_points)
    else:
        points, face_index = trimesh.sample.sample_surface(mesh, count=n_points)

    if normal_method == "face_normals":
        normals = mesh.face_normals[face_index]
    elif normal_method == "interpolation":
        # reference: https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179
        # a little faster , and probably more smooth
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[face_index], points=points
        )
        normals = trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[face_index]]
                * trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)
        )

    assert points.shape[0] == n_points

    return np.concatenate([points, normals], axis=1)


# -------------------------------------------------------------------------------


def test_different_normal_method():
    mesh_file = "/media/mingrui/NewDisk/dataset/DexYCB/models/002_master_chef_can/textured_simple.obj"
    mesh = trimesh.load(mesh_file, process=False, force="mesh", skip_materials=False)
    n_points = 1024

    points, face_index = trimesh.sample.sample_surface(mesh, count=n_points)

    for i in range(100):
        t1 = time.perf_counter()
        normals_1 = mesh.face_normals[face_index]
        print(f"Normal method 1 time cost: {time.perf_counter() - t1}")

        t1 = time.perf_counter()
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[face_index], points=points
        )
        normals_2 = trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[face_index]]
                * trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)
        )
        print(f"Normal method 2 time cost: {time.perf_counter() - t1}")

        err = np.linalg.norm(normals_2 - normals_1, axis=1)
        print(f"Error between two normal method: {err.mean()}")


if __name__ == "__main__":
    test_different_normal_method()
