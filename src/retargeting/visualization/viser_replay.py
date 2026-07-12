from pathlib import Path
import time
from typing import Any, List

import numpy as np
from retargeting.pipelines.offline_retargeting import (
    RetargetReplayFrame,
    RobotReplayContext,
    create_robot_replay_context_from_metadata,
    trajectory_to_replay_frames,
)
from retargeting.artifacts.trajectory import load_retargeting_trajectory
from scipy.spatial.transform import Rotation as sciR
from mr_utils.utils_mano import MANO_FINGERTIP_INDEX, MANO_LINE_PAIRS, MANO_POINTS_COLORS


HUMAN_COLOR = np.array([40, 180, 255], dtype=np.uint8)
HUMAN_TRAIL_COLOR = (40, 180, 255)


def rotation_matrix_to_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    xyzw = sciR.from_matrix(rot_mat).as_quat()
    return xyzw[[3, 0, 1, 2]]


def qpos_to_urdf_config(context: RobotReplayContext, urdf, qpos: np.ndarray) -> np.ndarray:
    qpos_by_name = dict(zip(context.actuated_joints_name, qpos))
    return np.asarray([qpos_by_name[name] for name in urdf.get_actuated_joint_names()])


def human_line_segments(frame: RetargetReplayFrame) -> np.ndarray:
    return np.asarray(
        [[frame.hand_keypoints_world[start], frame.hand_keypoints_world[end]] for start, end in MANO_LINE_PAIRS]
    )


def add_line_segments(scene, name: str, points: np.ndarray, color: tuple[int, int, int], line_width: float):
    if points.size == 0:
        return None
    if hasattr(scene, "add_line_segments"):
        colors = np.tile(np.asarray(color, dtype=np.uint8), (points.shape[0], 2, 1))
        return scene.add_line_segments(name, points=points, colors=colors, line_width=line_width)

    handles = []
    for idx, segment in enumerate(points):
        if hasattr(scene, "add_spline_catmull_rom"):
            handles.append(
                scene.add_spline_catmull_rom(
                    f"{name}/{idx}",
                    points=segment,
                    color=color,
                    line_width=line_width,
                )
            )
    return handles


def remove_handles(handles: List[object]):
    for handle in handles:
        if handle is None:
            continue
        if isinstance(handle, list):
            remove_handles(handle)
        elif hasattr(handle, "remove"):
            handle.remove()


def add_point_cloud(scene, name: str, points: np.ndarray, colors: np.ndarray, point_size: float):
    if points.size == 0:
        return None
    return scene.add_point_cloud(name, points=points, colors=colors, point_size=point_size)


def add_frame(scene, name: str, pose: np.ndarray, axes_length: float = 0.06):
    if not hasattr(scene, "add_frame"):
        return None
    return scene.add_frame(
        name,
        wxyz=rotation_matrix_to_wxyz(pose[:3, :3]),
        position=pose[:3, 3],
        axes_length=axes_length,
        axes_radius=0.004,
    )


def trajectory_points(frames: List[RetargetReplayFrame], current_idx: int, trail_length: int, getter) -> np.ndarray:
    start = 0 if trail_length <= 0 else max(0, current_idx - trail_length + 1)
    points = [getter(frame) for frame in frames[start : current_idx + 1]]
    points = [point for point in points if point is not None]
    if len(points) < 2:
        return np.zeros((0, 3))
    return np.asarray(points)


def add_trails(
    scene,
    frames: List[RetargetReplayFrame],
    current_idx: int,
    trail_length: int,
    show_human: bool,
):
    handles = []
    if show_human:
        for fingertip_idx in MANO_FINGERTIP_INDEX[:4]:
            points = trajectory_points(
                frames,
                current_idx,
                trail_length,
                lambda frame, idx=fingertip_idx: frame.hand_keypoints_world[idx],
            )
            if points.size and hasattr(scene, "add_spline_catmull_rom"):
                handles.append(
                    scene.add_spline_catmull_rom(
                        f"/trails/human/{fingertip_idx}",
                        points=points,
                        color=HUMAN_TRAIL_COLOR,
                        line_width=1.5,
                    )
                )
    return handles


def render_frame(
    server,
    frames: List[RetargetReplayFrame],
    current_idx: int,
    trail_length: int,
    human_keypoint_size: float,
    show_human: bool,
    show_trails: bool,
) -> List[object]:
    scene = server.scene
    frame = frames[current_idx]
    handles: List[object] = []

    if show_human:
        human_colors = np.asarray(MANO_POINTS_COLORS, dtype=np.uint8)
        handles.append(
            add_point_cloud(
                scene,
                "/current/human/keypoints",
                frame.hand_keypoints_world,
                human_colors,
                human_keypoint_size,
            )
        )
        handles.append(
            add_line_segments(scene, "/current/human/skeleton", human_line_segments(frame), tuple(HUMAN_COLOR), 2.0)
        )
        handles.append(add_frame(scene, "/current/human/wrist", frame.wrist_pose_world))

    if show_trails:
        handles.append(add_trails(scene, frames, current_idx, trail_length, show_human=show_human))

    return handles


def configure_initial_camera(server: Any, position: tuple[float, float, float], look_at: tuple[float, float, float]) -> None:
    """Configure Viser's initial and reset-view camera pose.

    Args:
        server: Viser server whose initial camera should be configured.
        position: Three-dimensional camera position in Viser world coordinates.
        look_at: Three-dimensional point that the camera initially targets.

    Returns:
        None.
    """
    # Viser applies this pose to new clients and uses it for the Reset View action.
    server.initial_camera.position = position
    server.initial_camera.look_at = look_at



def run_replay_viewer(options: dict[str, Any]) -> None:
    """Play one saved retargeting artifact in the Viser viewer.

    Args:
        options: Saved-artifact viewer options produced by an application entrypoint.

    Returns:
        None. This function runs until the process is interrupted.
    """
    result = options.get("result")
    if result is None or not str(result).strip():
        raise ValueError("Replay viewer requires a saved offline_retarget artifact.")

    try:
        import viser
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "viser is not installed. Install it in the retargeting environment before running this viewer."
        ) from exc

    trajectory, metadata = load_retargeting_trajectory(str(result))
    context = create_robot_replay_context_from_metadata(metadata)
    frames = trajectory_to_replay_frames(context, trajectory)

    server = viser.ViserServer(port=options["port"])
    configure_initial_camera(
        server,
        position=options["initial_camera_position"],
        look_at=options["initial_camera_look_at"],
    )
    print(f"Viser server started on port {options['port']}. Loaded {len(frames)} frames.")

    robot_urdf = None
    if not options["no_robot_mesh"]:
        robot_urdf = ViserUrdf(
            server,
            Path(context.robot_file_path),
            root_node_name="/robot_mesh",
            load_meshes=True,
            load_collision_meshes=False,
        )
        if frames[0].qpos is not None:
            robot_urdf.update_cfg(qpos_to_urdf_config(context, robot_urdf, frames[0].qpos))

    gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
    gui_frame = server.gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
    gui_fps = server.gui.add_number("FPS", initial_value=options["fps"])
    gui_show_human = server.gui.add_checkbox("Show human", initial_value=True)
    gui_show_robot_mesh = server.gui.add_checkbox("Show robot mesh", initial_value=robot_urdf is not None)
    gui_show_trails = server.gui.add_checkbox("Show trails", initial_value=False)
    gui_trail_length = server.gui.add_number("Trail length", initial_value=options["trail_length"])

    handles: List[object] = []
    last_rendered = None
    last_step_time = time.time()

    while True:
        now = time.time()
        if gui_playing.value and now - last_step_time >= 1.0 / max(float(gui_fps.value), 1e-6):
            gui_frame.value = (int(gui_frame.value) + 1) % len(frames)
            last_step_time = now

        state = (
            int(gui_frame.value),
            bool(gui_show_human.value),
            bool(gui_show_robot_mesh.value),
            bool(gui_show_trails.value),
            int(gui_trail_length.value),
        )
        if state != last_rendered:
            current_frame = frames[state[0]]
            if robot_urdf is not None:
                robot_urdf.show_visual = state[2]
                if current_frame.qpos is not None:
                    robot_urdf.update_cfg(qpos_to_urdf_config(context, robot_urdf, current_frame.qpos))
            remove_handles(handles)
            handles = render_frame(
                server,
                frames,
                current_idx=state[0],
                trail_length=state[4],
                human_keypoint_size=float(options["human_keypoint_size"]),
                show_human=state[1],
                show_trails=state[3],
            )
            last_rendered = state

        time.sleep(0.01)
