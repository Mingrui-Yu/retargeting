from pathlib import Path
import time
from typing import Any, List

import numpy as np
from mr_utils.utils_mano import MANO_FINGERTIP_INDEX
from retargeting_apps.offline_retargeting import (
    RetargetReplayFrame,
    RobotReplayContext,
    create_robot_replay_context_from_metadata,
    trajectory_to_replay_frames,
)
from retargeting_apps.artifacts.trajectory import load_retargeting_trajectory
from retargeting_apps.visualization.viser_scene import (
    ViserHandObservationRenderer,
    configure_initial_camera,
)


HUMAN_TRAIL_COLOR = (40, 180, 255)


def qpos_to_urdf_config(context: RobotReplayContext, urdf, qpos: np.ndarray) -> np.ndarray:
    qpos_by_name = dict(zip(context.actuated_joints_name, qpos))
    return np.asarray([qpos_by_name[name] for name in urdf.get_actuated_joint_names()])


def remove_handles(handles: List[object]):
    for handle in handles:
        if handle is None:
            continue
        if isinstance(handle, list):
            remove_handles(handle)
        elif hasattr(handle, "remove"):
            handle.remove()


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
    human_renderer = ViserHandObservationRenderer(
        server,
        point_size=float(options["human_keypoint_size"]),
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

    trail_handles: List[object] = []
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
            if state[1]:
                human_renderer.update_world(
                    current_frame.hand_keypoints_world,
                    current_frame.wrist_pose_world,
                )
            else:
                human_renderer.hide()
            remove_handles(trail_handles)
            trail_handles = (
                add_trails(
                    server.scene,
                    frames,
                    current_idx=state[0],
                    trail_length=state[4],
                    show_human=state[1],
                )
                if state[3]
                else []
            )
            last_rendered = state

        time.sleep(0.01)
