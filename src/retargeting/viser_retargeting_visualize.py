#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import time
from typing import Any, List

import numpy as np
from retargeting.config import (
    load_detection_source_config,
    load_replay_app_config,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
    load_teleoperation_mode_config,
    resolve_project_path,
    to_plain_config_data,
)
from retargeting.retargeting_replay import (
    RetargetReplayFrame,
    RobotReplayContext,
    build_retarget_replay_frames,
    create_robot_replay_context_from_metadata,
    trajectory_to_replay_frames,
)
from retargeting.trajectory_result import load_retargeting_trajectory
from scipy.spatial.transform import Rotation as sciR
from retargeting.utils.utils_mano import MANO_FINGERTIP_INDEX, MANO_LINE_PAIRS, MANO_POINTS_COLORS


HUMAN_COLOR = np.array([40, 180, 255], dtype=np.uint8)
HUMAN_TRAIL_COLOR = (40, 180, 255)
DEFAULT_REPLAY_DATA = "tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz"


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
    show_human: bool,
    show_trails: bool,
) -> List[object]:
    scene = server.scene
    frame = frames[current_idx]
    handles: List[object] = []

    if show_human:
        human_colors = np.asarray(MANO_POINTS_COLORS, dtype=np.uint8)
        handles.append(
            add_point_cloud(scene, "/current/human/keypoints", frame.hand_keypoints_world, human_colors, 0.018)
        )
        handles.append(
            add_line_segments(scene, "/current/human/skeleton", human_line_segments(frame), tuple(HUMAN_COLOR), 2.0)
        )
        handles.append(add_frame(scene, "/current/human/wrist", frame.wrist_pose_world))

    if show_trails:
        handles.append(add_trails(scene, frames, current_idx, trail_length, show_human=show_human))

    return handles


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Visualize offline hand retargeting replay with viser.")
    parser.add_argument("--config", default=None, help="Replay app config path.")
    parser.add_argument("--profile", default=None, help="Retargeting profile config path.")
    parser.add_argument("--detection-source", default=None, help="Detection source config path.")
    parser.add_argument("--robot", default=None, help="Robot config path.")
    parser.add_argument("--retarget", default=None, help="Retargeting method config path.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--result", default=None, help="Saved retargeting result directory or result.npz path.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-robot-mesh", action="store_true", default=None, help="Disable URDF mesh rendering.")
    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="Use 0 to show the full trajectory up to current frame.",
    )
    return parser.parse_args(argv)


def resolve_replay_options(args):
    app_config = load_replay_app_config(args.config) if args.config is not None else None
    viewer_config = app_config.viewer if app_config is not None else None

    profile_config_path = args.profile if args.profile is not None else (app_config.profile if app_config is not None else None)
    detection_source_path = (
        args.detection_source
        if args.detection_source is not None
        else (app_config.detection_source if app_config is not None else None)
    )
    profile_config = load_retargeting_profile_config(profile_config_path) if profile_config_path is not None else None
    detection_source_config = load_detection_source_config(detection_source_path) if detection_source_path is not None else None
    robot_config_path = args.robot if args.robot is not None else (profile_config.robot if profile_config is not None else None)
    retargeting_config_path = args.retarget if args.retarget is not None else (
        profile_config.method if profile_config is not None else None
    )
    solver_config_path = app_config.solver if app_config is not None else None
    robot_config = load_robot_config(robot_config_path) if robot_config_path is not None else None
    retargeting_config = (
        load_retargeting_config(retargeting_config_path) if retargeting_config_path is not None else None
    )
    solver_config = load_solver_config(solver_config_path)
    teleoperation_mode_config = load_teleoperation_mode_config(None)

    return {
        "result": args.result if args.result is not None else getattr(app_config, "result", None),
        "data": args.data if args.data is not None else (app_config.data if app_config is not None else DEFAULT_REPLAY_DATA),
        "start": args.start if args.start is not None else (app_config.start if app_config is not None else 0),
        "end": args.end if args.end is not None else (app_config.end if app_config is not None else -1),
        "stride": args.stride if args.stride is not None else (app_config.stride if app_config is not None else 1),
        "fps": args.fps if args.fps is not None else (viewer_config.fps if viewer_config is not None else 30.0),
        "port": args.port if args.port is not None else (viewer_config.port if viewer_config is not None else 8080),
        "no_robot_mesh": args.no_robot_mesh
        if args.no_robot_mesh is not None
        else (viewer_config.no_robot_mesh if viewer_config is not None else False),
        "trail_length": args.trail_length
        if args.trail_length is not None
        else (viewer_config.trail_length if viewer_config is not None else 120),
        "robot_config": robot_config,
        "retargeting_config": retargeting_config,
        "retargeting_profile_config": profile_config,
        "detection_source_config": detection_source_config,
        "teleoperation_mode_config": teleoperation_mode_config,
        "solver_config": solver_config,
    }


def compose_hydra_replay_config(overrides: List[str] | None = None) -> dict[str, Any]:
    """Compose the replay app config with Hydra and resolve interpolations.

    Args:
        overrides: Hydra override strings supplied after the command name.

    Returns:
        A resolved plain dictionary containing app, robot, retargeting, and viewer config sections.
    """
    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "hydra-core is required for the Hydra replay entrypoint. "
            "Install the project dependencies, for example with `pip install -e .[replay]`."
        ) from exc

    config_dir = resolve_project_path("configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="replay", overrides=list(overrides or []))
    return OmegaConf.to_container(config, resolve=True)


def resolve_replay_options_from_config(config: Any) -> dict[str, Any]:
    """Build replay runtime options from a Hydra-composed config.

    Args:
        config: A Hydra/OmegaConf config object or an equivalent plain dictionary.

    Returns:
        A runtime options dictionary consumed by run_replay_viewer().
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected Hydra replay config to be a mapping.")
    app_data = config_data.get("app", {})
    viewer_data = config_data.get("viewer", app_data.get("viewer", {}))

    profile_source = config_data.get("profile", app_data.get("profile"))
    detection_source = config_data.get("detection_source", app_data.get("detection_source"))
    solver_source = config_data.get("solver", app_data.get("solver"))
    teleoperation_mode_source = config_data.get("teleoperation_mode", app_data.get("teleoperation_mode"))
    profile_config = load_retargeting_profile_config(profile_source) if profile_source is not None else None
    detection_source_config = load_detection_source_config(detection_source) if detection_source is not None else None
    robot_config = load_robot_config(profile_config.robot) if profile_config is not None else None
    retargeting_config = load_retargeting_config(profile_config.method) if profile_config is not None else None
    solver_config = load_solver_config(solver_source)
    teleoperation_mode_config = load_teleoperation_mode_config(teleoperation_mode_source)

    return {
        "result": config_data.get("result", app_data.get("result")),
        "data": config_data.get("data", app_data.get("data", DEFAULT_REPLAY_DATA)),
        "start": int(config_data.get("start", app_data.get("start", 0))),
        "end": int(config_data.get("end", app_data.get("end", -1))),
        "stride": int(config_data.get("stride", app_data.get("stride", 1))),
        "fps": float(viewer_data.get("fps", 30.0)),
        "port": int(viewer_data.get("port", 8080)),
        "no_robot_mesh": bool(viewer_data.get("no_robot_mesh", False)),
        "trail_length": int(viewer_data.get("trail_length", 120)),
        "robot_config": robot_config,
        "retargeting_config": retargeting_config,
        "retargeting_profile_config": profile_config,
        "detection_source_config": detection_source_config,
        "teleoperation_mode_config": teleoperation_mode_config,
        "solver_config": solver_config,
    }


def _has_legacy_cli_args(argv: List[str]) -> bool:
    """Detect whether the user is invoking the old argparse-compatible CLI.

    Args:
        argv: Command-line arguments after the program name.

    Returns:
        True when at least one argument uses argparse-style dash prefixes.
    """
    return any(arg.startswith("-") for arg in argv)


def run_replay_viewer(options: dict[str, Any]) -> None:
    """Run the viser replay loop from already-resolved runtime options.

    Args:
        options: Runtime options produced by argparse compatibility or Hydra composition.

    Returns:
        None. This function runs until the process is interrupted.
    """
    try:
        import viser
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "viser is not installed. Install it in the retargeting environment before running this viewer."
        ) from exc

    if options.get("result"):
        trajectory, metadata = load_retargeting_trajectory(options["result"])
        context = create_robot_replay_context_from_metadata(metadata)
        frames = trajectory_to_replay_frames(context, trajectory)
    else:
        context, frames = build_retarget_replay_frames(
            data_file=options["data"],
            start=options["start"],
            end=options["end"],
            stride=options["stride"],
            robot_config=options["robot_config"],
            retargeting_config=options["retargeting_config"],
            retargeting_profile_config=options["retargeting_profile_config"],
            detection_source_config=options["detection_source_config"],
            teleoperation_mode_config=options["teleoperation_mode_config"],
            solver_config=options["solver_config"],
        )

    server = viser.ViserServer(port=options["port"])
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
                show_human=state[1],
                show_trails=state[3],
            )
            last_rendered = state

        time.sleep(0.01)


def main(argv: List[str] | None = None):
    """Run the replay viewer CLI using Hydra overrides or legacy argparse flags.

    Args:
        argv: Optional command-line arguments after the program name.

    Returns:
        None. The selected viewer entrypoint runs until interrupted.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if _has_legacy_cli_args(argv):
        options = resolve_replay_options(parse_args(argv))
    else:
        options = resolve_replay_options_from_config(compose_hydra_replay_config(argv))
    run_replay_viewer(options)


if __name__ == "__main__":
    main()
