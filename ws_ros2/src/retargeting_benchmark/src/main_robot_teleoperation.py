#!/usr/bin/env python3
"""ROS/real-robot compatibility entrypoint using the canonical execution flow."""

import os
from datetime import datetime
from pathlib import Path
from threading import Thread

import numpy as np
import rclpy
from _retargeting_compat import ensure_retargeting_package
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_adaptor import RobotAdaptor
from robot_control import RobotControl
from robot_pinocchio import RobotPinocchio
from rviz_visualize import RvizVisualizer
from utils.utils_keyboard import KeyboardListener

ensure_retargeting_package()

from retargeting.config import (
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)
from retargeting.core import Retargeter
from retargeting.evaluation.robot_metrics import RobotBenchmark
from retargeting_ros.backends import RosCommandBackend
from teleoperation.config import (
    load_detection_source_config,
    load_teleoperation_command_config,
    load_teleoperation_mode_config,
)
from teleoperation.flow import ExecutionFlow
from teleoperation.inputs.avp import AvpOfflineInput, AvpOnlineInput
from teleoperation.observation_mapping import AvpRelativeWristMapper
from teleoperation.output import QposCommandLimiter, QposOutputFilter
from teleoperation.types import ExecutionStepResult


def flatten_stream_data(data_dict: dict) -> dict:
    """Flatten recorded raw stream mappings into NPZ-compatible arrays.

    Args:
        data_dict: Recording mapping containing a list under ``stream``.

    Returns:
        Mapping with each stream field promoted to a ``stream_*`` array.
    """
    flattened: dict = {}
    for stream in data_dict["stream"]:
        for key, value in stream.items():
            flattened.setdefault(f"stream_{key}", []).append(value)
    for key, value in data_dict.items():
        if key != "stream":
            flattened[key] = value
    return flattened


class RobotTeleoperationMain:
    """Compose ROS acquisition, canonical execution flow, and passive recording."""

    def __init__(self) -> None:
        """Construct robot, retargeting, ROS, and command-policy components.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = Path(__file__).resolve().parents[4]
        profile_source = repo_root / "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
        self.profile_config = load_retargeting_profile_config(profile_source)
        self.detection_config = load_detection_source_config(repo_root / "configs/detection_sources/avp.yaml")
        self.robot_config = load_robot_config(repo_root / self.profile_config.robot)
        self.method_config = load_retargeting_config(repo_root / self.profile_config.method)
        self.solver_config = load_solver_config(repo_root / "configs/solvers/nlopt_slsqp.yaml")
        self.mode_config = load_teleoperation_mode_config(repo_root / "configs/teleoperation_modes/simulation.yaml")
        command_config = load_teleoperation_command_config(profile_source, robot_config=self.robot_config)
        self.max_joint_speed = np.asarray(command_config.max_joint_speed, dtype=float)
        self.avp_ip = "192.168.52.6"
        self.load_offline_data = True
        self.use_hardware = self.mode_config.robot_control.use_hardware
        self.use_virtual_hardware = self.mode_config.robot_control.use_virtual_hardware
        self.use_high_freq_interp = self.mode_config.robot_control.use_high_freq_interp

        rclpy.init(args=None)
        self.node = Node("main_node")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.rviz_visualizer = RvizVisualizer(node=self.node)
        self.robot_model = RobotPinocchio(
            robot_file_path=self.robot_config.robot_file_path,
            robot_file_type=self.robot_config.model.type,
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            actuated_joints_name=list(self.robot_config.actuated_joints),
        )
        self.robot_control = RobotControl(
            self.robot_model,
            self.robot_adaptor,
            initial_qpos=np.asarray(self.robot_config.initial_qpos, dtype=float),
            arm_dof=self.profile_config.retargeting.arm_dof,
            use_hardware=self.use_hardware,
            use_virtual_hardware=self.use_virtual_hardware,
            use_high_freq_interp=self.use_high_freq_interp,
            node=self.node,
        )
        self.keyboard_listener = KeyboardListener()
        self.keyboard_listener.start_keyboard_listening_thread()
        self.data = {"stream": [], "retarget_qpos": []}
        self.metric_history: dict[str, list[float]] = {
            "position_err": [],
            "orientation_err": [],
            "relative_position_err": [],
            "relative_position_to_wrist_err": [],
            "optimization_time": [],
        }

    def save_data(self, save_dir: str) -> None:
        """Persist recorded raw streams and retargeted qpos arrays.

        Args:
            save_dir: Existing output directory.

        Returns:
            None.
        """
        file_name = os.path.join(save_dir, "data.npz")
        np.savez(file_name, **flatten_stream_data(self.data))
        print(f"Save stream data to {file_name}.")

    def _build_flow(self, data_file: str | None) -> ExecutionFlow:
        """Construct one complete offline or online AVP execution flow.

        Args:
            data_file: Offline NPZ path, or None for live acquisition.

        Returns:
            Canonical execution flow using the ROS robot-control backend.
        """
        retargeter = Retargeter(
            self.robot_adaptor,
            self.robot_config,
            self.profile_config,
            self.method_config,
            self.solver_config,
        )
        output_filter = QposOutputFilter(retargeter.qpos_init, self.mode_config)
        mapper = AvpRelativeWristMapper(
            self.detection_config,
            self.robot_config.human_hand_scale,
            self.robot_adaptor,
            self.robot_model,
            self.robot_config.wrist_frame_name,
        )
        evaluator = RobotBenchmark(self.robot_adaptor, self.robot_config.benchmark)

        def execute_robot(qpos: np.ndarray) -> np.ndarray:
            """Execute one complete RobotControl command period.

            Args:
                qpos: Requested actuated-joint target.

            Returns:
                Measured robot state after the period.
            """
            self.robot_control.ctrl_joint_pos(qpos)
            self.robot_control.step()
            return self.robot_control.get_joint_pos(update=True)

        def reset_robot(qpos: np.ndarray) -> np.ndarray:
            """Move RobotControl to a synchronized flow reset target.

            Args:
                qpos: Requested reset target.

            Returns:
                Measured robot state after the reset move.
            """
            self.robot_control.move_to_joint_pos(qpos, max_joint_speed=self.max_joint_speed)
            return self.robot_control.get_joint_pos(update=True)

        backend = RosCommandBackend(
            initial_qpos=self.robot_config.initial_qpos,
            control_period=float(self.robot_control.env.timestep),
            execute_callback=execute_robot,
            reset_callback=reset_robot,
        )
        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        command_policy = QposCommandLimiter(
            initial_qpos=backend.get_target_joint_pos(),
            max_joint_speed=self.max_joint_speed,
            command_hz=1.0 / backend.control_period,
            lower=joint_limits[:, 0],
            upper=joint_limits[:, 1],
        )
        hand_input = AvpOnlineInput(self.avp_ip) if data_file is None else AvpOfflineInput(data_file)
        flow = ExecutionFlow(
            input=hand_input,
            observation_mapper=mapper,
            retargeter=retargeter,
            output_filter=output_filter,
            evaluator=evaluator,
            command_policy=command_policy,
            backend=backend,
            realtime=False,
            startup_move_frames=10 if self.use_hardware else 0,
        )
        flow.add_step_observer(self._observe_step)
        return flow

    def _observe_step(self, result: ExecutionStepResult) -> None:
        """Record and visualize a completed canonical execution result.

        Args:
            result: Immutable source-frame result from the flow.

        Returns:
            None.
        """
        frame = result.retargeted_frame
        if frame is None:
            return
        observation = frame.observation
        self.rviz_visualizer.publish_hand_detection_results(
            observation.keypoints_wrist,
            observation.wrist_pose_world,
            frame_id="visualize/world",
        )
        qpos_dof = self.robot_adaptor.forward_qpos(frame.retargeted_qpos)
        self.rviz_visualizer.publish_robot_joint_states(self.robot_model.joint_names, qpos_dof)
        self.data["stream"].append(observation.raw)
        self.data["retarget_qpos"].append(np.asarray(frame.retargeted_qpos, dtype=float).copy())
        for key in self.metric_history:
            if key in frame.diagnostics:
                self.metric_history[key].append(float(frame.diagnostics[key]))
        if "p" in self.keyboard_listener.pressed_keys:
            raise KeyboardInterrupt

    def main(self) -> None:
        """Run the configured acquisition flow and persist compatibility outputs.

        Args:
            None.

        Returns:
            None.
        """
        project_dir = "/home/mingrui/mingrui/research/retargeting"
        save_dir = os.path.join(project_dir, f"outputs/teleop/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(save_dir, exist_ok=True)
        data_file = (
            os.path.join(project_dir, "data/test_teleop/avp/data_2025-01-16_20-27-43.npz")
            if self.load_offline_data
            else None
        )
        flow = self._build_flow(data_file)
        flow.reset(np.asarray(self.robot_config.initial_qpos, dtype=float))
        if self.use_hardware and not self.use_virtual_hardware:
            self.robot_control.env.start_record_video(data_dir=save_dir)
        summary = flow.run()
        self.save_data(save_dir)
        if self.use_hardware and not self.use_virtual_hardware:
            self.robot_control.env.stop_record_video()
        if self.load_offline_data:
            print(f"Processed {summary.retarget_frames_processed} valid retargeted frames.")
            for key, values in self.metric_history.items():
                if values:
                    print(f"average_{key}: {float(np.mean(values))}")


def main() -> None:
    """Construct and run the ROS compatibility application.

    Args:
        None.

    Returns:
        None.
    """
    RobotTeleoperationMain().main()


if __name__ == "__main__":
    main()
