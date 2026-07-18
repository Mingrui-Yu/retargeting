#!/usr/bin/env python3
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Union

import numpy as np
import rclpy
from _retargeting_compat import ensure_retargeting_package
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_adaptor import RobotAdaptor
from robot_benchmark import RobotBenchmark
from robot_control import RobotControl
from robot_pinocchio import RobotPinocchio
from robot_real import RobotReal
from teleoperation.config import (
    load_detection_source_config,
    load_teleoperation_command_config,
    load_teleoperation_mode_config,
)
from teleoperation.inputs.offline_avp import load_offline_avp_trajectory
from teleoperation.avp_alignment import initialize_avp_alignment
from teleoperation.session import TeleoperationSession
from rviz_visualize import RvizVisualizer
from utils.utils_keyboard import KeyboardListener

ensure_retargeting_package()
from retargeting.config import (
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
)


def flatten_stream_data(data_dict):
    """
    Extract the elements in 'stream' dict to the overall dictp
    """
    new_data_dict = {}
    for step, stream in enumerate(data_dict["stream"]):
        for key, value in stream.items():
            new_key = f"stream_{key}"
            if new_key not in new_data_dict:
                new_data_dict[new_key] = []
            new_data_dict[new_key].append(value)

    for key, value in data_dict.items():
        if key != "stream":
            new_data_dict[key] = value

    return new_data_dict


class RobotTeleoperationMain:
    def __init__(self):
        # --------- hyper-parameters ---------
        repo_root = Path(__file__).resolve().parents[4]
        profile_source = repo_root / "configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml"
        profile_config = load_retargeting_profile_config(profile_source)
        detection_source_config = load_detection_source_config(repo_root / "configs/detection_sources/avp.yaml")
        robot_config = load_robot_config(repo_root / profile_config.robot)
        self.robot_config = robot_config
        retargeting_config = load_retargeting_config(repo_root / profile_config.method)
        solver_config = load_solver_config(repo_root / "configs/solvers/nlopt_slsqp.yaml")
        teleoperation_mode_config = load_teleoperation_mode_config(
            repo_root / "configs/teleoperation_modes/simulation.yaml"
        )
        urdf_file_name = robot_config.robot_file_path
        actuated_joints_name = list(robot_config.actuated_joints)
        command_config = load_teleoperation_command_config(profile_source, robot_config=robot_config)
        self.max_joint_speed = list(command_config.max_joint_speed)

        self.avp_ip = "192.168.52.6"
        # self.avp_ip = "192.168.60.250"
        self.input_device = detection_source_config.input_device
        self.load_offline_data = True
        self.use_hardware = teleoperation_mode_config.robot_control.use_hardware
        self.use_virtual_hardware = teleoperation_mode_config.robot_control.use_virtual_hardware
        self.use_high_freq_interp = teleoperation_mode_config.robot_control.use_high_freq_interp
        self.use_ros = True

        if self.use_ros:
            rclpy.init(args=None)
            self.node = Node("main_node")
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()
            self.rviz_visualizer = RvizVisualizer(node=self.node)
        else:
            self.node = None

        self.robot_model = RobotPinocchio(
            robot_file_path=urdf_file_name,
            robot_file_type="urdf",
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            actuated_joints_name=actuated_joints_name,
        )
        self.robot_control = RobotControl(
            self.robot_model,
            self.robot_adaptor,
            initial_qpos=np.asarray(robot_config.initial_qpos, dtype=float),
            arm_dof=profile_config.retargeting.arm_dof,
            use_hardware=self.use_hardware,
            use_virtual_hardware=self.use_virtual_hardware,
            use_high_freq_interp=self.use_high_freq_interp,
            node=self.node,
        )
        self.robot_teleop = TeleoperationSession(
            robot_adaptor=self.robot_adaptor,
            robot_config=robot_config,
            profile_config=profile_config,
            method_config=retargeting_config,
            detection_source_config=detection_source_config,
            teleoperation_mode_config=teleoperation_mode_config,
            solver_config=solver_config,
        )
        # self.robot_benchmark = RobotBenchmark(robot_adaptor=self.robot_adaptor)
        if self.input_device == "avp":
            if not self.load_offline_data:
                self.robot_teleop.detector.connect(avp_ip=self.avp_ip)
        else:
            raise NotImplementedError()

        # check the retargeting type
        print(retargeting_config.type)

        # for keyboard control
        self.keyboard_listener = KeyboardListener()
        self.keyboard_listener.start_keyboard_listening_thread()
        # for recording data
        self.data = {}
        self.data["stream"] = []
        self.data["retarget_qpos"] = []

    def save_data(self, save_dir):
        file = os.path.join(save_dir, "data.npz")
        data = flatten_stream_data(self.data)
        np.savez(file, **data)
        print(f"Save stream data to {file}.")

    def load_data(self, file_name):
        trajectory = load_offline_avp_trajectory(file_name)
        return {"stream": [trajectory.get_frame(frame_idx) for frame_idx in range(trajectory.n_frames)]}

    def main(self):
        # ----------- hyper-parameters -----------
        project_dir = "/home/mingrui/mingrui/research/retargeting"
        save_dir = os.path.join(project_dir, f"outputs/teleop/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(save_dir, exist_ok=True)
        if self.load_offline_data:
            file_name = os.path.join(project_dir, "data/test_teleop/avp/data_2025-01-16_20-27-43.npz")
            data_dict = self.load_data(file_name)
            stream_data = data_dict["stream"]

        i = 0
        total_position_err = 0
        total_orientation_err = 0
        total_relative_position_err = 0
        total_relative_position_to_wrist_err = 0
        total_time_cost = 0

        position_err_list = []
        orientation_err_list = []
        relative_position_err_list = []
        relative_position_to_wrist_err_list = []
        time_cost_list = []

        # move to initial configuration
        self.robot_control.move_to_joint_pos(self.robot_control.init_joint_pos, max_joint_speed=self.max_joint_speed)

        if self.use_hardware and not self.use_virtual_hardware:
            self.robot_control.env.start_record_video(data_dir=save_dir)

        # i_start = 217
        # i_end = 250
        i_start = 0
        i_end = len(stream_data) - 1
        while True:
            t_frame_start = time.time()
            print(f"Frame {i}:")

            if self.load_offline_data:
                if i > i_end:
                    break
                if i < i_start:
                    i += 1
                    continue

            # -------- get human motion --------
            if self.load_offline_data:
                r = stream_data[i]
                # print("right_wrist: ", r["right_wrist"])
            else:
                r = self.robot_teleop.detector.get_raw_stream()

            # print(f"Frame time cost 1: {(time.time() - t_frame_start):.3f}")

            # -------- retargeting --------
            if i == i_start:  # set initial poses
                init_joint_pos = self.robot_control.get_joint_pos(update=True)
                if not initialize_avp_alignment(self.robot_teleop, r, init_joint_pos):
                    raise ValueError(f"Unable to initialize AVP alignment from frame {i}.")

            observation, qpos, err = self.robot_teleop.retarget_input(r)
            if observation is None:
                i += 1
                continue
            hand_kps_in_wrist = observation.keypoints_wrist
            wrist_pose = observation.wrist_pose_world

            # print(f"Frame time cost 2: {(time.time() - t_frame_start):.3f}")

            # -------- control robot --------
            if self.use_hardware and i < i_start + 10:
                self.robot_control.move_to_joint_pos(qpos, max_joint_speed=self.max_joint_speed)
                print("Slowly move to the first 10 retargeted configuration.")
            else:
                self.robot_control.ctrl_joint_pos(qpos)

            self.robot_control.step()

            position_err_list.append(err["position_err"])
            orientation_err_list.append(err["orientation_err"])
            relative_position_err_list.append(err["relative_position_err"])
            relative_position_to_wrist_err_list.append(err["relative_position_to_wrist_err"])
            time_cost_list.append(err["optimization_time"])

            total_position_err += err["position_err"]
            total_orientation_err += err["orientation_err"]
            total_relative_position_err += err["relative_position_err"]
            total_relative_position_to_wrist_err += err["relative_position_to_wrist_err"]
            total_time_cost += err["optimization_time"]

            # print(f"Frame time cost 3: {(time.time() - t_frame_start):.3f}")

            # -------- visualization --------
            if hand_kps_in_wrist is not None:
                # visualize the human hand in rviz
                self.rviz_visualizer.publish_hand_detection_results(
                    hand_kps_in_wrist, wrist_pose, frame_id="visualize/world"
                )
                # visualize the robot hand in rviz
                joints_name = self.robot_model.joint_names
                qpos_dof = self.robot_adaptor.forward_qpos(qpos)
                self.rviz_visualizer.publish_robot_joint_states(joints_name=joints_name, joints_pos=qpos_dof)

            # print(f"Frame time cost 4: {(time.time() - t_frame_start):.3f}")

            # -------- record data --------
            self.data["stream"].append(r)
            self.data["retarget_qpos"].append(qpos)

            t_frame_end = time.time()
            print(f"Frame total time cost: {(t_frame_end - t_frame_start):.3f}")
            i += 1

            # quit loop criterian
            if "p" in self.keyboard_listener.pressed_keys:
                self.save_data(save_dir)
                if self.use_hardware and not self.use_virtual_hardware:
                    self.robot_control.env.stop_record_video()
                break

            # return to the initial configuration and re-start
            if self.load_offline_data:
                if i >= len(stream_data):
                    # i = 0
                    # self.robot_control.move_to_joint_pos(
                    #     self.robot_control.init_joint_pos, max_joint_speed=self.max_joint_speed
                    # )
                    self.save_data(save_dir)
                    if self.use_hardware and not self.use_virtual_hardware:
                        self.robot_control.env.stop_record_video()
                    break
        # --------------------------------- end loop ---------------------------------

        # save quantitative results
        if self.load_offline_data:
            print("---------------------------------------")
            print("average_position_err: ", total_position_err / len(stream_data))
            print("average_orientation_err: ", total_orientation_err / len(stream_data))
            print("average_relative_position_err: ", total_relative_position_err / len(stream_data))
            print("average_relative_position_to_wrist_err: ", total_relative_position_to_wrist_err / len(stream_data))
            print("average_time_cost: ", total_time_cost / len(stream_data))
            output_file = f"outputs/simulation/{self.robot_config.name}/complex_8.npz"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez(
                output_file,
                position_err=np.array(position_err_list),
                orientation_err=np.array(orientation_err_list),
                relative_position_err=np.array(relative_position_err_list),
                relative_position_to_wrist_err=np.array(relative_position_to_wrist_err_list),
                time_cost=np.array(time_cost_list),
            )
            print(f"Saved quantitative results to {output_file}")


def main():
    teleoperation = RobotTeleoperationMain()
    teleoperation.main()


if __name__ == "__main__":
    main()
