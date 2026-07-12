from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from retargeting.config import (
    DetectionSourceConfig,
    RetargetingConfig,
    RetargetingObjectiveConfig,
    RetargetingProfileConfig,
    RobotConfig,
    SolverConfig,
    default_solver_config,
    TeleoperationModeConfig,
)
from retargeting.retarget_optimizer import get_optimizer_class
from retargeting.robot_adaptor import RobotAdaptor
from retargeting.robot_benchmark import RobotBenchmark

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from sklearn.preprocessing import normalize
from retargeting.utils.utils_calc import posRotMat2Isometry3d, quatXYZW2WXYZ, sciR, transformPositions


@dataclass(frozen=True)
class RetargetObservation:
    hand_kps_in_wrist: np.ndarray
    wrist_pose_in_world: np.ndarray
    keypoint_2d: Optional[np.ndarray] = None
    raw: Any = None


def sigmoid(x, c=0, w=1):
    return 1 / (1 + np.exp(w * (x - c)))


def merge_objective_solver_params(objective_params: dict, solver_config: SolverConfig) -> dict:
    """
    Args:
        objective_params: Objective-specific retargeting parameters.
        solver_config: Solver backend selection and backend-specific parameters.

    Returns:
        Runtime optimizer parameter mapping consumed by current optimizer classes.
    """
    params = dict(objective_params)
    params["solver"] = solver_config.name
    params["solver_params"] = dict(solver_config.params)
    return params


class RobotTeleoperation:
    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        robot_control: Any,
        robot_config: RobotConfig,
        profile_config: RetargetingProfileConfig,
        method_config: RetargetingConfig,
        detection_source_config: DetectionSourceConfig,
        teleoperation_mode_config: TeleoperationModeConfig | None = None,
        solver_config: SolverConfig | None = None,
    ):
        """Create a teleoperation retargeter from the repository config layers.

        Args:
            robot_adaptor: Robot kinematics adapter for actuated qpos conversion.
            robot_control: Optional hardware/control adapter used by live teleoperation.
            robot_config: Robot embodiment config from configs/robots.
            profile_config: Robot-method profile config from configs/retargeting_profiles.
            method_config: Retargeting method config from configs/retargeting_methods.
            detection_source_config: Detector input and calibration config from configs/detection_sources.
            teleoperation_mode_config: Runtime mode config for output filtering.
            solver_config: Optional solver backend config.
        """
        if robot_config is None:
            raise ValueError("RobotTeleoperation requires robot_config.")
        if method_config is None:
            raise ValueError("RobotTeleoperation requires method_config.")
        if profile_config is None:
            raise ValueError("RobotTeleoperation requires profile_config.")
        if detection_source_config is None:
            raise ValueError("RobotTeleoperation requires detection_source_config.")
        if teleoperation_mode_config is None:
            raise ValueError("RobotTeleoperation requires teleoperation_mode_config.")

        robot_config.validate()
        method_config.validate()
        profile_config.validate(robot_config)
        detection_source_config.validate()

        self.retarget_type = method_config.type
        self.setting_id = method_config.setting_id
        self.ablation_option = method_config.ablation_option

        teleoperation_mode_config.validate()

        self.teleoperation_mode_config = teleoperation_mode_config
        self.smooth_output_qpos = teleoperation_mode_config.output.smooth_output_qpos
        self.ema_alpha = teleoperation_mode_config.output.smoothing_alpha
        self.solver_config = default_solver_config() if solver_config is None else solver_config
        self.objective_config: RetargetingObjectiveConfig = profile_config.objective
        self.teleoperation_config = profile_config.teleoperation
        self.detection_source_config = detection_source_config
        print(
            "[RobotTeleoperation] Retargeting method config:\n"
            f"  type: {self.retarget_type}\n"
            f"  setting_id: {self.setting_id}\n"
            f"  ablation_option: {self.ablation_option} ({method_config.ablation_description})\n"
            f"  optimizer_class: {method_config.optimizer_class}\n"
            f"  solver: {self.solver_config.name}\n"
            f"  teleoperation_mode: {self.teleoperation_mode_config.name}\n"
            f"  smooth_output_qpos: {self.smooth_output_qpos}\n"
            f"  smoothing_alpha: {self.ema_alpha}"
        )

        if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
            self.retarget_wrist_method = "separate"
        else:
            self.retarget_wrist_method = "joint"  # "separate" or "joint"

        self.human_hand_scale = robot_config.human_hand_scale
        self.use_relative_pos = self.detection_source_config.use_relative_wrist_alignment
        # ----------------------------------

        self.input_device = self.detection_source_config.input_device
        self.robot_model = robot_adaptor.robot_model
        self.robot_adaptor = robot_adaptor
        self.robot_control = robot_control

        self.robot_benchmark = RobotBenchmark(
            robot_adaptor=self.robot_adaptor,
            benchmark_config=robot_config.benchmark,
        )
        self.benchmark_config = robot_config.benchmark
        benchmark_config = robot_config.benchmark
        thumb_fingertip = benchmark_config.thumb_fingertip
        primary_fingertips = benchmark_config.primary_fingertips
        self.human_fingertip_indices = np.asarray(
            [thumb_fingertip.human_tip_index, *[fingertip.human_tip_index for fingertip in primary_fingertips]],
            dtype=int,
        )
        self.human_fingertip_base_indices = np.asarray(
            [
                thumb_fingertip.human_direction_base_index,
                *[fingertip.human_direction_base_index for fingertip in primary_fingertips],
            ],
            dtype=int,
        )
        self.arm_dof = self.teleoperation_config.arm_dof

        optimizer_class = get_optimizer_class(method_config.optimizer_class)
        target_config = profile_config.target
        target_link_pairs = target_config.link_pairs
        self.target_link_count = len(target_link_pairs)
        targets = {
            "origin_links_name": [pair[0] for pair in target_link_pairs],
            "task_links_name": [pair[1] for pair in target_link_pairs],
            "wrist_link_name": target_config.wrist_link_name,
        }
        params = merge_objective_solver_params(method_config.optimizer_params, self.solver_config)
        joint_limit_overrides = [
            {
                "indices": list(override.indices),
                "lower": override.lower,
                "upper": override.upper,
            }
            for override in profile_config.joint_limit_overrides
        ]
        self.optimizer = optimizer_class(
            robot_adaptor=self.robot_adaptor,
            targets=targets,
            params=params,
            joint_limit_overrides=joint_limit_overrides,
            solver=self.solver_config.name,
        )
        self.arm_optimizer = None
        if self.retarget_wrist_method == "separate":
            arm_targets = {
                "origin_links_name": ["world"],
                "task_links_name": [target_config.wrist_link_name],
                "wrist_link_name": target_config.wrist_link_name,
            }
            self.arm_optimizer = optimizer_class(
                robot_adaptor=self.robot_adaptor,
                targets=arm_targets,
                params=params,
                joint_limit_overrides=joint_limit_overrides,
                solver=self.solver_config.name,
            )

        if self.input_device == "rgb":
            from retargeting.single_hand_detector import SingleHandDetector

            self.detector = SingleHandDetector("Right")
        elif self.input_device == "avp":
            from retargeting.avp_detector import AvpDetector

            self.detector = AvpDetector()
        else:
            raise NotImplementedError(f"Unsupported input_device: {self.input_device}")

        # variables
        initial_qpos = np.asarray(robot_config.initial_qpos, dtype=float)
        self.qpos_init = initial_qpos
        self.qpos_last = initial_qpos.copy()
        self.qpos_arm_last = initial_qpos[: self.arm_dof].copy()
        # print("self.qpos_arm_last: ", self.qpos_arm_last)

        self.robot_init_wrist_pose: Optional[np.ndarray] = None
        self.detection_source_init_wrist_pose: Optional[np.ndarray] = None

    def detect_observation(self, sensor_data: Any, camera_K: np.ndarray | None = None) -> RetargetObservation | None:
        """Convert one input frame into the common retargeting observation format.

        Args:
            sensor_data: Raw frame from the configured input device.
            camera_K: Optional camera intrinsic matrix required by RGB input.

        Returns:
            A unified retargeting observation, or None when no hand is detected.
        """
        if self.input_device == "rgb":
            if cv2 is None:
                raise ModuleNotFoundError("cv2 is required for RGB teleoperation, but it is not installed.")
            if camera_K is None:
                raise ValueError("RGB input requires camera_K.")

            _, hand_kps_in_wrist, keypoint_2d, wrist_pose_in_cam = self.detector.detect(sensor_data, camera_K)
            if hand_kps_in_wrist is None:
                return None

            wrist_pose_in_world = self.pose_from_detection_world_to_robot_world(wrist_pose_in_cam)
            return RetargetObservation(
                hand_kps_in_wrist=hand_kps_in_wrist * self.human_hand_scale,
                wrist_pose_in_world=wrist_pose_in_world,
                keypoint_2d=keypoint_2d,
                raw=sensor_data,
            )

        if self.input_device == "avp":
            _, hand_kps_in_wrist, _, wrist_pose_in_detection_world = self.detector.detect(sensor_data)
            if hand_kps_in_wrist is None:
                return None

            detected_wrist_pose_in_world = self.pose_from_detection_world_to_robot_world(
                wrist_pose_in_detection_world
            )
            target_wrist_pose_in_world = detected_wrist_pose_in_world.copy()
            if self.use_relative_pos:
                if self.robot_init_wrist_pose is None or self.detection_source_init_wrist_pose is None:
                    raise ValueError("Relative wrist alignment requires initial robot and detector wrist poses.")
                target_wrist_pose_in_world[:3, 3] += (
                    self.robot_init_wrist_pose[:3, 3] - self.detection_source_init_wrist_pose[:3, 3]
                )

            return RetargetObservation(
                hand_kps_in_wrist=hand_kps_in_wrist * self.human_hand_scale,
                wrist_pose_in_world=target_wrist_pose_in_world,
                raw=sensor_data,
            )

        raise NotImplementedError(f"Unsupported input_device: {self.input_device}")

    def pose_from_detection_world_to_robot_world(self, pose_in_detection_world):
        """Transform a detector-source wrist pose into the robot world frame.

        Args:
            pose_in_detection_world: 4x4 wrist pose matrix in the detector source world frame.

        Returns:
            4x4 wrist pose matrix in the robot world frame.
        """
        transform = posRotMat2Isometry3d(
            pos=[0, 0, 0],
            rot_mat=sciR.from_euler(
                "xyz", self.detection_source_config.rotation_euler_xyz_deg, degrees=True
            ).as_matrix(),
        )
        pose_in_world = transform @ pose_in_detection_world
        pose_in_world[:3, 3] += np.asarray(self.detection_source_config.translation)
        return pose_in_world

    def set_robot_init_wrist_pose(self, pose):
        """Store the robot wrist pose used for relative input alignment.

        Args:
            pose: 4x4 wrist pose matrix in the robot world frame.

        Returns:
            None.
        """
        self.robot_init_wrist_pose = pose.copy()

    def set_detection_source_init_wrist_pose(self, pose):
        """Store the initial detector-source wrist pose after robot-world conversion.

        Args:
            pose: 4x4 wrist pose matrix in the robot world frame.

        Returns:
            None.
        """
        self.detection_source_init_wrist_pose = pose.copy()

    def retarget_input(
        self,
        sensor_data: Any,
        camera_K: np.ndarray | None = None,
        show_detection: bool = False,
    ):
        """Detect and retarget one frame from the configured input device.

        Args:
            sensor_data: Raw frame from the configured input device.
            camera_K: Optional camera intrinsic matrix required by RGB input.
            show_detection: Whether to show the RGB detector overlay with cv2.

        Returns:
            Tuple of observation, qpos, and metric errors. All values are None
            when no hand is detected.
        """
        observation = self.detect_observation(sensor_data, camera_K=camera_K)
        if observation is None:
            print("hand is not detected.")
            if show_detection and self.input_device == "rgb":
                cv2.imshow("detection result", sensor_data)
                cv2.waitKey(1)
            return None, None, None

        qpos, err = self.retarget_observation(observation)
        if show_detection and self.input_device == "rgb":
            annotated_image = self.detector.draw_skeleton_on_image(
                sensor_data, observation.keypoint_2d, style="default"
            )
            cv2.imshow("detection result", annotated_image)
            cv2.waitKey(1)
        return observation, qpos, err

    def retarget_observation(self, observation: RetargetObservation):
        """Retarget one normalized observation with the shared hand-retargeting path.

        Args:
            observation: Unified observation containing hand keypoints and wrist pose in robot world coordinates.

        Returns:
            Tuple of retargeted qpos and metric errors.
        """
        return self.hand_retarget(observation.hand_kps_in_wrist, observation.wrist_pose_in_world)

    def _build_ref_values(self, hand_kps_in_world: np.ndarray, wrist_quat: np.ndarray) -> dict[str, Any]:
        """Build optimizer reference values from configured fingertips and objective weights.

        Args:
            hand_kps_in_world: Human hand keypoints transformed into the robot world frame.
            wrist_quat: Wrist orientation as a quaternion in optimizer WXYZ order.

        Returns:
            Reference value mapping consumed by the vector wrist joint optimizer.
        """
        num_fingertips = len(self.human_fingertip_indices)
        expected_link_count = 3 * num_fingertips
        if self.target_link_count != expected_link_count:
            raise ValueError(
                f"Retarget target link count {self.target_link_count} does not match "
                f"configured fingertip objective count {expected_link_count}."
            )

        objective = self.objective_config
        weights = objective.weights
        ref_link_vec = np.zeros((expected_link_count, 3))
        weights_links_vec = np.zeros(expected_link_count)
        wrist_pos = hand_kps_in_world[self.teleoperation_config.human_wrist_index, :]
        fingertip_pos = hand_kps_in_world[self.human_fingertip_indices]
        fingertip_base_pos = hand_kps_in_world[self.human_fingertip_base_indices]
        thumb_tip = fingertip_pos[0]
        primary_tip_pos = fingertip_pos[1:]

        thumb_primary_dist = np.linalg.norm(primary_tip_pos - thumb_tip.reshape(1, 3), axis=1)
        sigmoid_weights_thumb_primary = sigmoid(
            thumb_primary_dist,
            c=objective.pinch_transition_threshold,
            w=objective.pinch_sigmoid_slope,
        )
        sigmoid_weights_wrist_fingertips = sigmoid(
            np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0),
            c=objective.pinch_transition_threshold,
            w=-objective.pinch_sigmoid_slope,
        )

        wrist_fingertip_start = 1
        wrist_fingertip_end = wrist_fingertip_start + num_fingertips
        thumb_primary_start = wrist_fingertip_end
        thumb_primary_end = thumb_primary_start + num_fingertips - 1
        orientation_start = thumb_primary_end
        orientation_end = orientation_start + num_fingertips

        ref_link_vec[0, :] = thumb_tip
        weights_links_vec[0] = 0.0 if self.ablation_option in {5, 6, 8} else weights.world_thumb

        ref_link_vec[wrist_fingertip_start:wrist_fingertip_end, :] = fingertip_pos - wrist_pos
        if self.ablation_option in {1, 6, 8}:
            weights_links_vec[wrist_fingertip_start:wrist_fingertip_end] = weights.wrist_fingertip
        else:
            weights_links_vec[wrist_fingertip_start:wrist_fingertip_end] = (
                weights.wrist_fingertip * sigmoid_weights_wrist_fingertips
            )

        if self.ablation_option in {1, 6, 8}:
            weights_links_vec[thumb_primary_start:thumb_primary_end] = 0.0
        elif self.ablation_option == 2:
            ref_link_vec[thumb_primary_start:thumb_primary_end, :] = primary_tip_pos - thumb_tip.reshape(1, 3)
            weights_links_vec[thumb_primary_start:thumb_primary_end] = (
                weights.thumb_primary * sigmoid_weights_thumb_primary
            )
        else:
            rel_pos = primary_tip_pos - thumb_tip.reshape(1, 3)
            rel_dist = np.linalg.norm(rel_pos, axis=1)
            # The pinch rescale keeps very small thumb-finger distances at zero
            # while preserving the original distance once the transition threshold is exceeded.
            scale = objective.pinch_transition_threshold / (
                objective.pinch_transition_threshold - objective.pinch_contact_threshold
            )
            rescaled_rel_dist = scale * (rel_dist - objective.pinch_contact_threshold)
            rescaled_rel_dist[rel_dist < objective.pinch_contact_threshold] = 0
            rescaled_rel_dist[rel_dist > objective.pinch_transition_threshold] = rel_dist[
                rel_dist > objective.pinch_transition_threshold
            ]
            ref_link_vec[thumb_primary_start:thumb_primary_end, :] = (
                normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
            )
            weights_links_vec[thumb_primary_start:thumb_primary_end] = (
                weights.thumb_primary * sigmoid_weights_thumb_primary
            )

        if self.ablation_option in {3, 6, 8}:
            weights_links_vec[orientation_start:orientation_end] = 0.0
        elif self.ablation_option == 4:
            ref_link_vec[orientation_start:orientation_end, :] = fingertip_base_pos - wrist_pos.reshape(1, 3)
            weights_links_vec[orientation_start:orientation_end] = weights.fingertip_orientation
        else:
            ref_link_vec[orientation_start:orientation_end, :] = fingertip_pos - fingertip_base_pos
            weights_links_vec[orientation_start:orientation_end] = weights.fingertip_orientation

        if self.ablation_option in {5, 6, 8}:
            weights_wrist_rot = 0.0
        else:
            weights_wrist_rot = weights.wrist_rotation
        if self.ablation_option in {7, 8}:
            weights_joint_pos = np.zeros_like(self.qpos_init)
        else:
            weights_joint_pos = np.asarray(self.teleoperation_config.joint_position_weights, dtype=float)

        return {
            "links_vec": ref_link_vec,
            "wrist_quat": wrist_quat,
            "qpos_doa": self.qpos_init.copy(),
            "qpos_doa_last": self.qpos_last.copy(),
            "weights": {
                "links_vec": weights_links_vec,
                "wrist_rot": weights_wrist_rot,
                "joint_pos": weights_joint_pos,
                "joint_vel": np.asarray(self.teleoperation_config.joint_velocity_weights, dtype=float),
            },
        }

    def hand_retarget(self, hand_kps_in_wrist, wrist_pose_in_world):
        """Retarget normalized hand keypoints and wrist pose to robot qpos.

        Args:
            hand_kps_in_wrist: Human hand keypoints in the detected wrist frame.
            wrist_pose_in_world: 4x4 wrist pose matrix in the robot world frame.

        Returns:
            Tuple of retargeted qpos and metric errors.
        """
        hand_kps_in_world = transformPositions(hand_kps_in_wrist, target_frame_pose_inv=wrist_pose_in_world)
        wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)
        ref_values = self._build_ref_values(hand_kps_in_world, wrist_quat)

        # ---------------------joint/separate arm-hand retargeting----------------------

        t1 = time.perf_counter()

        if self.retarget_wrist_method == "joint":
            print("Joint retargeting.")
            qpos = self.optimizer.retarget(ref_values)

        elif self.retarget_wrist_method == "separate":
            print("Separate retargeting.")
            if self.arm_optimizer is None:
                raise ValueError("Separate wrist retargeting requires an arm optimizer.")

            # solve retargeting based on wrist pose to get qpos of robot arm (using the VectorWristJoint class)
            # to incorporate same cost terms for arm as the joint-retargeting
            ref_link_vec_arm = np.zeros((1, 3))
            ref_link_vec_arm[0, :] = hand_kps_in_world[self.teleoperation_config.human_wrist_index, :]
            weights_links_vec_arm = np.zeros((1))
            weights_links_vec_arm[0] = self.objective_config.weights.arm_link_vector
            arm_ref_values = {
                "links_vec": ref_link_vec_arm,
                "wrist_quat": wrist_quat,
                "qpos_doa": self.qpos_init.copy(),
                "qpos_doa_last": self.qpos_last.copy(),
                "weights": {
                    "links_vec": weights_links_vec_arm,
                    "wrist_rot": self.objective_config.weights.arm_wrist_rotation,
                    "joint_pos": ref_values["weights"]["joint_pos"],
                    "joint_vel": ref_values["weights"]["joint_vel"],
                },
            }

            qpos = self.arm_optimizer.retarget(arm_ref_values)
            qpos_arm = qpos[: self.arm_dof]
            ref_values["qpos_doa"][: self.arm_dof] = qpos_arm
            ref_values["qpos_doa_last"][: self.arm_dof] = qpos_arm
            qpos = self.optimizer.retarget(ref_values, qpos_arm)
            qpos = np.concatenate([qpos_arm, qpos[self.arm_dof :]])

        # -----------------------------Benchmark---------------------------------

        print(f"retarget opt time cost: {(time.perf_counter() - t1):.3f}")
        optimization_time = time.perf_counter() - t1

        position_err = self.robot_benchmark.position_error(qpos, hand_kps_in_world, 1)
        orientation_err = self.robot_benchmark.orientation_error(qpos, hand_kps_in_world, 1)
        relative_position_err = self.robot_benchmark.relative_position_error(qpos, hand_kps_in_world, 1)
        relative_position_to_wrist_err = self.robot_benchmark.relative_position_to_wrist_error(qpos, hand_kps_in_world, 1)

        err = {
            "position_err": position_err,
            "orientation_err": orientation_err,
            "relative_position_err": relative_position_err,
            "relative_position_to_wrist_err": relative_position_to_wrist_err,
            "optimization_time": optimization_time,
        }

        if self.smooth_output_qpos:
            qpos = self.ema_alpha * qpos + (1 - self.ema_alpha) * self.qpos_last
        self.qpos_last = qpos

        return qpos, err


def main():
    from retargeting.config import (
        load_detection_source_config,
        load_retargeting_config,
        load_retargeting_profile_config,
        load_robot_config,
        load_solver_config,
        load_teleoperation_mode_config,
    )
    from retargeting.robot_pinocchio import RobotPinocchio

    profile_config = load_retargeting_profile_config("configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml")
    detection_source_config = load_detection_source_config("configs/detection_sources/avp.yaml")
    robot_config = load_robot_config(profile_config.robot)
    retargeting_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config("configs/solvers/nlopt_slsqp.yaml")
    teleoperation_mode_config = load_teleoperation_mode_config("configs/teleoperation_modes/simulation.yaml")

    robot_model = RobotPinocchio(
        robot_file_path=robot_config.robot_file_path,
        robot_file_type=robot_config.model.type,
    )
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )

    # image_file = "data/test_teleop/rgb/image.png"
    # color_img = cv2.imread(image_file)
    # camera_K = np.array(
    #     [
    #         [605.2662353515625, 0.0, 319.0435485839844],
    #         [0.0, 603.8218994140625, 244.2542266845703],
    #         [0.0, 0.0, 1.0],
    #     ]
    # )

    # for i in range(1000):
    #     teleop.retarget_input(color_img.copy(), camera_K=camera_K, show_detection=True)

    # ------------------------------------------------------------

    teleop = RobotTeleoperation(
        robot_adaptor=robot_adaptor,
        robot_control=None,
        robot_config=robot_config,
        profile_config=profile_config,
        method_config=retargeting_config,
        detection_source_config=detection_source_config,
        teleoperation_mode_config=teleoperation_mode_config,
        solver_config=solver_config,
    )
    data = np.load("data/test_teleop/avp/data.npy", allow_pickle=True).item()
    stream_data = data["stream"]

    for i in range(len(stream_data) - 1):
        print("frame: ", i)
        teleop.retarget_input(stream_data[i])


if __name__ == "__main__":
    main()
