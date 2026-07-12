"""Device-independent retargeting solver implementation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from retargeting.config import (
    RetargetingConfig,
    RetargetingObjectiveConfig,
    RetargetingProfileConfig,
    RobotConfig,
    SolverConfig,
    default_solver_config,
)
from retargeting.core.types import RetargetingResult
from retargeting.inputs import HandObservation
from retargeting.core.optimizers import get_optimizer_class
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.utils.utils_calc import quatXYZW2WXYZ, sciR, transformPositions


def sigmoid(x: np.ndarray, c: float = 0.0, w: float = 1.0) -> np.ndarray:
    """Compute the sigmoid used by the pinch-dependent objective weights.

    Args:
        x: Values to transform.
        c: Sigmoid center.
        w: Sigmoid slope.

    Returns:
        Elementwise sigmoid values.
    """
    return 1 / (1 + np.exp(w * (x - c)))


def merge_objective_solver_params(objective_params: dict, solver_config: SolverConfig) -> dict:
    """Merge method and solver configuration for an optimizer instance.

    Args:
        objective_params: Method-specific optimizer parameters.
        solver_config: Solver backend selection and backend parameters.

    Returns:
        Runtime parameter mapping consumed by optimizer classes.
    """
    params = dict(objective_params)
    params["solver"] = solver_config.name
    params["solver_params"] = dict(solver_config.params)
    return params


class Retargeter:
    """Map canonical hand observations to raw robot joint configurations.

    This class deliberately has no detector, camera, ROS, robot-control, display,
    or output-filter dependency.  Its only state is the previous configuration
    needed by the configured retargeting objective.
    """

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        robot_config: RobotConfig,
        profile_config: RetargetingProfileConfig,
        method_config: RetargetingConfig,
        solver_config: SolverConfig | None = None,
    ) -> None:
        """Create a retargeting solver from embodiment and method configuration.

        Args:
            robot_adaptor: Kinematics adapter for the target robot.
            robot_config: Target robot embodiment configuration.
            profile_config: Robot-method target and objective configuration.
            method_config: Optimizer method and ablation configuration.
            solver_config: Optional numerical solver backend configuration.

        Returns:
            None.
        """
        for name, value in {
            "robot_config": robot_config,
            "profile_config": profile_config,
            "method_config": method_config,
        }.items():
            if value is None:
                raise ValueError(f"Retargeter requires {name}.")
        robot_config.validate()
        profile_config.validate(robot_config)
        method_config.validate()

        self.robot_adaptor = robot_adaptor
        self.robot_config = robot_config
        self.profile_config = profile_config
        self.method_config = method_config
        self.solver_config = default_solver_config() if solver_config is None else solver_config
        self.solver_config.validate()
        self.objective_config: RetargetingObjectiveConfig = profile_config.objective
        self.retargeting_config = profile_config.retargeting
        self.ablation_option = method_config.ablation_option
        self.retarget_wrist_method = "separate" if self.ablation_option in {5, 6, 8} else "joint"
        self.arm_dof = self.retargeting_config.arm_dof

        benchmark_config = robot_config.benchmark
        thumb_fingertip = benchmark_config.thumb_fingertip
        primary_fingertips = benchmark_config.primary_fingertips
        self.human_fingertip_indices = np.asarray(
            [thumb_fingertip.human_tip_index, *[item.human_tip_index for item in primary_fingertips]], dtype=int
        )
        self.human_fingertip_base_indices = np.asarray(
            [
                thumb_fingertip.human_direction_base_index,
                *[item.human_direction_base_index for item in primary_fingertips],
            ],
            dtype=int,
        )

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
            {"indices": list(item.indices), "lower": item.lower, "upper": item.upper}
            for item in profile_config.joint_limit_overrides
        ]
        optimizer_class = get_optimizer_class(method_config.optimizer_class)
        self.optimizer = optimizer_class(
            robot_adaptor=robot_adaptor,
            targets=targets,
            params=params,
            joint_limit_overrides=joint_limit_overrides,
            solver=self.solver_config.name,
        )
        self.arm_optimizer = None
        if self.retarget_wrist_method == "separate":
            self.arm_optimizer = optimizer_class(
                robot_adaptor=robot_adaptor,
                targets={
                    "origin_links_name": ["world"],
                    "task_links_name": [target_config.wrist_link_name],
                    "wrist_link_name": target_config.wrist_link_name,
                },
                params=params,
                joint_limit_overrides=joint_limit_overrides,
                solver=self.solver_config.name,
            )

        self.qpos_init = np.asarray(robot_config.initial_qpos, dtype=float)
        self.previous_qpos = self.qpos_init.copy()

    def reset(self, previous_qpos: np.ndarray | None = None) -> None:
        """Reset the algorithm's temporal reference configuration.

        Args:
            previous_qpos: Optional configuration to use as the next-frame reference.

        Returns:
            None.
        """
        self.previous_qpos = self.qpos_init.copy() if previous_qpos is None else np.asarray(previous_qpos, dtype=float).copy()

    def _build_ref_values(
        self,
        hand_kps_in_world: np.ndarray,
        wrist_quat: np.ndarray,
        previous_qpos: np.ndarray,
    ) -> dict[str, Any]:
        """Build optimizer references from a canonical hand observation.

        Args:
            hand_kps_in_world: Human hand keypoints in robot world coordinates.
            wrist_quat: Wrist orientation in optimizer WXYZ convention.
            previous_qpos: Previous configuration used by temporal regularization.

        Returns:
            Optimizer reference values for the configured objective.
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
        wrist_pos = hand_kps_in_world[self.retargeting_config.human_wrist_index, :]
        fingertip_pos = hand_kps_in_world[self.human_fingertip_indices]
        fingertip_base_pos = hand_kps_in_world[self.human_fingertip_base_indices]
        thumb_tip = fingertip_pos[0]
        primary_tip_pos = fingertip_pos[1:]
        thumb_primary_dist = np.linalg.norm(primary_tip_pos - thumb_tip.reshape(1, 3), axis=1)
        sigmoid_weights_thumb_primary = sigmoid(
            thumb_primary_dist, c=objective.pinch_transition_threshold, w=objective.pinch_sigmoid_slope
        )
        sigmoid_weights_wrist_fingertips = sigmoid(
            np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist]),
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
        weights_links_vec[wrist_fingertip_start:wrist_fingertip_end] = weights.wrist_fingertip * (
            1.0 if self.ablation_option in {1, 6, 8} else sigmoid_weights_wrist_fingertips
        )
        if self.ablation_option in {1, 6, 8}:
            weights_links_vec[thumb_primary_start:thumb_primary_end] = 0.0
        elif self.ablation_option == 2:
            ref_link_vec[thumb_primary_start:thumb_primary_end, :] = primary_tip_pos - thumb_tip.reshape(1, 3)
            weights_links_vec[thumb_primary_start:thumb_primary_end] = weights.thumb_primary * sigmoid_weights_thumb_primary
        else:
            rel_pos = primary_tip_pos - thumb_tip.reshape(1, 3)
            rel_dist = np.linalg.norm(rel_pos, axis=1)
            scale = objective.pinch_transition_threshold / (
                objective.pinch_transition_threshold - objective.pinch_contact_threshold
            )
            rescaled_rel_dist = scale * (rel_dist - objective.pinch_contact_threshold)
            rescaled_rel_dist[rel_dist < objective.pinch_contact_threshold] = 0
            rescaled_rel_dist[rel_dist > objective.pinch_transition_threshold] = rel_dist[
                rel_dist > objective.pinch_transition_threshold
            ]
            ref_link_vec[thumb_primary_start:thumb_primary_end, :] = normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
            weights_links_vec[thumb_primary_start:thumb_primary_end] = weights.thumb_primary * sigmoid_weights_thumb_primary
        if self.ablation_option in {3, 6, 8}:
            weights_links_vec[orientation_start:orientation_end] = 0.0
        elif self.ablation_option == 4:
            ref_link_vec[orientation_start:orientation_end, :] = fingertip_base_pos - wrist_pos.reshape(1, 3)
            weights_links_vec[orientation_start:orientation_end] = weights.fingertip_orientation
        else:
            ref_link_vec[orientation_start:orientation_end, :] = fingertip_pos - fingertip_base_pos
            weights_links_vec[orientation_start:orientation_end] = weights.fingertip_orientation
        weights_wrist_rot = 0.0 if self.ablation_option in {5, 6, 8} else weights.wrist_rotation
        weights_joint_pos = (
            np.zeros_like(self.qpos_init)
            if self.ablation_option in {7, 8}
            else np.asarray(self.retargeting_config.joint_position_weights, dtype=float)
        )
        return {
            "links_vec": ref_link_vec,
            "wrist_quat": wrist_quat,
            "qpos_doa": self.qpos_init.copy(),
            "qpos_doa_last": previous_qpos.copy(),
            "weights": {
                "links_vec": weights_links_vec,
                "wrist_rot": weights_wrist_rot,
                "joint_pos": weights_joint_pos,
                "joint_vel": np.asarray(self.retargeting_config.joint_velocity_weights, dtype=float),
            },
        }

    def solve(
        self,
        observation: HandObservation,
        previous_qpos: np.ndarray | None = None,
    ) -> RetargetingResult:
        """Solve one canonical observation without input or output side effects.

        Args:
            observation: Hand keypoints in wrist coordinates and wrist pose in robot world coordinates.
            previous_qpos: Optional external temporal reference, normally the last commanded qpos.

        Returns:
            Raw optimized qpos and solver timing diagnostics.
        """
        temporal_reference = self.previous_qpos if previous_qpos is None else np.asarray(previous_qpos, dtype=float)
        hand_kps_in_world = transformPositions(observation.keypoints_wrist, target_frame_pose_inv=observation.wrist_pose_world)
        wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(observation.wrist_pose_world[:3, :3]).as_quat())
        ref_values = self._build_ref_values(hand_kps_in_world, wrist_quat, temporal_reference)
        started_at = time.perf_counter()
        if self.retarget_wrist_method == "joint":
            qpos = self.optimizer.retarget(ref_values)
        else:
            if self.arm_optimizer is None:
                raise ValueError("Separate wrist retargeting requires an arm optimizer.")
            arm_ref_values = {
                "links_vec": np.asarray([hand_kps_in_world[self.retargeting_config.human_wrist_index, :]]),
                "wrist_quat": wrist_quat,
                "qpos_doa": self.qpos_init.copy(),
                "qpos_doa_last": temporal_reference.copy(),
                "weights": {
                    "links_vec": np.asarray([self.objective_config.weights.arm_link_vector]),
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
        optimization_time = time.perf_counter() - started_at
        self.previous_qpos = np.asarray(qpos, dtype=float).copy()
        return RetargetingResult(qpos=np.asarray(qpos, dtype=float), diagnostics={"optimization_time": optimization_time})
