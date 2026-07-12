"""Vector wrist-joint retargeting objective implementations."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from retargeting.core.optimizers.base import RetargetOptimizer, extract_solver_params
from retargeting.core.kinematics.adaptor import RobotAdaptor
from retargeting.utils import utils_calc as ucalc
from retargeting.utils import utils_torch as utorch


def _ordered_unique(values: List[str]) -> List[str]:
    """Deduplicate names while preserving their first-occurrence order.

    Args:
        values: Names that may contain duplicates.

    Returns:
        Ordered unique names.
    """
    return list(dict.fromkeys(values))


class VectorWristJointOptimizer(RetargetOptimizer):
    retargeting_type = "VECTOR_WRIST_JOINT"

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        targets: Dict,
        params: Dict,
        joint_limit_overrides: Optional[List[Dict]] = None,
        solver: str = "nlopt",
    ):
        super().__init__(
            robot_adaptor,
            joint_limit_overrides=joint_limit_overrides,
            solver=solver,
            solver_params=extract_solver_params(params),
        )

        self.origin_links_name = targets["origin_links_name"]
        self.task_links_name = targets["task_links_name"]
        self.wrist_link_name = targets["wrist_link_name"]

        self.computed_links_name = list(set(self.origin_links_name + self.task_links_name + [self.wrist_link_name]))
        self.origin_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.origin_links_name])
        self.task_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.task_links_name])
        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        ref_links_vec = ref_values["links_vec"]
        ref_wrist_quat = ref_values["wrist_quat"]  # (w, x, y, z)
        ref_qpos_doa = ref_values["qpos_doa"]
        qpos_doa_last = ref_values["qpos_doa_last"]
        # to torch (do not requires grad)
        ref_links_vec_torch = torch.as_tensor(ref_links_vec).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(ref_wrist_quat).requires_grad_(False)
        ref_qpos_doa_torch = torch.as_tensor(ref_qpos_doa).requires_grad_(False)
        qpos_doa_last_torch = torch.as_tensor(qpos_doa_last).requires_grad_(False)

        # cost weights
        weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"])
        weight_wrist_rot = ref_values["weights"]["wrist_rot"]
        weight_joint_vel = torch.as_tensor(ref_values["weights"]["joint_vel"]).requires_grad_(False)
        weight_joint_pos = torch.as_tensor(ref_values["weights"]["joint_pos"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa[:] = x

            if "fixed_qpos" in ref_values:
                fixed_qpos_indices = ref_values["fixed_qpos_indices"]
                qpos_doa[fixed_qpos_indices] = ref_values["fixed_qpos"]

            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
            wrist_quat_torch = utorch.matrix_to_quaternion(wrist_pose_torch[:3, :3])
            wrist_quat_torch.requires_grad_(True)
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            origin_links_pos = links_pos_torch[self.origin_links_idx, :]
            task_links_pos = links_pos_torch[self.task_links_idx, :]
            links_vec = task_links_pos - origin_links_pos
            links_vec_err = torch.norm(links_vec - ref_links_vec_torch, dim=-1)
            wrist_rot_err = utorch.quaternion_angular_error(
                ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
            ).squeeze()
            qpos_doa_err = qpos_doa_torch - ref_qpos_doa_torch
            qvel_doa_torch = qpos_doa_torch - qpos_doa_last_torch

            # costs with weights
            links_vec_cost = self.huber_loss(weight_links_vec * links_vec_err, torch.zeros_like(links_vec_err))
            wrist_rot_cost = weight_wrist_rot * wrist_rot_err**2
            # print(qpos_doa_err.shape, weight_joint_pos.shape)
            joint_pos_cost = self.huber_loss(weight_joint_pos * qpos_doa_err, torch.zeros_like(qpos_doa_err))
            joint_vel_cost = self.huber_loss(weight_joint_vel * qvel_doa_torch, torch.zeros_like(qvel_doa_torch))

            # total cost
            total_cost = links_vec_cost + wrist_rot_cost + joint_pos_cost + joint_vel_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    links_jaco_list.append(link_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 6, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

                # gradient w.r.t. joint pos
                grad_qpos_doa = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # total gradient
                grad[:] = link_vec_grad[:] + wrist_rot_grad[:] + grad_qpos_doa[:]

                if "fixed_qpos_indices" in ref_values:
                    grad[ref_values["fixed_qpos_indices"]] = 0.0

            return total_cost.cpu().detach().item()

        return objective


class VectorWristJointOptimizerV2(RetargetOptimizer):
    retargeting_type = "VECTOR_WRIST_JOINT"

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        targets: Dict,
        params: Dict,
        joint_limit_overrides: Optional[List[Dict]] = None,
        solver: str = "nlopt",
    ):
        """
        Args:
            robot_adaptor: Robot kinematics adaptor used by the objective function.
            targets: Mapping with origin link names, task link names, and wrist link name.
            params: Optimizer params containing `huber_delta` and runtime solver params.
            joint_limit_overrides: Optional per-joint bound overrides in optimizer DOA order.
            solver: Numeric solver backend name.

        Returns:
            None.
        """
        super().__init__(
            robot_adaptor,
            joint_limit_overrides=joint_limit_overrides,
            solver=solver,
            solver_params=extract_solver_params(params),
        )

        self.origin_links_name = targets["origin_links_name"]
        self.task_links_name = targets["task_links_name"]
        self.wrist_link_name = targets["wrist_link_name"]

        self.computed_links_name = _ordered_unique(
            self.origin_links_name + self.task_links_name + [self.wrist_link_name]
        )
        self.origin_links_idx_np = np.asarray(
            [self.computed_links_name.index(name) for name in self.origin_links_name], dtype=int
        )
        self.task_links_idx_np = np.asarray(
            [self.computed_links_name.index(name) for name in self.task_links_name], dtype=int
        )
        self.origin_links_idx = torch.as_tensor(self.origin_links_idx_np, dtype=torch.long)
        self.task_links_idx = torch.as_tensor(self.task_links_idx_np, dtype=torch.long)
        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)
        self.actuated_joint_model_idx = np.asarray(self.robot_adaptor.actuated_joints_model_idx, dtype=int)

        self.computed_frame_indices = None
        if hasattr(self.robot_model, "get_frames_index"):
            self.computed_frame_indices = self.robot_model.get_frames_index(self.computed_links_name)

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])

    def _write_qpos_dof(self, qpos_doa: np.ndarray, qpos_dof: np.ndarray) -> None:
        """
        Args:
            qpos_doa: Joint values in robot adaptor DOA order.
            qpos_dof: Reusable output buffer in robot model DOF order.

        Returns:
            None.
        """
        qpos_dof.fill(0.0)
        qpos_dof[self.actuated_joint_model_idx] = qpos_doa

    def _get_frame_pose(self, link_idx: int) -> np.ndarray:
        """
        Args:
            link_idx: Index into `self.computed_links_name`.

        Returns:
            Current frame pose as a homogeneous matrix.
        """
        if self.computed_frame_indices is not None and hasattr(self.robot_model, "get_frame_pose_by_id"):
            return self.robot_model.get_frame_pose_by_id(self.computed_frame_indices[link_idx])
        return self.robot_model.get_frame_pose(self.computed_links_name[link_idx])

    def _get_frame_jacobian(self, link_idx: int) -> np.ndarray:
        """
        Args:
            link_idx: Index into `self.computed_links_name`.

        Returns:
            Current frame spatial Jacobian in robot model DOF order.
        """
        if self.computed_frame_indices is not None and hasattr(self.robot_model, "get_frame_space_jacobian_by_id"):
            return self.robot_model.get_frame_space_jacobian_by_id(self.computed_frame_indices[link_idx])
        return self.robot_model.get_frame_space_jacobian(self.computed_links_name[link_idx])

    def _collect_link_state(
        self, qpos_dof: np.ndarray, need_grad: bool
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Args:
            qpos_dof: Joint values in robot model DOF order.
            need_grad: Whether spatial Jacobians are required for this objective evaluation.

        Returns:
            Tuple of link positions, wrist rotation matrix, and optional link Jacobians in DOA order.
        """
        if need_grad:
            self.robot_model.compute_jacobians(qpos_dof)
        else:
            self.robot_model.compute_forward_kinematics(qpos_dof)

        links_pos = np.empty((len(self.computed_links_name), 3), dtype=float)
        wrist_rot = None
        for link_idx in range(len(self.computed_links_name)):
            pose = self._get_frame_pose(link_idx)
            links_pos[link_idx, :] = pose[:3, 3]
            if link_idx == self.wrist_link_idx:
                wrist_rot = pose[:3, :3]

        links_jaco = None
        if need_grad:
            links_jaco_model = np.stack(
                [self._get_frame_jacobian(link_idx) for link_idx in range(len(self.computed_links_name))],
                axis=0,
            )
            links_jaco = self.robot_adaptor.backward_jacobian(links_jaco_model)

        return links_pos, wrist_rot, links_jaco

    def _validate_ref_values(self, ref_values: Dict[str, np.ndarray]) -> None:
        """
        Args:
            ref_values: Reference values used by the objective function.

        Returns:
            None.
        """
        num_link_pairs = len(self.origin_links_name)
        if np.asarray(ref_values["links_vec"]).shape != (num_link_pairs, 3):
            raise ValueError(f"links_vec must have shape {(num_link_pairs, 3)}.")
        if np.asarray(ref_values["weights"]["links_vec"]).shape != (num_link_pairs,):
            raise ValueError(f"weights['links_vec'] must have shape {(num_link_pairs,)}.")
        for key in ["qpos_doa", "qpos_doa_last"]:
            if np.asarray(ref_values[key]).shape != (self.opt_dim,):
                raise ValueError(f"{key} must have shape {(self.opt_dim,)}.")
        for key in ["joint_pos", "joint_vel"]:
            if np.asarray(ref_values["weights"][key]).shape != (self.opt_dim,):
                raise ValueError(f"weights['{key}'] must have shape {(self.opt_dim,)}.")

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        """
        Args:
            ref_values: Reference values used by the vector, wrist, and joint objective terms.

        Returns:
            Objective callback with the nlopt-style signature `(x, grad) -> cost`.
        """
        self._validate_ref_values(ref_values)

        ref_links_vec_torch = torch.as_tensor(ref_values["links_vec"]).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(ref_values["wrist_quat"]).requires_grad_(False)
        ref_qpos_doa_torch = torch.as_tensor(ref_values["qpos_doa"]).requires_grad_(False)
        qpos_doa_last_torch = torch.as_tensor(ref_values["qpos_doa_last"]).requires_grad_(False)

        weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"]).requires_grad_(False)
        weight_wrist_rot = ref_values["weights"]["wrist_rot"]
        weight_joint_vel = torch.as_tensor(ref_values["weights"]["joint_vel"]).requires_grad_(False)
        weight_joint_pos = torch.as_tensor(ref_values["weights"]["joint_pos"]).requires_grad_(False)

        fixed_qpos = ref_values.get("fixed_qpos")
        fixed_qpos_indices = ref_values.get("fixed_qpos_indices")

        qpos_doa = np.zeros(self.opt_dim)
        qpos_dof = np.zeros(self.robot_model.dof)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            """
            Args:
                x: Current optimization vector in robot adaptor DOA order.
                grad: Gradient output buffer. Empty when the solver requests cost only.

            Returns:
                Scalar objective value.
            """
            need_grad = grad.size > 0
            qpos_doa[:] = x

            if fixed_qpos is not None:
                qpos_doa[fixed_qpos_indices] = fixed_qpos

            self._write_qpos_dof(qpos_doa, qpos_dof)
            links_pos, wrist_rot, links_jaco = self._collect_link_state(qpos_dof, need_grad)

            links_pos_torch = torch.as_tensor(links_pos)
            wrist_quat_torch = utorch.matrix_to_quaternion(torch.as_tensor(wrist_rot))
            qpos_doa_torch = torch.as_tensor(qpos_doa)

            if need_grad:
                links_pos_torch.requires_grad_(True)
                wrist_quat_torch.requires_grad_(True)
                qpos_doa_torch.requires_grad_(True)

            origin_links_pos = links_pos_torch[self.origin_links_idx, :]
            task_links_pos = links_pos_torch[self.task_links_idx, :]
            links_vec = task_links_pos - origin_links_pos
            if need_grad:
                links_vec.retain_grad()

            links_vec_err = torch.norm(links_vec - ref_links_vec_torch, dim=-1)
            wrist_rot_err = utorch.quaternion_angular_error(
                ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
            ).squeeze()
            qpos_doa_err = qpos_doa_torch - ref_qpos_doa_torch
            qvel_doa_torch = qpos_doa_torch - qpos_doa_last_torch

            links_vec_cost = self.huber_loss(weight_links_vec * links_vec_err, torch.zeros_like(links_vec_err))
            wrist_rot_cost = weight_wrist_rot * wrist_rot_err**2
            joint_pos_cost = self.huber_loss(weight_joint_pos * qpos_doa_err, torch.zeros_like(qpos_doa_err))
            joint_vel_cost = self.huber_loss(weight_joint_vel * qvel_doa_torch, torch.zeros_like(qvel_doa_torch))
            total_cost = links_vec_cost + wrist_rot_cost + joint_pos_cost + joint_vel_cost

            if need_grad:
                total_cost.backward()

                grad_links_vec = links_vec.grad.cpu().numpy()
                pair_jaco_pos = links_jaco[self.task_links_idx_np, :3, :] - links_jaco[self.origin_links_idx_np, :3, :]
                link_vec_grad = np.einsum("pk,pkd->d", grad_links_vec, pair_jaco_pos)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

                grad_qpos_doa = qpos_doa_torch.grad.cpu().numpy().reshape(-1)
                grad[:] = link_vec_grad + wrist_rot_grad + grad_qpos_doa

                if fixed_qpos_indices is not None:
                    grad[fixed_qpos_indices] = 0.0

            return total_cost.cpu().detach().item()

        return objective
