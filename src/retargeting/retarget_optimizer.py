from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from retargeting.optimization import create_callback_solver
from retargeting.robot_adaptor import RobotAdaptor
from retargeting.utils import utils_calc as ucalc
from retargeting.utils import utils_torch as utorch


def extract_solver_params(optimizer_params: Dict) -> Dict:
    """
    Args:
        optimizer_params: Runtime optimizer params containing `solver_params`.

    Returns:
        Backend-specific solver params to pass to the solver adapter.
    """
    solver_params = optimizer_params.get("solver_params")
    if not isinstance(solver_params, dict):
        raise ValueError("Runtime optimizer params must include solver_params.")
    return dict(solver_params)


class RetargetOptimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot_adaptor: RobotAdaptor,
        joint_limit_overrides: Optional[List[Dict]] = None,
        solver: str = "nlopt",
        solver_params: Optional[Dict] = None,
    ):
        """
        Args:
            robot_adaptor: Robot kinematics adaptor used by the objective functions.
            joint_limit_overrides: Optional per-joint bound overrides in optimizer DOA order.
            solver: Numeric solver backend name. Supported values are `nlopt` and `scipy_slsqp`.
            solver_params: Backend-specific solver params interpreted by the solver adapter.

        Returns:
            None.
        """
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model

        self.solver = solver.lower()
        self.opt_dim = self.robot_adaptor.doa
        self.opt = create_callback_solver(solver, self.opt_dim)
        self.opt.configure({} if solver_params is None else solver_params)
        self.joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        # self.apply_joint_limit_overrides(
        #     self.default_joint_limit_overrides if joint_limit_overrides is None else joint_limit_overrides
        # )
        self.set_joint_limit(self.joint_limits)

    def apply_joint_limit_overrides(self, joint_limit_overrides: List[Dict]):
        """
        Args:
            joint_limit_overrides: List of dictionaries with `indices`, optional `lower`, and optional `upper`.

        Returns:
            None.
        """
        for override in joint_limit_overrides:
            indices = override["indices"]
            if override.get("lower") is not None:
                self.joint_limits[indices, 0] = override["lower"]
            if override.get("upper") is not None:
                self.joint_limits[indices, 1] = override["upper"]

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """
        Args:
            joint_limits: shape (n_joint_dof, 2)
            epsilon: Small bound padding used to match the previous nlopt behavior.

        Returns:
            None.
        """
        if joint_limits.shape != (self.opt_dim, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dim, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def retarget(
        self,
        ref_values: Dict[str, np.ndarray],
        arm_qpos: Optional[np.ndarray] = None,
        fixed_qpos_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            ref_values: Reference values used by the selected retargeting objective.
            arm_qpos: Optional fixed joint positions injected into `ref_values`.
            fixed_qpos_indices: Optional DOA indices where `arm_qpos` should be applied.

        Returns:
            Optimized qpos in robot adaptor DOA order.
        """
        if arm_qpos is not None:
            fixed_qpos = np.asarray(arm_qpos, dtype=np.float32)
            if fixed_qpos_indices is None:
                fixed_qpos_indices = np.arange(len(fixed_qpos), dtype=int)
            fixed_qpos_indices = np.asarray(fixed_qpos_indices, dtype=int)
            if fixed_qpos_indices.shape != fixed_qpos.shape:
                raise ValueError(
                    "fixed_qpos_indices must have the same shape as arm_qpos, "
                    f"got {fixed_qpos_indices.shape} and {fixed_qpos.shape}."
                )
            ref_values["fixed_qpos"] = fixed_qpos
            ref_values["fixed_qpos_indices"] = fixed_qpos_indices

        objective_fn = self.get_objective_function(ref_values)
        self.opt.set_min_objective(objective_fn)

        x_init = ref_values["qpos_doa_last"]
        try:
            x_opt = self.opt.optimize(x_init)
            qpos_doa = x_opt
        except ValueError as e:
            print(e)
            qpos_doa = x_init

        qpos_doa = np.clip(qpos_doa, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return np.array(qpos_doa, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        pass


class DexPilotOptimizer(RetargetOptimizer):
    retargeting_type = "DEXPILOT"

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

        # self.origin_links_name = targets["origin_links_name"]
        # self.task_links_name = targets["task_links_name"]
        self.fingertip_links_name = targets["fingertip_links_name"]  # determines the number of fingers
        self.wrist_link_name = targets["wrist_link_name"]

        self.num_fingers = len(self.fingertip_links_name)
        origin_link_index, task_link_index = self.generate_link_indices(self.num_fingers)

        links_name = [self.wrist_link_name] + self.fingertip_links_name
        target_origin_links_name = [links_name[index] for index in origin_link_index]
        target_origin_links_name.append("world")
        target_task_links_name = [links_name[index] for index in task_link_index]
        target_task_links_name.append(self.wrist_link_name)

        self.origin_links_name = target_origin_links_name
        self.task_links_name = target_task_links_name

        self.computed_links_name = list(set(target_origin_links_name).union(set(target_task_links_name)))
        self.origin_links_indice = torch.tensor(
            [self.computed_links_name.index(name) for name in target_origin_links_name]
        )
        self.task_links_indice = torch.tensor([self.computed_links_name.index(name) for name in target_task_links_name])

        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

        # params for DexPilot
        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.project_dist = params["project_dist"]
        self.escape_dist = params["escape_dist"]
        self.eta1 = params["eta1"]
        self.eta2 = params["eta2"]

        self.projected, self.s2_project_index_origin, self.s2_project_index_task, self.projected_dist = (
            self.set_dexpilot_cache(self.num_fingers, self.eta1, self.eta2)
        )

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        target_vector = ref_values["target_vector"]
        qpos_doa_init = ref_values["qpos_doa_last"]
        wrist_link_pos_human = ref_values["wrist_link_pos"]
        wrist_link_quat_human = ref_values["wrist_quat"]

        # to torch (do not requires grad)
        qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(wrist_link_quat_human).requires_grad_(False)
        wrist_link_pos_human_torch = torch.as_tensor(wrist_link_pos_human).requires_grad_(False)

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
        )

        # Compute reference distance vector
        normal_vec = target_vector  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
        reference_vec = np.concatenate([reference_vec, wrist_link_pos_human[None, :]], axis=0)  # (11, 3)

        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

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
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
            wrist_quat_torch = utorch.matrix_to_quaternion(wrist_pose_torch[:3, :3])
            wrist_quat_torch.requires_grad_(True)
            wrist_pos_torch = wrist_pose_torch[:3, 3]
            wrist_pos_torch.requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            origin_links_pos = links_pos_torch[self.origin_links_indice, :]
            task_links_pos = links_pos_torch[self.task_links_indice, :]
            links_vec = task_links_pos - origin_links_pos
            links_vec_err = torch.norm(links_vec - torch_target_vec, dim=-1)

            # costs with weights
            huber_distance = (
                self.huber_loss(links_vec_err, torch.zeros_like(links_vec_err)) * weight / (links_vec.shape[0])
            ).sum()
            links_vec_cost = huber_distance.sum()
            action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

            # wrist_pos_cost = 0.5 * torch.norm(wrist_pos_torch - wrist_link_pos_human_torch, dim=-1) ** 2
            wrist_rot_err = utorch.quaternion_angular_error(
                ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
            ).squeeze()
            wrist_rot_cost = 0.1 * wrist_rot_err**2

            # total cost
            # total_cost = links_vec_cost + action_cost + wrist_pos_cost + wrist_rot_cost
            total_cost = links_vec_cost + action_cost + wrist_rot_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # shadow finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    links_jaco_list.append(link_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 3, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                # action gradient w.r.t. joint pos
                action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # wrist_pos_grad = wrist_pos_torch.grad.cpu().numpy().reshape(-1)
                # wrist_jaco = self.robot_model.get_frame_space_jacobian(self.wrist_link_name)[
                #     :3, :
                # ]  # 取线速度部分 (3, dof)
                # wrist_jaco_doa = self.robot_adaptor.backward_jacobian(wrist_jaco)  # 转换到 DOA 维度 (3, doa)
                # wrist_pos_grad = wrist_pos_grad @ wrist_jaco_doa  # (3,) @ (3, 23) -> (23,)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

                # total_grad = link_vec_grad + action_grad + wrist_pos_grad + wrist_rot_grad
                total_grad = link_vec_grad + action_grad + wrist_rot_grad

                if "fixed_qpos_indices" in ref_values:
                    total_grad[ref_values["fixed_qpos_indices"]] = 0.0

                grad[:] = total_grad

                # total gradient
                # grad[:] = link_vec_grad[:] + action_grad[:] + wrist_pos_grad[:] + wrist_rot_grad[:]

            return total_cost.cpu().detach().item()

        return objective


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
