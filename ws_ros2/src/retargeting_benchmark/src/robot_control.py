import os
import sys
import time
from threading import Thread
from typing import List, Optional, Union

import nlopt
import numpy as np
import rclpy
from pynput import keyboard
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_adaptor import RobotAdaptor
from robot_mujoco import RobotMujoco
from robot_pinocchio import RobotPinocchio
from robot_pinocchio_env import RobotPinocchioEnv
from robot_real import RobotReal
from scipy.optimize import minimize
from utils.utils_calc import (
    isometry3dToPosOri,
    jacoLeftBCHInverse,
    posOri2Isometry3d,
    sciR,
)
from utils.utils_keyboard import KeyboardListener


class RobotControl:
    """
    Robot controller connected to the simulated robot or real robot.
    """

    def __init__(
        self,
        robot_model: RobotPinocchio,
        robot_adaptor: RobotAdaptor,
        initial_qpos: Optional[np.ndarray] = None,
        arm_dof: int = 7,
        use_hardware: bool = False,
        use_virtual_hardware: bool = True,
        use_high_freq_interp: bool = False,
        node: Optional[Node] = None,
    ):
        self.robot_model = robot_model
        self.robot_adaptor = robot_adaptor
        self.use_hardware = use_hardware
        self.use_ros = self.use_hardware or False

        if self.use_ros:
            self.node = node

        if self.use_hardware:
            self.env = RobotReal(
                self.node, virtual_hardware=use_virtual_hardware, use_high_freq_interp=use_high_freq_interp
            )
            self.env.wait_for_initialization()
        else:
            self.env = RobotPinocchioEnv(robot_model, node)

        if not 0 < arm_dof <= self.robot_adaptor.doa:
            raise ValueError(f"arm_dof must be in [1, {self.robot_adaptor.doa}], got {arm_dof}.")
        self.arm_dof = arm_dof
        if initial_qpos is None:
            self.init_joint_pos = np.zeros((self.robot_adaptor.doa))
            self.init_joint_pos[:7] = [
                0.21511886,
                -0.15896096,
                -0.80094644,
                -2.59708501,
                -0.36324861,
                2.30919053,
                0.4965313,
            ]
            self.init_joint_pos[7:] = [
                0,
                0.1,
                0.1,
                0.1,
                0,
                0.1,
                0.1,
                0.1,
                0,
                0.1,
                0.1,
                0,
                0.1,
                0,
                0.1,
                0.1,
            ]
        else:
            self.init_joint_pos = np.asarray(initial_qpos, dtype=float).copy()
            if len(self.init_joint_pos) != self.robot_adaptor.doa:
                raise ValueError(
                    f"initial_qpos has {len(self.init_joint_pos)} values but robot adaptor has {self.robot_adaptor.doa}."
                )

    def single_link_ik_nlopt(
        self,
        target_link_name: str,
        ref_link_pose: np.ndarray,
        weights: Optional[np.ndarray] = np.diag([10, 10, 10, 1, 1, 1]),
        qpos_init: Optional[np.ndarray] = None,
    ):
        """
        Args:
            qpos_init: if None, use the current joint positions.
        """
        ref_link_pos, ref_link_ori = isometry3dToPosOri(ref_link_pose)
        if qpos_init is None:
            qpos_init = self.get_joint_pos()

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa = x
            qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)

            # -------------- objective --------------
            self.robot_model.compute_forward_kinematics(qpos_dof)
            link_pose = self.robot_model.get_frame_pose(target_link_name)
            link_pos, link_ori = isometry3dToPosOri(link_pose)
            link_pos_err = link_pos - ref_link_pos
            link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
            err = np.concatenate([link_pos_err, link_ori_err], axis=0)
            err = err.reshape(-1, 1)

            cost_pose = 1.0 / 2.0 * err.T @ weights @ err
            cost = cost_pose[0, 0]

            # -------------- gradients --------------
            if grad.size > 0:
                self.robot_model.compute_jacobians(qpos_dof)
                jaco = self.robot_adaptor.backward_jacobian(self.robot_model.get_frame_space_jacobian(target_link_name))
                jaco[3:6, :] = np.matmul(jacoLeftBCHInverse(link_ori_err), jaco[3:6, :])
                grad[:] = (err.T @ weights @ jaco).reshape(-1)

            return cost

        opt_dim = self.robot_adaptor.doa
        opt = nlopt.opt(nlopt.LD_SLSQP, opt_dim)
        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        epsilon = 1e-3
        opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())
        opt.set_ftol_abs(1e-8)
        opt.set_min_objective(objective)

        qpos_doa_res = opt.optimize(qpos_init)
        return qpos_doa_res

    def single_link_ik_nlopt_arm_only(
        self,
        target_link_name: str,
        ref_link_pose: np.ndarray,
        weights: Optional[np.ndarray] = np.diag([10, 10, 10, 1, 1, 1]),
        qpos_init_arm: Optional[np.ndarray] = None,
        arm_dof: Optional[int] = None,
    ):
        """
        Optimize only the arm joint angles without changing the hand (finger) joint angles.

        Args:
            target_link_name: Name of the target link (e.g., wrist).
            ref_link_pose: Reference link pose (4x4 homogeneous transformation matrix).
            weights: Weight matrix for position and orientation error.
            qpos_init_arm: Initial arm joint angles. If None, the current arm joint angles are used.
            arm_dof: Number of leading actuated joints to optimize. If None, use the controller config.
        """
        # Decompose the target pose into position and orientation
        ref_link_pos, ref_link_ori = isometry3dToPosOri(ref_link_pose)

        # Get the current full joint configuration
        qpos_full_init = self.get_joint_pos(update=False)
        dof = len(qpos_full_init)
        arm_dof = self.arm_dof if arm_dof is None else arm_dof
        if not 0 < arm_dof <= dof:
            raise ValueError(f"arm_dof must be in [1, {dof}], got {arm_dof}.")
        arm_indices = list(range(arm_dof))
        hand_indices = [i for i in range(dof) if i not in arm_indices]

        # print(self.robot_model.dof_joint_names)

        # Initial arm joint angles
        if qpos_init_arm is None:
            qpos_init_arm = qpos_full_init[arm_indices]
        # Fixed hand joint angles (not optimized)
        qpos_hand_fixed = self.init_joint_pos[arm_dof:]

        # print(f"qpos_init_arm: {qpos_init_arm}")
        # print(f"qpos_hand_fixed: {qpos_hand_fixed}")

        action_weight = 0.1

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            # x represents the current optimized arm joint angles
            qpos_arm = x
            # Combine arm joint angles and fixed hand joint angles into a full joint configuration
            qpos_dof = np.concatenate((qpos_arm, qpos_hand_fixed))

            # ------------------- costs -------------------
            self.robot_model.compute_forward_kinematics(qpos_dof)
            link_pose = self.robot_model.get_frame_pose(target_link_name)
            link_pos, link_ori = isometry3dToPosOri(link_pose)
            link_pos_err = link_pos - ref_link_pos
            link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
            err = np.concatenate([link_pos_err, link_ori_err], axis=0).reshape(-1, 1)

            cost_pose = 0.5 * err.T @ weights @ err
            cost_pose = cost_pose[0, 0]

            cost_action = 0.5 * action_weight * np.sum((qpos_arm - qpos_init_arm) ** 2)
            total_cost = cost_pose + cost_action

            # ------------------- gradients -------------------
            if grad.size > 0:
                self.robot_model.compute_jacobians(qpos_dof)
                full_jaco = self.robot_model.get_frame_space_jacobian(target_link_name)
                full_jaco = self.robot_adaptor.backward_jacobian(full_jaco)
                full_jaco[3:6, :] = np.matmul(jacoLeftBCHInverse(link_ori_err), full_jaco[3:6, :])
                # Keep only the gradients for the arm
                jaco_arm = full_jaco[:, arm_indices]
                grad_pose = (err.T @ weights @ jaco_arm).reshape(-1)
                grad_action = action_weight * (qpos_arm - qpos_init_arm)
                total_grad = grad_pose + grad_action

                grad[:] = total_grad

            return total_cost

        opt_dim = arm_dof
        opt = nlopt.opt(nlopt.LD_SLSQP, opt_dim)
        # Extract arm joint limits from the full joint limits
        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        arm_joint_limits = joint_limits[arm_indices, :]
        epsilon = 1e-3
        opt.set_lower_bounds((arm_joint_limits[:, 0] - epsilon).tolist())
        opt.set_upper_bounds((arm_joint_limits[:, 1] + epsilon).tolist())
        opt.set_ftol_abs(1e-8)
        opt.set_min_objective(objective)

        qpos_arm_res = opt.optimize(qpos_init_arm)
        return qpos_arm_res

    def single_link_ik_scipy(self, target_link_name, ref_link_pose, weights, qpos_init):
        ref_link_pos, ref_link_ori = isometry3dToPosOri(ref_link_pose)

        def objective(x: np.ndarray):
            qpos_doa = x
            qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)

            # -------------- objective --------------
            self.robot_model.compute_forward_kinematics(qpos_dof)
            link_pose = self.robot_model.get_frame_pose(target_link_name)
            link_pos, link_ori = isometry3dToPosOri(link_pose)
            link_pos_err = link_pos - ref_link_pos
            link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
            err = np.concatenate([link_pos_err, link_ori_err], axis=0)
            err = err.reshape(-1, 1)

            cost_pose = 1.0 / 2.0 * err.T @ weights @ err
            cost = cost_pose[0, 0]

            # -------------- gradients --------------
            self.robot_model.compute_jacobians(qpos_dof)
            jaco = self.robot_adaptor.backward_jacobian(self.robot_model.get_frame_space_jacobian(target_link_name))
            jaco[3:6, :] = np.matmul(jacoLeftBCHInverse(link_ori_err), jaco[3:6, :])

            grad = (err.T @ weights @ jaco).reshape(-1)

            return cost, grad

        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        joint_pos_bounds = [(joint_limits[i, 0], joint_limits[i, 1]) for i in range(joint_limits.shape[0])]

        res = minimize(
            fun=objective,
            jac=True,
            x0=qpos_init,
            bounds=joint_pos_bounds,
            method="SLSQP",
            options={"ftol": 1e-8, "disp": False},
        )

        qpos_doa_res = res.x.reshape(-1)
        return qpos_doa_res

    def move_to_joint_pos(self, target_joint_pos, max_joint_speed=[[0.1] * 7 + [0.5] * 16], exe_one_step=False):
        """
        Move to the target joint position in a controllable speed.
        Args:
            exe_one_step:
                If True: only execute the first step.
                If False: rhe function is blocked until reaching the target.
        Return:
            Whether the executed target reach the argument 'target_joint_pos'.
        """
        # curr_joint_pos = self.env.get_joint_pos()
        curr_joint_pos = self.env.get_target_joint_pos()
        delta_joint_pos = target_joint_pos - curr_joint_pos
        delta_joint_pos_max = np.asarray(max_joint_speed) * self.env.timestep

        t_max = np.abs(delta_joint_pos_max / (delta_joint_pos + 1e-8))
        t_interp = np.min([np.min(t_max), 1.0])
        n_step = int(1.0 / t_interp)

        # print(f"move_to_joint_pos(): n_step={n_step}")

        for i in range(1, n_step + 1):
            t = float(i) / float(n_step)
            joint_pos = (1 - t) * curr_joint_pos + t * target_joint_pos
            self.env.ctrl_joint_pos(joint_pos)
            if exe_one_step:
                break
            if i < n_step:  # not call step() at the last step
                self.env.step()

        # check if the executed target reach the argument target
        if np.linalg.norm(self.env.get_target_joint_pos() - target_joint_pos) < 1e-8:
            return True
        else:
            return False

    def cartesian_move(self, tcp_link_name: str, tcp_motion: np.ndarray):
        """
        Args:
            tcp_motion[:3]: translation defined in the world frame.
            tcp_motion[3:]: rotation defined in the tcp body frame.
        """
        assert len(tcp_motion) == 6
        target_link_name = tcp_link_name
        qpos_doa = self.env.get_target_joint_pos()
        qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        last_target_pose = self.robot_model.get_frame_pose(target_link_name, qpos_dof)
        curr_pos, curr_ori = isometry3dToPosOri(last_target_pose)

        target_pos = curr_pos + tcp_motion[:3]
        target_ori = curr_ori * sciR.from_rotvec(tcp_motion[3:])
        target_pose = posOri2Isometry3d(target_pos, target_ori)

        weights = np.diag([100, 100, 100, 10, 10, 10])
        qpos_doa_res = self.single_link_ik_nlopt(target_link_name, target_pose, weights, qpos_init=qpos_doa)
        self.env.ctrl_joint_pos(qpos_doa_res)

    def keys_to_tcp_motion(self, pressed_keys, trans_speed=0.001, rot_speed=0.01):
        tcp_motion = np.zeros((6))
        if "d" in pressed_keys:
            tcp_motion[0] = -trans_speed
        if "a" in pressed_keys:
            tcp_motion[0] = +trans_speed
        if "w" in pressed_keys:
            tcp_motion[1] = -trans_speed
        if "s" in pressed_keys:
            tcp_motion[1] = +trans_speed
        if "q" in pressed_keys:
            tcp_motion[2] = -trans_speed
        if "e" in pressed_keys:
            tcp_motion[2] = +trans_speed
        return tcp_motion

    def step(self):
        self.env.step()

    def ctrl_joint_pos(self, target_joint_pos):
        self.env.ctrl_joint_pos(target_joint_pos)

    def get_joint_pos(self, update=True):
        return self.env.get_joint_pos(update)


if __name__ == "__main__":
    pass
