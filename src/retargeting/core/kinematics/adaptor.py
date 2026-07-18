from typing import List

import numpy as np


class RobotAdaptor:
    def __init__(
        self,
        robot_model,
        actuated_joints_name: List[str],
    ):
        self.robot_model = robot_model
        self.actuated_joints_name = actuated_joints_name

        if len(self.actuated_joints_name) != self.robot_model.dof:
            raise NotImplementedError("Currently, no support for coupled joints.")

        # 'model_idx' refers to the index in the model class 'self.robot_model'
        self.actuated_joints_model_idx = [self.robot_model.get_joint_index(name) for name in self.actuated_joints_name]

    @property
    def doa(self) -> int:
        return len(self.actuated_joints_name)

    def check_doa(self, q):
        assert len(q) == self.doa

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Args:
            qpos: position of the actuated joints
        Return:
            qpos_f: position of all dof joints
        """
        self.check_doa(qpos)
        qpos_dof = np.zeros((self.robot_model.dof))
        qpos_dof[self.actuated_joints_model_idx] = qpos.copy()
        return qpos_dof

    def backward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        qpos_doa to qpos_dof.
        """
        self.robot_model.check_joint_dim(qpos)
        return qpos[self.actuated_joints_model_idx].copy()

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Args:
            jacobian: shape (n_batch, 6, n_dof) computed by self.robot_model
        Return:
            jacobian: shape (n_batch, 6, n_doa)
        """
        jacobian_doa = jacobian[..., self.actuated_joints_model_idx]
        return jacobian_doa


if __name__ == "__main__":
    from retargeting.config.core import load_robot_config
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")

    robot_model = RobotPinocchio(
        robot_file_path=robot_config.robot_file_path,
        robot_file_type=robot_config.model.type,
    )

    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=list(robot_config.actuated_joints),
    )

    doa = robot_adaptor.doa
    dof = robot_model.dof

    qpos_dof = robot_adaptor.forward_qpos(np.zeros((doa)))
    print(qpos_dof.shape)

    jaco_doa = robot_adaptor.backward_jacobian(np.zeros((6, dof)))
    print(jaco_doa.shape)
