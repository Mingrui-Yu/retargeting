from typing import List, Optional

import numpy as np
import pinocchio as pin


class RobotPinocchio:
    """
    This class does not take mimic joint into consideration.
    All joint orders used in this class is in the format of pinocchio.
    """

    def __init__(self, robot_file_path: str, robot_file_type: str):
        self.model: pin.Model
        if robot_file_type == "mjcf":
            self.model = pin.buildModelFromMJCF(robot_file_path)
        elif robot_file_type == "urdf":
            self.model = pin.buildModelFromUrdf(robot_file_path)
        else:
            raise NotImplementedError(f"Unsupported robot file type: {robot_file_type}.")
        self.data: pin.Data = self.model.createData()

    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names[1:])  # exclude the first 'universe'

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def frame_names(self) -> List[str]:
        frame_names = []
        for i, frame in enumerate(self.model.frames):
            frame_names.append(frame.name)
        return frame_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_frame_index(self, name: str):
        if name not in self.frame_names:
            raise ValueError(f"{name} is not a frame name. Valid link names: \n{self.frame_names}")
        return self.model.getFrameId(name)

    def get_frames_index(self, names: List[str]):
        return [self.get_frame_index(name) for name in names]

    def check_joint_dim(self, q):
        assert len(q) == self.dof

    def compute_forward_kinematics(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None):
        """
        Update forward kinematics of joints and frames.
        """
        self.check_joint_dim(qpos)
        if qvel is None:
            pin.framesForwardKinematics(self.model, self.data, qpos)
        else:
            self.check_joint_dim(qvel)
            pin.forwardKinematics(self.model, self.data, qpos, qvel)  # This only updates joint data
            pin.updateFramePlacements(self.model, self.data)  # Update frame data

    def compute_jacobians(self, qpos: np.ndarray):
        self.check_joint_dim(qpos)
        pin.computeJointJacobians(self.model, self.data, qpos)  # call FK internally
        pin.updateFramePlacements(self.model, self.data)

    def get_frame_pose(self, frame_name: str, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            frame_name:
            qpos: joint position (DoF)
        """
        if qpos is not None:
            self.compute_forward_kinematics(qpos)
        pose = self.data.oMf[self.get_frame_index(frame_name)]
        return pose.homogeneous

    def get_frame_pose_by_id(self, frame_id: int, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            frame_id: Pinocchio frame id.
            qpos: Optional joint position in model DoF order.

        Returns:
            Current frame pose as a homogeneous matrix.
        """
        if qpos is not None:
            self.compute_forward_kinematics(qpos)
        return self.data.oMf[frame_id].homogeneous

    def get_frame_space_jacobian(self, frame_name: str, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        frame_id = self.get_frame_index(frame_name)
        return self.get_frame_space_jacobian_by_id(frame_id, qpos=qpos)

    def get_frame_space_jacobian_by_id(self, frame_id: int, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            frame_id: Pinocchio frame id.
            qpos: Optional joint position in model DoF order.

        Returns:
            Current frame spatial Jacobian in the LOCAL_WORLD_ALIGNED reference frame.
        """
        reference_frame = pin.LOCAL_WORLD_ALIGNED
        if qpos is not None:
            self.check_joint_dim(qpos)
            jaco = pin.computeFrameJacobian(
                self.model,
                self.data,
                q=qpos,
                frame_id=frame_id,
                reference_frame=reference_frame,
            )
        else:
            jaco = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id=frame_id,
                reference_frame=reference_frame,
            )
        return jaco


if __name__ == "__main__":
    from retargeting.config import load_robot_config

    robot_config = load_robot_config("configs/robots/panda_leap_paxini.yaml")
    robot = RobotPinocchio(
        robot_file_path=robot_config.robot_file_path,
        robot_file_type=robot_config.model.type,
    )

    a = robot.get_frame_space_jacobian(frame_name="finger3_tip_center")
    print(a.shape)
