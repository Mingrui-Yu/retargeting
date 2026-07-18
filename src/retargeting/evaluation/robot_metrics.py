import numpy as np

from retargeting.config.core import RobotBenchmarkConfig
from retargeting.core.kinematics.adaptor import RobotAdaptor


AXIS_TO_COLUMN = {"x": 0, "y": 1, "z": 2}


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize an array of 3D row vectors.

    Args:
        vectors: Array with shape (n, 3).

    Returns:
        Row-normalized vectors with zero rows left unchanged.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms >= 1e-6)


class RobotBenchmark:
    def __init__(self, robot_adaptor: RobotAdaptor, benchmark_config: RobotBenchmarkConfig):
        """Create a benchmark metric helper from robot-specific config.

        Args:
            robot_adaptor: Adapter that owns the loaded robot model.
            benchmark_config: Configured robot-human benchmark frame mappings.

        Returns:
            None.
        """
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model
        self.benchmark_config = benchmark_config
        self.fingertip_links_names = benchmark_config.fingertip_link_names

    def _fingertip_positions(self) -> np.ndarray:
        """Read configured robot fingertip positions from the robot model.

        Args:
            None.

        Returns:
            Fingertip positions with shape (n, 3), ordered by benchmark config.
        """
        poses = np.stack(
            [self.robot_model.get_frame_pose(name) for name in self.fingertip_links_names],
            axis=0,
        )
        return poses[:, :3, 3]

    def _fingertip_direction_axes(self) -> np.ndarray:
        """Read configured robot fingertip direction axes from the robot model.

        Args:
            None.

        Returns:
            Unit direction vectors with shape (n, 3), ordered by benchmark config.
        """
        poses = np.stack(
            [self.robot_model.get_frame_pose(name) for name in self.fingertip_links_names],
            axis=0,
        )
        axis_columns = [
            AXIS_TO_COLUMN[fingertip.robot_direction_axis] for fingertip in self.benchmark_config.fingertips
        ]
        axes = np.stack([pose[:3, axis] for pose, axis in zip(poses, axis_columns)], axis=0)
        return _normalize_rows(axes)

    def position_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the position error between the retargeted qpos and the target qpos.
        """
        # self.robot_model.compute_forward_kinematics(retarget_qpos)
        if type_id == 1:
            retarget_links_pos = self._fingertip_positions()
            human_links_pos = np.asarray(target_pos)[list(self.benchmark_config.human_tip_indices)]
            # print("retarget_links_pos:", retarget_links_pos)
            # print("human_links_pos:", human_links_pos)
            err = np.linalg.norm(retarget_links_pos - human_links_pos, axis=1)
        elif type_id == 2:
            retarget_links_pose_list = [self.robot_model.get_frame_pose(self.benchmark_config.wrist_link_name)]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)
            retarget_links_pos = retarget_links_pose[0, 0:3, 3]
            human_links_pos = target_pos[0]
            err = np.linalg.norm(retarget_links_pos - human_links_pos)
        return err

    def orientation_error(self, retarget_qpos, target_qpos, type_id):
        """
        Calculate the orientation error between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Compare each configured robot fingertip axis to the corresponding
            # human fingertip segment direction.
            robot_dirs = self._fingertip_direction_axes()

            # Compute fingertip direction vectors from hand keypoints (assumed shape: (21, 3))
            hand_kps = np.array(target_qpos)
            fingertip_indices = self.benchmark_config.human_tip_indices
            next_indices = self.benchmark_config.human_direction_base_indices
            human_dirs = []
            for tip_idx, next_idx in zip(fingertip_indices, next_indices):
                direction = hand_kps[tip_idx] - hand_kps[next_idx]
                norm_dir = np.linalg.norm(direction)
                direction = direction / norm_dir if norm_dir >= 1e-6 else np.array([1, 0, 0])
                human_dirs.append(direction)
            human_dirs = np.array(human_dirs)

            # Compute angle error (in radians) between corresponding vectors
            dots = np.sum(robot_dirs * human_dirs, axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            err = np.arccos(dots)
        return err

    def relative_position_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the relative position error from primary fingertips to thumb fingertip
        between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Get retargeted primary fingertip positions from configured non-thumb fingertips.
            primary_fingertips = self.benchmark_config.primary_fingertips
            primary_fingertip_names = [fingertip.link_name for fingertip in primary_fingertips]
            retarget_links_pose_list = [self.robot_model.get_frame_pose(name) for name in primary_fingertip_names]
            retarget_links_pose = np.stack(retarget_links_pose_list, axis=0)  # shape: (n, 4, 4)
            retarget_links_pos = retarget_links_pose[:, :3, 3]  # shape: (n, 3)

            # Get retargeted thumb tip position
            thumb = self.benchmark_config.thumb_fingertip
            thumb_link_pose = self.robot_model.get_frame_pose(thumb.link_name)
            thumb_link_pos = thumb_link_pose[:3, 3]

            # Compute relative vectors for the retargeted hand (from thumb to each primary fingertip)
            retarget_relative_vectors = retarget_links_pos - thumb_link_pos

            # Use the configured human keypoint indices to form the same thumb-relative vectors.
            target_pos = np.array(target_pos)
            human_thumb_pos = target_pos[thumb.human_tip_index]
            primary_indices = [fingertip.human_tip_index for fingertip in primary_fingertips]
            human_fingertip_pos = target_pos[primary_indices]
            human_relative_vectors = human_fingertip_pos - human_thumb_pos

            # Calculate the Euclidean distance error for each primary finger
            error = np.linalg.norm(retarget_relative_vectors - human_relative_vectors, axis=1)
        return error

    def relative_position_to_wrist_error(self, retarget_qpos, target_pos, type_id):
        """
        Calculate the relative position error from all fingertips to wrist
        between the retargeted qpos and the target qpos.
        """
        if type_id == 1:
            # Get retargeted fingertip positions
            retarget_links_pos = self._fingertip_positions()
            wrist_link_pose = self.robot_model.get_frame_pose(self.benchmark_config.wrist_link_name)
            wrist_link_pos = wrist_link_pose[:3, 3]
            retarget_vector = retarget_links_pos - wrist_link_pos
            human_links_pos = np.asarray(target_pos)[list(self.benchmark_config.human_tip_indices)]
            human_wrist_pos = target_pos[0]
            human_vector = human_links_pos - human_wrist_pos
            error = np.linalg.norm(retarget_vector - human_vector, axis=1)
        return error
