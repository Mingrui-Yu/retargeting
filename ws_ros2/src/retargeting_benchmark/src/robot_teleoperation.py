import os
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
from retarget_optimizer import (
    DexPilotOptimizer,
    PositionOptimizer,
    VectorOptimizer,
    VectorWristJointOptimizer,
    VectorWristOptimizer,
)
from robot_adaptor import RobotAdaptor
from robot_benchmark import RobotBenchmark
from robot_control import RobotControl
from robot_mujoco import RobotMujoco
from robot_pinocchio import RobotPinocchio

try:
    from single_hand_detector import SingleHandDetector
except ModuleNotFoundError as e:
    print(e)

from sklearn.preprocessing import normalize
from utils.utils_calc import posRotMat2Isometry3d, quatXYZW2WXYZ, sciR, transformPositions
from utils.utils_mano import MANO_FINGERTIP_INDEX
from utils.utils_mjcf import find_actuated_joints_name, find_touch_joints_name
from vision_pro_detector import VisionProDetector


def sigmoid(x, c=0, w=1):
    return 1 / (1 + np.exp(w * (x - c)))


class RobotTeleoperation:
    def __init__(
        self,
        hand_type,
        robot_adaptor: RobotAdaptor,
        robot_control: RobotControl,
        qpos_init: np.ndarray,
        input_device="rgb",
        mujoco_vis=False,
        use_real_hardware=False,
    ):
        # -------- hyper-parameters --------
        mjcf_file_name = "assets/panda_leap_tac3d_touch_asset.xml"

        """
        retarget_type:
            VECTOR_WRIST_JOINT:
                setting_id = 0: fingertip-wrist vector
                setting_id = 1: fingertip-thumb vector
                setting_id = 2: fingertip-index vector & fingertip_center-lower vector
                setting_id = 3: similar to DexPilot
        """
        self.retarget_type = "VECTOR_WRIST_JOINT"  # "POSITION", "VECTOR", "DEXMV", "DEXPILOT", "VECTOR_WRIST_JOINT", "VECTOR_WRIST_JOINT_2"
        self.setting_id = 3

        self.ablation_option = 0
        # 0: full
        # 1: without pinch
        # 2: actual pinch distance
        # 3: without orientation
        # 4: DexMV orientation
        # 5: replace thumb position with wrist position
        # 6: Replace the thumb position term with a wrist position term, remove the fingertip orientation term and the pinch term
        # 7: remove joint position term
        # 8: 6 + 7

        self.hand_type = hand_type
        self.use_real_hardware = use_real_hardware

        if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
            self.retarget_wrist_method = "separate"
        else:
            self.retarget_wrist_method = "joint"  # "separate" or "joint"

        if self.hand_type == "leap":
            self.human_hand_scale = 1.5
        elif self.hand_type == "shadow":
            self.human_hand_scale = 1.0
        self.ema_alpha = 0.3  # exponential moving average: 1.0 means no smoothing
        self.use_relative_pos = True
        # ----------------------------------

        self.input_device = input_device
        self.robot_model = robot_adaptor.robot_model
        self.robot_adaptor = robot_adaptor
        self.robot_control = robot_control

        self.robot_benchmark = RobotBenchmark(hand_type=self.hand_type, robot_adaptor=self.robot_adaptor)

        self.robot_mujoco = RobotMujoco(mjcf_file_name) if mujoco_vis else None  # only for visualization

        if self.hand_type == "leap":
            if self.ablation_option == 4:
                target_link_pairs = [
                    ["world", "thumb_tip_center"],
                    #
                    ["wrist", "thumb_tip_center"],
                    ["wrist", "finger1_tip_center"],
                    ["wrist", "finger2_tip_center"],
                    ["wrist", "finger3_tip_center"],
                    #
                    ["thumb_tip_center", "finger1_tip_center"],
                    ["thumb_tip_center", "finger2_tip_center"],
                    ["thumb_tip_center", "finger3_tip_center"],
                    #
                    ["wrist", "thumb_tip_center_lower"],
                    ["wrist", "finger1_tip_center_lower"],
                    ["wrist", "finger2_tip_center_lower"],
                    ["wrist", "finger3_tip_center_lower"],
                ]
            else:
                target_link_pairs = [
                    ["world", "thumb_tip_center"],
                    #
                    ["wrist", "thumb_tip_center"],
                    ["wrist", "finger1_tip_center"],
                    ["wrist", "finger2_tip_center"],
                    ["wrist", "finger3_tip_center"],
                    #
                    ["thumb_tip_center", "finger1_tip_center"],
                    ["thumb_tip_center", "finger2_tip_center"],
                    ["thumb_tip_center", "finger3_tip_center"],
                    #
                    ["thumb_tip_center_lower", "thumb_tip_center"],
                    ["finger1_tip_center_lower", "finger1_tip_center"],
                    ["finger2_tip_center_lower", "finger2_tip_center"],
                    ["finger3_tip_center_lower", "finger3_tip_center"],
                ]

            targets = {
                "origin_links_name": [pair[0] for pair in target_link_pairs],
                "task_links_name": [pair[1] for pair in target_link_pairs],
                "wrist_link_name": "wrist",
            }
            params = {"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05}

            self.optimizer = VectorWristJointOptimizer(
                robot_adaptor=self.robot_adaptor,
                targets=targets,
                params=params,
            )

            # for arm end effector retargeting
            self.arm_optimizer = None
            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                target_link_pairs = [
                    ["world", "wrist"],
                ]
                targets = {
                    "origin_links_name": [pair[0] for pair in target_link_pairs],
                    "task_links_name": [pair[1] for pair in target_link_pairs],
                    "wrist_link_name": "wrist",
                }
                self.arm_optimizer = VectorWristJointOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets=targets,
                    params=params,
                )

        elif self.hand_type == "shadow":
            if self.ablation_option == 4:
                target_link_pairs = [
                    ["world", "thtip"],
                    #
                    ["ee_link", "thtip"],
                    ["ee_link", "fftip"],
                    ["ee_link", "mftip"],
                    ["ee_link", "rftip"],
                    ["ee_link", "lftip"],
                    #
                    ["thtip", "fftip"],
                    ["thtip", "mftip"],
                    ["thtip", "rftip"],
                    ["thtip", "lftip"],
                    #
                    ["ee_link", "thdistal"],
                    ["ee_link", "ffdistal"],
                    ["ee_link", "mfdistal"],
                    ["ee_link", "rfdistal"],
                    ["ee_link", "lfdistal"],
                ]
            else:
                target_link_pairs = [
                    ["world", "thtip"],
                    #
                    ["ee_link", "thtip"],
                    ["ee_link", "fftip"],
                    ["ee_link", "mftip"],
                    ["ee_link", "rftip"],
                    ["ee_link", "lftip"],
                    #
                    ["thtip", "fftip"],
                    ["thtip", "mftip"],
                    ["thtip", "rftip"],
                    ["thtip", "lftip"],
                    #
                    ["thdistal", "thtip"],
                    ["ffdistal", "fftip"],
                    ["mfdistal", "mftip"],
                    ["rfdistal", "rftip"],
                    ["lfdistal", "lftip"],
                ]

            targets = {
                "origin_links_name": [pair[0] for pair in target_link_pairs],
                "task_links_name": [pair[1] for pair in target_link_pairs],
                "wrist_link_name": "ee_link",
            }
            params = {"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05}

            self.optimizer = VectorWristJointOptimizer(
                robot_adaptor=self.robot_adaptor,
                targets=targets,
                params=params,
            )

            # for arm end effector retargeting
            self.arm_optimizer = None
            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                target_link_pairs = [
                    ["world", "ee_link"],
                ]
                targets = {
                    "origin_links_name": [pair[0] for pair in target_link_pairs],
                    "task_links_name": [pair[1] for pair in target_link_pairs],
                    "wrist_link_name": "ee_link",
                }
                self.arm_optimizer = VectorWristJointOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets=targets,
                    params=params,
                )

        if self.input_device == "rgb":
            self.detector = SingleHandDetector("Right")
        elif self.input_device == "vision_pro":
            self.detector = VisionProDetector()
        else:
            return NotImplementedError()

        # variables
        self.qpos_init = qpos_init
        self.qpos_last = qpos_init
        self.qpos_arm_last = qpos_init[:9]  # include the 2 dof of shadow hand wrist
        # print("self.qpos_arm_last: ", self.qpos_arm_last)

        self.robot_init_wrist_pose: Optional[np.ndarray] = None
        self.avp_init_wrist_pose: Optional[np.ndarray] = None

    def rgb_retarget(self, color_img: np.ndarray, camera_K: np.ndarray):
        _, hand_kps_in_wrist, keypoint_2d, wrist_pose_in_cam = self.detector.detect(color_img, camera_K)

        if hand_kps_in_wrist is None:
            print(f"{self.detector.hand_type} hand is not detected.")
            cv2.imshow("detection result", color_img)
            qpos = None
        else:
            hand_kps_in_wrist *= self.human_hand_scale
            wrist_pose_in_cam[0, 3] += 0.4
            qpos = self.hand_retarget(hand_kps_in_wrist, wrist_pose_in_cam)

            # -------------- visualize hand detection results by cv2 --------------
            annotated_image = self.detector.draw_skeleton_on_image(color_img, keypoint_2d, style="default")
            cv2.imshow("detection result", annotated_image)
        cv2.waitKey(1)

        return hand_kps_in_wrist, wrist_pose_in_cam, qpos

    def pose_from_avp_world_to_robot_world(self, pose_in_avp_world):
        transform = posRotMat2Isometry3d(
            pos=[0, 0, 0], rot_mat=sciR.from_euler("xyz", [0, 0, 180], degrees=True).as_matrix()
        )
        pose_in_world = transform @ pose_in_avp_world  # rotate along z-axis for 180 degree
        pose_in_world[:3, 3] += [0.7, 0.2, -1.0]  # further translation
        return pose_in_world

    def set_robot_init_wrist_pose(self, pose):
        self.robot_init_wrist_pose = pose.copy()

    def set_avp_init_wrist_pose(self, pose):
        self.avp_init_wrist_pose = pose.copy()

    def vision_pro_retarget(self, stream):
        """
        Return:
            target_hand_kps_in_wrist:
            target_wrist_pose_in_world:
            qpos_retarget:
        """
        _, hand_kps_in_wrist, _, wrist_pose_in_avp_world = VisionProDetector.detect(stream)

        if hand_kps_in_wrist is None:
            print("hand is not detected.")
            return None, None, None
        else:
            target_hand_kps_in_wrist = hand_kps_in_wrist * self.human_hand_scale
            detected_wrist_pose_in_world = self.pose_from_avp_world_to_robot_world(wrist_pose_in_avp_world)
            if self.use_relative_pos:
                target_wrist_pose_in_world = detected_wrist_pose_in_world.copy()
                target_wrist_pose_in_world[:3, 3] += self.robot_init_wrist_pose[:3, 3] - self.avp_init_wrist_pose[:3, 3]
            else:
                target_wrist_pose_in_world = detected_wrist_pose_in_world.copy()

            # print("self.robot_init_wrist_pose: ", self.robot_init_wrist_pose[:3, 3])
            # print("target_wrist_pose_in_world: ", target_wrist_pose_in_world[:3, 3])

            qpos_retarget, err = self.hand_retarget(target_hand_kps_in_wrist, target_wrist_pose_in_world)

            return target_hand_kps_in_wrist, target_wrist_pose_in_world, qpos_retarget, err

    def hand_retarget(self, hand_kps_in_wrist, wrist_pose_in_world):
        """
        Args:
            hand_kps_in_wrist:
            wrist_pose_in_world:
        """

        hand_kps_in_world = transformPositions(hand_kps_in_wrist, target_frame_pose_inv=wrist_pose_in_world)
        wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)

        if self.hand_type == "leap":

            ref_link_vec = np.zeros((12, 3))
            weights_links_vec = np.zeros((12))
            wrist_pos = hand_kps_in_world[0, :]
            thumb_tip = hand_kps_in_world[MANO_FINGERTIP_INDEX[0]]
            thumb_primary_dist = np.linalg.norm(
                hand_kps_in_world[MANO_FINGERTIP_INDEX[1:4]] - thumb_tip.reshape(1, 3), axis=1
            )
            pinch_thres_1 = 0.1  # pinch wrist transition thres
            pinch_thres_2 = 0.01  # pinch in-contact thres; lower than this threshold will be regard as 0 distance
            sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_thres_1, w=10)
            sigmoid_weights_wrist_fingertips = sigmoid(
                np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0),
                c=pinch_thres_1,
                w=-10,
            )
            # world-thumb vector
            ref_link_vec[0, :] = thumb_tip
            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                weights_links_vec[0] = 0.0
            else:
                weights_links_vec[0] = 10.0
            # wrist-fingertip vector
            ref_link_vec[1:5, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[:4]] - wrist_pos
            if self.ablation_option == 1 or self.ablation_option == 6 or self.ablation_option == 8:    
                weights_links_vec[1:5] = 1.0
            else:
                weights_links_vec[1:5] = 1.0 * sigmoid_weights_wrist_fingertips
            # thumb-primary vector
            if self.ablation_option == 1 or self.ablation_option == 6 or self.ablation_option == 8:    
                weights_links_vec[5:8] = 0.0
            elif self.ablation_option == 2:
                ref_link_vec[5:8, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:4]] - thumb_tip.reshape(1, 3)
                weights_links_vec[5:8] = 10.0 * sigmoid_weights_thumb_primary
            else:
                rel_pos = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:4]] - thumb_tip.reshape(1, 3)
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                # rescale [pinch_thres_2, pinch_thres_1] to [0, pinch_thres_1]
                k = pinch_thres_1 / (pinch_thres_1 - pinch_thres_2)
                rescaled_rel_dist = k * (rel_dist - pinch_thres_2)
                rescaled_rel_dist[rel_dist < pinch_thres_2] = 0
                rescaled_rel_dist[rel_dist > pinch_thres_1] = rel_dist[rel_dist > pinch_thres_1]
                rescaled_rel_pos = normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
                ref_link_vec[5:8, :] = rescaled_rel_pos
                weights_links_vec[5:8] = 10.0 * sigmoid_weights_thumb_primary
            # fingertip orientation vector
            mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:4])
            if self.ablation_option == 3:  #  not considering fingertip orientation
                weights_links_vec[8:12] = 0.0
            elif self.ablation_option == 4:
                ref_link_vec[8:12, :] = hand_kps_in_world[mano_fingertip_index - 1] - wrist_pos.reshape(1, 3)
                weights_links_vec[8:12] = 10.0
            elif self.ablation_option == 6 or self.ablation_option == 8:
                weights_links_vec[8:12] = 0.0
            else:
                ref_link_vec[8:12, :] = (
                    hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                )
                weights_links_vec[8:12] = 10.0

            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                weights_wrist_rot = 0
            else:
                weights_wrist_rot = 0.1

            if self.ablation_option == 7 or self.ablation_option == 8:
                weights_joint_pos = np.zeros(23)
            else:  # [0, 0, 1.0, 0, 0.5, 0, 0]
                weights_joint_pos = [0, 0, 1.0, 0, 0.5, 0, 0] + [
                    0.5, # joint 0
                    0,
                    0,
                    0,
                    0.5, # joint 4
                    0,
                    0,
                    0,
                    0.5, # joint 8
                    0,
                    0,
                    0.5, # joint 11
                    0,
                    0.1, # joint 13
                    0,
                    0,
                ]

            # -------------------------------------

            ref_values = {
                "links_vec": ref_link_vec,
                "wrist_quat": wrist_quat,
                "qpos_doa": self.qpos_init.copy(),
                "qpos_doa_last": self.qpos_last.copy(),
                "weights": {
                    "links_vec": weights_links_vec,
                    "wrist_rot": weights_wrist_rot,
                    "joint_pos": weights_joint_pos,
                    "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-2] * 16,
                },
            }

        # TODO: implement for shadow hand
        elif self.hand_type == "shadow":
            ref_link_vec = np.zeros((15, 3))
            weights_links_vec = np.zeros((15))
            wrist_pos = hand_kps_in_world[0, :]
            thumb_tip = hand_kps_in_world[MANO_FINGERTIP_INDEX[0]]
            thumb_primary_dist = np.linalg.norm(
                hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3), axis=1
            )
            pinch_thres_1 = 0.1  # pinch wrist transition thres
            pinch_thres_2 = 0.01  # pinch in-contact thres; lower than this threshold will be regard as 0 distance
            sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_thres_1, w=10)
            sigmoid_weights_wrist_fingertips = sigmoid(
                np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0),
                c=pinch_thres_1,
                w=-10,
            )
            # world-thumb vector
            ref_link_vec[0, :] = thumb_tip
            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                weights_links_vec[0] = 0.0
            else:
                weights_links_vec[0] = 10.0
            # wrist-fingertip vector
            ref_link_vec[1:6, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[:5]] - wrist_pos
            if self.ablation_option == 1 or self.ablation_option == 6 or self.ablation_option == 8:    
                weights_links_vec[1:6] = 1.0
            else:
                weights_links_vec[1:6] = 1.0 * sigmoid_weights_wrist_fingertips
            # thumb-primary vector
            if self.ablation_option == 1 or self.ablation_option == 6 or self.ablation_option == 8:    
                weights_links_vec[6:10] = 0.0
            elif self.ablation_option == 2:
                ref_link_vec[6:10, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3)
                weights_links_vec[6:10] = 10.0 * sigmoid_weights_thumb_primary
            else:
                rel_pos = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3)
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                # rescale [pinch_thres_2, pinch_thres_1] to [0, pinch_thres_1]
                k = pinch_thres_1 / (pinch_thres_1 - pinch_thres_2)
                rescaled_rel_dist = k * (rel_dist - pinch_thres_2)
                rescaled_rel_dist[rel_dist < pinch_thres_2] = 0
                rescaled_rel_dist[rel_dist > pinch_thres_1] = rel_dist[rel_dist > pinch_thres_1]
                rescaled_rel_pos = normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
                ref_link_vec[6:10, :] = rescaled_rel_pos
                weights_links_vec[6:10] = 10.0 * sigmoid_weights_thumb_primary
            # fingertip orientation vector
            mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:5])
            if self.ablation_option == 3:  #  not considering fingertip orientation
                weights_links_vec[10:15] = 0.0
            elif self.ablation_option == 4:
                ref_link_vec[10:15, :] = hand_kps_in_world[mano_fingertip_index - 1] - wrist_pos.reshape(1, 3)
                weights_links_vec[10:15] = 10.0
            elif self.ablation_option == 6 or self.ablation_option == 8:
                weights_links_vec[10:15] = 0.0
            else:
                ref_link_vec[10:15, :] = (
                    hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                )
                weights_links_vec[10:15] = 10.0

            if self.ablation_option == 5 or self.ablation_option == 6 or self.ablation_option == 8:
                weights_wrist_rot = 0
            else:
                weights_wrist_rot = 0.1

            if self.ablation_option == 7 or self.ablation_option == 8:
                weights_joint_pos = np.zeros(31)
            else:  # [0, 0, 1.0, 0, 0.5, 0, 0]
                weights_joint_pos = [0, 0, 1.0, 0, 0.5, 0, 0] + [
                    0,
                    0,
                    0.5, # FF4
                    0,
                    0,
                    0,
                    0.5, # MF4
                    0,
                    0,
                    0,
                    0.5, # RF4
                    0,
                    0,
                    0,
                    0, # LF5
                    0.5, # LF4
                    0,
                    0,
                    0,
                    0.1, # TH5
                    0,
                    0,
                    0,
                    0,
                ]

            # -------------------------------------

            ref_values = {
                "links_vec": ref_link_vec,
                "wrist_quat": wrist_quat,
                "qpos_doa": self.qpos_init.copy(),
                "qpos_doa_last": self.qpos_last.copy(),
                "weights": {
                    "links_vec": weights_links_vec,
                    "wrist_rot": weights_wrist_rot,
                    "joint_pos": weights_joint_pos,
                    "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-2] * 24,
                },
            }

        # ---------------------joint/separate arm-hand retargeting----------------------

        t1 = time.perf_counter()

        if self.retarget_wrist_method == "joint":
            print("Joint retargeting.")
            qpos = self.optimizer.retarget(ref_values)

        elif self.retarget_wrist_method == "separate":
            print("Separate retargeting.")

            # solve retargeting based on wrist pose to get qpos of robot arm (using the VectorWristJoint class)
            # to incorporate same cost terms for arm as the joint-retargeting
            ref_link_vec_arm = np.zeros((1, 3))
            ref_link_vec_arm[0, :] = hand_kps_in_world[0, :]  # wrist pos in world
            weights_links_vec_arm = np.zeros((1))
            weights_links_vec_arm[0] = 10
            weights_wrist_rot_arm = 1.0

            if self.hand_type == "leap":
                ref_link_vec
                arm_ref_values = {
                    "links_vec": ref_link_vec_arm,
                    "wrist_quat": wrist_quat,
                    "qpos_doa": self.qpos_init.copy(),
                    "qpos_doa_last": self.qpos_last.copy(),
                    "weights": {
                        "links_vec": weights_links_vec_arm,
                        "wrist_rot": weights_wrist_rot_arm,
                        "joint_pos": weights_joint_pos,
                        "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-2] * 16,
                    },
                }

                # arm ee pose retargeting
                qpos = self.arm_optimizer.retarget(arm_ref_values)
                qpos_arm = qpos[:7]
                # finger retargeting
                ref_values["qpos_doa"][:7] = qpos_arm
                ref_values["qpos_doa_last"][:7] = qpos_arm
                qpos = self.optimizer.retarget(ref_values, qpos_arm)
                qpos = np.concatenate([qpos_arm[:7], qpos[7:]])
            
            elif self.hand_type == "shadow":
                ref_link_vec
                arm_ref_values = {
                    "links_vec": ref_link_vec_arm,
                    "wrist_quat": wrist_quat,
                    "qpos_doa": self.qpos_init.copy(),
                    "qpos_doa_last": self.qpos_last.copy(),
                    "weights": {
                        "links_vec": weights_links_vec_arm,
                        "wrist_rot": weights_wrist_rot_arm,
                        "joint_pos": weights_joint_pos,
                        "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-2] * 24,
                    },
                }

                # arm ee pose retargeting
                qpos = self.arm_optimizer.retarget(arm_ref_values)
                qpos_arm = qpos[:9]
                # finger retargeting
                ref_values["qpos_doa"][:9] = qpos_arm
                ref_values["qpos_doa_last"][:9] = qpos_arm
                qpos = self.optimizer.retarget(ref_values, qpos_arm)
                qpos = np.concatenate([qpos_arm[:9], qpos[9:]])

        # -----------------------------Benchmark---------------------------------

        print(f"retarget opt time cost: {(time.perf_counter() - t1):.3f}")
        optimizaion_time = time.perf_counter() - t1

        position_err = self.robot_benchmark.position_error(qpos, hand_kps_in_world, 1)
        orientation_err = self.robot_benchmark.orientation_error(qpos, hand_kps_in_world, 1)
        relative_position_err = self.robot_benchmark.relative_position_error(qpos, hand_kps_in_world, 1)
        relative_position_to_wrist_err = self.robot_benchmark.relative_position_to_wrist_error(qpos, hand_kps_in_world, 1)

        err = {
            "position_err": position_err,
            "orientation_err": orientation_err,
            "relative_position_err": relative_position_err,
            "relative_position_to_wrist_err": relative_position_to_wrist_err,
            "optimizaion_time": optimizaion_time,
        }

        if self.use_real_hardware:
            qpos = self.ema_alpha * qpos + (1 - self.ema_alpha) * self.qpos_last
        self.qpos_last = qpos

        if self.robot_mujoco:
            self.robot_mujoco.set_joint_pos(qpos)
            self.robot_mujoco.sim_step(refresh=True)

        return qpos, err


def main():
    urdf_file_name = os.readlink("assets/panda_leap_tac3d.urdf")  # no touch bodies/joints/sensors
    actuated_joints_name = [f"panda_joint{i+1}" for i in range(7)] + [f"joint_{i}" for i in range(16)]
    touch_joints_name: List[str] = []

    robot_model = RobotPinocchio(
        robot_file_path=urdf_file_name,
        robot_file_type="urdf",
    )
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=actuated_joints_name,
        touch_joints_name=touch_joints_name,
    )

    # teleop = RobotTeleoperation(input_device="rgb", mujoco_vis=True)
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
    #     teleop.rgb_retarget(color_img.copy(), camera_K)

    # ------------------------------------------------------------

    teleop = RobotTeleoperation(robot_adaptor=robot_adaptor, input_device="vision_pro", mujoco_vis=True)
    data = np.load("data/test_teleop/vision_pro/data.npy", allow_pickle=True).item()
    stream_data = data["stream"]

    for i in range(len(stream_data) - 1):
        print("frame: ", i)
        teleop.vision_pro_retarget(stream=stream_data[i])


if __name__ == "__main__":
    main()
