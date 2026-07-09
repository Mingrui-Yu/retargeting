import time
from pathlib import Path
from typing import Optional

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
from retargeting.config import default_robot_config_path, load_robot_config

try:
    from single_hand_detector import SingleHandDetector
except ModuleNotFoundError as e:
    print(e)

from sklearn.preprocessing import normalize
from utils.utils_calc import posRotMat2Isometry3d, quatXYZW2WXYZ, sciR, transformPositions
from utils.utils_mano import MANO_FINGERTIP_INDEX
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

        self.hand_type = hand_type
        self.use_real_hardware = use_real_hardware

        if self.ablation_option == 1:
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

        robot_config = load_robot_config(default_robot_config_path(self.hand_type))
        self.robot_benchmark = RobotBenchmark(
            robot_adaptor=self.robot_adaptor,
            benchmark_config=robot_config.benchmark,
        )

        if mujoco_vis:
            raise ValueError("MuJoCo visualization requires a configured MJCF asset.")
        self.robot_mujoco = None

        if self.hand_type == "leap":
            if self.ablation_option == 3:
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

        elif self.hand_type == "shadow":
            if self.retarget_type == "POSITION":
                self.optimizer = PositionOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets={
                        "target_links_name": [
                            "ee_link",
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                            # "thmiddle",
                            # "ffmiddle",
                            # "mfmiddle",
                            # "rfmiddle",
                            # "lfmiddle",
                        ],
                        "wrist_name": "ee_link",
                    },
                    params={"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.03},
                )
            elif self.retarget_type == "VECTOR":
                self.optimizer = VectorOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets={
                        "origin_links_name": [
                            "world",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                        ],
                        "task_links_name": [
                            "ee_link",
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                        ],
                    },
                    params={"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05},
                )
            elif self.retarget_type == "DEXMV":
                self.optimizer = VectorOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets={
                        "origin_links_name": [
                            "world",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "thdistal",
                            "ffdistal",
                            "mfdistal",
                            "rfdistal",
                            "lfdistal",
                        ],
                        "task_links_name": [
                            "ee_link",
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                        ],
                    },
                    params={"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05},
                )
            elif self.retarget_type == "DEXPILOT":
                self.optimizer = DexPilotOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets={
                        "fingertip_links_name": [
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                        ],
                        "wrist_link_name": "ee_link",
                    },
                    params={
                        "huber_delta": 0.02,
                        "project_dist": 0.03,
                        "escape_dist": 0.05,
                        "eta1": 1e-4,
                        "eta2": 3e-2,
                        "opt_ftol_abs": 1e-5,
                        "opt_maxtime": 0.05,
                    },
                )
            elif self.retarget_type == "VECTOR_WRIST":
                self.optimizer = VectorWristOptimizer(
                    robot_adaptor=self.robot_adaptor,
                    targets={
                        "origin_links_name": [
                            "world",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                            "ee_link",
                        ],
                        "task_links_name": [
                            "ee_link",
                            "thtip",
                            "fftip",
                            "mftip",
                            "rftip",
                            "lftip",
                        ],
                        "wrist_link_name": "ee_link",
                    },
                    params={"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05},
                )
            elif self.retarget_type == "VECTOR_WRIST_JOINT":
                if self.setting_id == 3:
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
            elif self.retarget_type == "VECTOR_WRIST_JOINT_2":
                if self.setting_id == 3:
                    target_link_pairs = [
                        ["world", "ee_link"],
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
            sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_thres_1, w=100)
            sigmoid_weights_wrist_fingertips = sigmoid(
                np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0),
                c=pinch_thres_1,
                w=-100,
            )
            # world-thumb vector
            ref_link_vec[0, :] = thumb_tip
            if self.ablation_option == 1:
                weights_links_vec[0] = 0.0
            else:
                weights_links_vec[0] = 10.0
            # wrist-fingertip vector
            ref_link_vec[1:5, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[:4]] - wrist_pos
            weights_links_vec[1:5] = 1.0 * sigmoid_weights_wrist_fingertips  # TODO: no sigmoid for ablation 4
            # thumb-primary vector
            if self.ablation_option == 4:
                weights_links_vec[5:8] = 0.0
            elif self.ablation_option == 5:
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
            if self.ablation_option == 2:  #  not considering fingertip orientation
                weights_links_vec[8:12] = 0.0
            else:
                mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:4])
                ref_link_vec[8:12, :] = (
                    hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                )  # TODO: reference is different for ablation 3
                weights_links_vec[8:12] = 10.0

            if self.ablation_option == 1:
                weights_wrist_rot = 0
            else:
                weights_wrist_rot = 0.1

            if self.ablation_option == 6:
                weights_joint_pos = np.zeros(23)
            else:
                weights_joint_pos = [0, 0, 0.1, 0, 0.1, 0, 0] + [
                    0.5,
                    0,
                    0,
                    0,
                    0.5,
                    0,
                    0,
                    0,
                    0.5,
                    0,
                    0,
                    0.5,
                    0,
                    0.1,
                    0,
                    0,
                ]

            # -------------------------------------

            wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)

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

        elif self.hand_type == "shadow":

            if self.retarget_type == "POSITION":
                ref_link_pos = hand_kps_in_world[[0] + MANO_FINGERTIP_INDEX[0:5], :]
                ref_values = {
                    "links_pos": ref_link_pos,
                    "qpos_doa_last": self.qpos_last,
                    "weights": {
                        "links_pos": [1, 1, 1, 1, 1, 1],
                        "action": [1e-1] * 7 + [1e-3] * 24,
                    },
                }
            # if self.retarget_type == "POSITION":
            #     ref_link_pos = hand_kps_in_world[[0] + MANO_FINGERTIP_INDEX[0:5] + [2, 6, 10, 14, 18], :]
            #     ref_values = {
            #         "links_pos": ref_link_pos,
            #         "qpos_doa_last": self.qpos_last,
            #         "weights": {
            #             "links_pos": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            #             "action": [1e-1] * 7 + [1e-3] * 24,
            #         },
            #     }
            elif self.retarget_type == "VECTOR":
                ref_link_vec = np.zeros((6, 3))
                ref_link_vec[0, :] = hand_kps_in_world[0, :]
                ref_link_vec[1:, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[0:5], :] - hand_kps_in_world[0, :].reshape(
                    1, 3
                )
                ref_values = {
                    "links_vec": ref_link_vec,
                    "qpos_doa_last": self.qpos_last,
                    "weights": {
                        "links_vec": [10, 1, 1, 1, 1, 1],
                        "action": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2] + [1e-4] * 24,
                    },
                }
            elif self.retarget_type == "DEXMV":
                ref_link_vec = np.zeros((11, 3))
                ref_link_vec[0, :] = hand_kps_in_world[0, :]
                ref_link_vec[1:6, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[0:5], :] - hand_kps_in_world[
                    0, :
                ].reshape(1, 3)
                mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:5])
                ref_link_vec[6:, :] = (
                    hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                )
                ref_values = {
                    "links_vec": ref_link_vec,
                    "qpos_doa_last": self.qpos_last,
                    "weights": {
                        "links_vec": [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        "action": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2] + [1e-4] * 24,
                    },
                }
            elif self.retarget_type == "DEXPILOT":
                """
                ([2, 3, 4, 5, 3, 4, 5, 4, 5, 5, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 3, 4, 5])'
                """
                ref_link_vec = np.zeros((15, 3))
                fingertip_vec = np.zeros((5, 3))
                wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())
                for i in range(5):
                    fingertip_vec[i, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[i], :]
                ref_link_vec[0, :] = fingertip_vec[0, :] - fingertip_vec[1, :]
                ref_link_vec[1, :] = fingertip_vec[0, :] - fingertip_vec[2, :]
                ref_link_vec[2, :] = fingertip_vec[0, :] - fingertip_vec[3, :]
                ref_link_vec[3, :] = fingertip_vec[0, :] - fingertip_vec[4, :]
                ref_link_vec[4, :] = fingertip_vec[1, :] - fingertip_vec[2, :]
                ref_link_vec[5, :] = fingertip_vec[1, :] - fingertip_vec[3, :]
                ref_link_vec[6, :] = fingertip_vec[1, :] - fingertip_vec[4, :]
                ref_link_vec[7, :] = fingertip_vec[2, :] - fingertip_vec[3, :]
                ref_link_vec[8, :] = fingertip_vec[2, :] - fingertip_vec[4, :]
                ref_link_vec[9, :] = fingertip_vec[3, :] - fingertip_vec[4, :]
                ref_link_vec[10, :] = fingertip_vec[0, :] - hand_kps_in_world[0, :]
                ref_link_vec[11, :] = fingertip_vec[1, :] - hand_kps_in_world[0, :]
                ref_link_vec[12, :] = fingertip_vec[2, :] - hand_kps_in_world[0, :]
                ref_link_vec[13, :] = fingertip_vec[3, :] - hand_kps_in_world[0, :]
                ref_link_vec[14, :] = fingertip_vec[4, :] - hand_kps_in_world[0, :]
                ref_values = {
                    "target_vector": ref_link_vec,
                    "qpos_doa_last": self.qpos_last,
                    "wrist_link_pos": hand_kps_in_world[0, :],
                    "wrist_quat": wrist_quat,
                    "weights": {
                        "action": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2] + [1e-4] * 24,
                    },
                }
            elif self.retarget_type == "VECTOR_WRIST":
                ref_link_vec = np.zeros((6, 3))
                # vector between wrist and world
                ref_link_vec[0, :] = hand_kps_in_world[0, :]
                ref_link_vec[1:, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[0:5], :] - hand_kps_in_world[0, :].reshape(
                    1, 3
                )
                wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)
                qpos_doa_init = self.qpos_last.copy()
                ref_values = {
                    "links_vec": ref_link_vec,
                    "wrist_quat": wrist_quat,
                    "qpos_doa_last": qpos_doa_init,
                    "weights": {
                        "links_vec": [1, 1, 1, 1, 1, 1],
                        "wrist_rot": 0.01,
                        "action": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-3] * 24,
                    },
                }
            elif self.retarget_type == "VECTOR_WRIST_JOINT":
                if self.setting_id == 3:
                    ref_link_vec = np.zeros((15, 3))
                    weights_links_vec = np.zeros((15))
                    wrist_pos = hand_kps_in_world[0, :]
                    thumb_tip = hand_kps_in_world[MANO_FINGERTIP_INDEX[0]]
                    thumb_primary_dist = np.linalg.norm(
                        hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3), axis=1
                    )
                    sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=0.1, w=100)
                    sigmoid_weights_wrist_fingertips = sigmoid(
                        np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0), c=0.1, w=-100
                    )
                    # world-thumb vector
                    ref_link_vec[0, :] = thumb_tip
                    weights_links_vec[0] = 10.0
                    # wrist-fingertip vector
                    ref_link_vec[1:6, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[:5]] - wrist_pos
                    weights_links_vec[1:6] = 1.0 * sigmoid_weights_wrist_fingertips
                    # thumb-primary vector
                    ref_link_vec[6:10, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3)
                    weights_links_vec[6:10] = 10.0 * sigmoid_weights_thumb_primary
                    # fingertip orientation vector
                    mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:5])
                    ref_link_vec[10:15, :] = (
                        hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                    )
                    weights_links_vec[10:15] = 10.0

                    weights_wrist_rot = 0.1
                    weights_joint_pos = np.zeros((31))

                # -------------------------------------

                wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)

                ref_values = {
                    "links_vec": ref_link_vec,
                    "wrist_quat": wrist_quat,
                    "qpos_doa": self.qpos_init.copy(),
                    "qpos_doa_last": self.qpos_last.copy(),
                    "weights": {
                        "links_vec": weights_links_vec,
                        "wrist_rot": weights_wrist_rot,
                        "joint_pos": weights_joint_pos,
                        "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-3] * 24,
                    },
                }
            elif self.retarget_type == "VECTOR_WRIST_JOINT_2":
                if self.setting_id == 3:
                    ref_link_vec = np.zeros((15, 3))
                    weights_links_vec = np.zeros((15))
                    wrist_pos = hand_kps_in_world[0, :]
                    thumb_tip = hand_kps_in_world[MANO_FINGERTIP_INDEX[0]]
                    thumb_primary_dist = np.linalg.norm(
                        hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3), axis=1
                    )
                    sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=0.1, w=100)
                    sigmoid_weights_wrist_fingertips = sigmoid(
                        np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist], axis=0), c=0.1, w=-100
                    )
                    # world-thumb vector
                    ref_link_vec[0, :] = wrist_pos
                    weights_links_vec[0] = 10.0
                    # wrist-fingertip vector
                    ref_link_vec[1:6, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[:5]] - wrist_pos
                    weights_links_vec[1:6] = 1.0 * sigmoid_weights_wrist_fingertips
                    weights_links_vec[1] = 10.0
                    # thumb-primary vector
                    ref_link_vec[6:10, :] = hand_kps_in_world[MANO_FINGERTIP_INDEX[1:5]] - thumb_tip.reshape(1, 3)
                    weights_links_vec[6:10] = 10.0 * sigmoid_weights_thumb_primary
                    # fingertip orientation vector
                    mano_fingertip_index = np.asarray(MANO_FINGERTIP_INDEX[:5])
                    ref_link_vec[10:15, :] = (
                        hand_kps_in_world[mano_fingertip_index] - hand_kps_in_world[mano_fingertip_index - 1]
                    )
                    weights_links_vec[10:15] = 10.0

                    weights_wrist_rot = 0.1
                    weights_joint_pos = np.zeros((31))

                # -------------------------------------

                wrist_quat = quatXYZW2WXYZ(sciR.from_matrix(wrist_pose_in_world[:3, :3]).as_quat())  # (w, x, y, z)

                ref_values = {
                    "links_vec": ref_link_vec,
                    "wrist_quat": wrist_quat,
                    "qpos_doa": self.qpos_init.copy(),
                    "qpos_doa_last": self.qpos_last.copy(),
                    "weights": {
                        "links_vec": weights_links_vec,
                        "wrist_rot": weights_wrist_rot,
                        "joint_pos": weights_joint_pos,
                        "joint_vel": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1] + [1e-3] * 24,
                    },
                }

        # # -------------------- check gradient --------------------

        # objective = self.optimizer.get_objective_function(ref_values)

        # x = np.zeros(23)
        # grad_analytical = np.zeros(23)
        # cost = objective(x, grad_analytical)

        # grad_numerical = np.zeros(23)
        # grad_temp = np.empty((0))
        # for i in range(len(x)):
        #     x_disturb = x.copy()
        #     step = 1e-3
        #     x_disturb[i] += step
        #     grad_numerical[i] = (objective(x_disturb, grad_temp) - cost) / step

        # print(f"grad_analytical: {grad_analytical}")
        # print(f"grad_numerical: {grad_numerical}")
        # print(f"relative err: {(grad_analytical - grad_numerical)/grad_analytical}")

        # exit()

        # ---------------------joint/separate arm-hand retargeting----------------------

        t1 = time.perf_counter()

        if self.retarget_wrist_method == "joint":
            print("Joint retargeting.")
            qpos = self.optimizer.retarget(ref_values)

        elif self.retarget_wrist_method == "separate":
            print("Separate retargeting.")
            # solve IK based on wrist pose to get qpos of robot arm
            wrist_pose = wrist_pose_in_world
            if self.hand_type == "leap":
                qpos_arm = self.robot_control.single_link_ik_nlopt_arm_only("wrist", "leap", wrist_pose)
                print("qpos_arm: ", qpos_arm)
                qpos = self.optimizer.retarget(ref_values, qpos_arm)
                qpos = np.concatenate([qpos_arm[:7], qpos[7:]])
            elif self.hand_type == "shadow":
                # qpos_arm_last_ik = self.qpos_arm_last
                # print("qpos_arm_last: ", qpos_arm_last_ik)
                # # qpos_arm = self.robot_control.single_link_ik_nlopt_arm_only(
                # #     "ee_link", wrist_pose, qpos_init_arm=qpos_arm_last_ik
                # # )
                # qpos_arm = self.robot_control.single_link_ik_nlopt_arm_only("ee_link", wrist_pose)
                # # qpos_arm = 0.1 * qpos_arm + 0.9 * qpos_arm_last_ik
                # # qpos_arm = self.robot_control.single_link_ik_nlopt_arm_only("ee_link", wrist_pose)
                # print("qpos_arm_last: ", self.qpos_arm_last)
                qpos_arm = self.robot_control.single_link_ik_nlopt_arm_only(
                    "ee_link", "shadow", wrist_pose, qpos_init_arm=self.qpos_arm_last
                )
                qpos = self.optimizer.retarget(ref_values, qpos_arm)
                qpos = np.concatenate([qpos_arm[:9], qpos[9:]])
                # self.qpos_arm_last = qpos_arm
            # print("qpos_arm: ", qpos_arm)
            # qpos = self.optimizer.retarget(ref_values, qpos_arm)
            # # 拼接 qpos_arm 和 qpos的第7及以后的元素
            # qpos = np.concatenate([qpos_arm[:7], qpos[7:]])
            # print("qpos: ", qpos)
            # self.qpos_arm_last = qpos[:7]

        # -----------------------------Benchmark---------------------------------

        print(f"retarget opt time cost: {(time.perf_counter() - t1):.3f}")
        optimization_time = time.perf_counter() - t1

        position_err = self.robot_benchmark.position_error(qpos, hand_kps_in_world, 1)
        orientation_err = self.robot_benchmark.orientation_error(qpos, hand_kps_in_world, 1)
        relative_position_err = self.robot_benchmark.relative_position_error(qpos, hand_kps_in_world, 1)

        err = {
            "position_err": position_err,
            "orientation_err": orientation_err,
            "relative_position_err": relative_position_err,
            "optimization_time": optimization_time,
        }
        # err = orientation_err

        if self.use_real_hardware:
            qpos = self.ema_alpha * qpos + (1 - self.ema_alpha) * self.qpos_last
        self.qpos_last = qpos

        if self.robot_mujoco:
            self.robot_mujoco.set_joint_pos(qpos)
            self.robot_mujoco.sim_step(refresh=True)

        return qpos, err


def main():
    repo_root = Path(__file__).resolve().parents[5]
    robot_config = load_robot_config(repo_root / default_robot_config_path("leap"))
    urdf_file_name = robot_config.robot_file_path
    actuated_joints_name = list(robot_config.actuated_joints)

    robot_model = RobotPinocchio(
        robot_file_path=urdf_file_name,
        robot_file_type="urdf",
    )
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=actuated_joints_name,
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

    teleop = RobotTeleoperation(
        hand_type=robot_config.hand_type,
        robot_adaptor=robot_adaptor,
        robot_control=None,
        qpos_init=np.asarray(robot_config.initial_qpos, dtype=float),
        input_device="vision_pro",
        mujoco_vis=False,
        use_real_hardware=False,
    )
    data = np.load("data/test_teleop/vision_pro/data.npy", allow_pickle=True).item()
    stream_data = data["stream"]

    for i in range(len(stream_data) - 1):
        print("frame: ", i)
        teleop.vision_pro_retarget(stream=stream_data[i])


if __name__ == "__main__":
    main()
