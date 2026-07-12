import copy
import pdb

import numpy as np
from scipy.spatial.transform import Rotation as sciR
from retargeting.utils.utils_mano import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT


def parse_avp_stream_frame(r):
    """
    Args:
        r: Raw stream data provided by VisionProStreamer, or loaded from an
           offline replay file.
    Return:
        num_box: 1
        kp_pos_in_wrist_mano: hand keypoints in wrist frame
        xxx: None
        wrist_pose_in_ground_mano: wrist pose in AVP's world frame
    """
    if r is None:
        return 0, None, None, None

    operator2mano = OPERATOR2MANO_RIGHT
    wrist_pose_in_ground = r["right_wrist"]  # shape (1, 4, 4)
    kp_pose_in_wrist = r["right_fingers"]  # shape (25, 4, 4)

    kp_pose_in_ground = np.matmul(wrist_pose_in_ground, kp_pose_in_wrist)  # shape (25, 4, 4)

    wrist_pose_in_ground_mano = wrist_pose_in_ground.copy().reshape(4, 4)
    wrist_pose_in_ground_mano[0:3, 0:3] = wrist_pose_in_ground_mano[0:3, 0:3] @ operator2mano  # shape (4, 4)

    kp_pose_in_wrist_mano = np.matmul(
        np.linalg.inv(wrist_pose_in_ground_mano).reshape(1, 4, 4), kp_pose_in_ground
    )  # shape (25, 4, 4)

    # convert from AVP index to MediaPipe index
    kp_index_visionpro_to_mediapipe = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]

    kp_pose_in_wrist_mano = kp_pose_in_wrist_mano[kp_index_visionpro_to_mediapipe, :, :]  # shape (21, 4, 4)
    kp_pos_in_wrist_mano = kp_pose_in_wrist_mano[:, 0:3, 3].reshape(-1, 3)

    num_box = 1
    return num_box, kp_pos_in_wrist_mano, None, wrist_pose_in_ground_mano


class AvpDetector:
    def __init__(self):
        self.has_connect = False

    def connect(self, avp_ip: str):
        from avp_stream import VisionProStreamer

        print("Waiting for connecting to AVP ...")
        self.s = VisionProStreamer(ip=avp_ip, record=True)
        self.has_connect = True

    def get_raw_stream(self):
        if not self.has_connect:
            raise NameError("The AVP has not been connected!")
        r = self.s.latest
        return r

    @staticmethod
    def detect(r):
        return parse_avp_stream_frame(r)
