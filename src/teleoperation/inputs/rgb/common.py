import mediapipe as mp
import mediapipe.framework as framework
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as sciR

from mr_utils.utils_calc import batchPosRotVec2Isometry3d
from mr_utils.utils_mano import (
    OPERATOR2MANO_RIGHT,
    OPERATOR2MANO_LEFT,
    estimate_frame_from_hand_points,
)
from teleoperation.types import SensorHandSample


def decode_rgb_sample(
    detector,
    image: np.ndarray,
    camera_K: np.ndarray,
    *,
    source_index: int | None = None,
    timestamp: float | None = None,
) -> SensorHandSample:
    """Detect and normalize one externally acquired RGB frame.

    Args:
        detector: Hand detector exposing the existing ``detect`` contract.
        image: RGB image supplied by an external transport or callback.
        camera_K: Camera intrinsic matrix used for wrist pose estimation.
        source_index: Optional source frame index.
        timestamp: Optional source timestamp in seconds.

    Returns:
        Sensor sample with presentation keypoints, including missing detections.
    """
    _, keypoints, keypoint_2d, wrist_pose = detector.detect(image, camera_K)
    return SensorHandSample(
        keypoints_wrist=keypoints,
        wrist_pose_sensor=wrist_pose,
        raw=image,
        source_index=source_index,
        timestamp=timestamp,
        presentation=keypoint_2d,
    )


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_type = hand_type
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
    ):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        return image

    def detect(self, rgb, cam_K=None):
        """
        Returns:
            joint_pos: joint positions in wrist frame.
            wrist_pose_in_cam: wrist pose in camera frame; will be None if args cam_K is None.
        """
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # Parse 3d keypoints from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = (
            keypoint_3d_array - keypoint_3d_array[0:1, :]
        )  # relative to wrist position
        mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = (
            keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
        )  # joint pos in wrist frame

        # estimate wrist frame by PnP
        if cam_K is not None:
            wrist_pose_in_cam = self.estimate_wrist_frame_in_cam(
                points_3d_in_wrist=joint_pos,
                points_2d_in_img=self.parse_keypoint_2d(keypoint_2d, rgb.shape),
                cam_K=cam_K,
                rvec_init=sciR.from_matrix(
                    mediapipe_wrist_rot @ self.operator2mano
                ).as_rotvec(),
            )
        else:
            wrist_pose_in_cam = None

        return num_box, joint_pos, keypoint_2d, wrist_pose_in_cam

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_wrist_frame_in_cam(
        points_3d_in_wrist, points_2d_in_img, cam_K, rvec_init, tvec_init=None
    ):
        """
        Function:
            estimate the wrist pose in camera frame by PnP solving
        Args:
            rvec_init: initial guess for rotation (rotation vector)
        """

        # guess a initial tvec for PnP solving
        if tvec_init is None:
            wrist_point_depth = 0.3
            wrist_point_3d = (
                wrist_point_depth
                * np.linalg.pinv(cam_K)
                @ np.array(
                    [points_2d_in_img[0][0], points_2d_in_img[0][1], 1.0]
                ).reshape(-1, 1)
            )
            tvec_init = wrist_point_3d

        points_2d_in_img = points_2d_in_img.reshape(-1, 1, 2)  # unit: pixel

        success, rvec, tvec = cv2.solvePnP(
            points_3d_in_wrist,
            points_2d_in_img,
            cam_K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec_init,
            tvec=tvec_init,
        )

        wrist_pose_in_cam = batchPosRotVec2Isometry3d(tvec, rvec).reshape(4, 4)

        return wrist_pose_in_cam
