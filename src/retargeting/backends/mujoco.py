import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import open3d as o3d
from retargeting.utils.utils_calc import (
    camera_orientation_opengl_to_common,
    posQuat2Isometry3d,
    posRotMat2Isometry3d,
    quatWXYZ2XYZW,
    rgbd_to_pointcloud,
    transformPositions,
)
from retargeting.utils.utils_open3d import create_coordinate_frame, o3d_vis_pointcloud_with_color, rgbd_to_pointcloud_by_open3d


# Function to check if body name exists
def body_name_exists(model, body_name):
    try:
        _ = model.body(body_name).id
        return True  # The body exists
    except IndexError:
        return False  # The body does not exist


class RobotMujoco:
    def __init__(self, robot_file_path):
        # hyper-parameters for simulation
        self.timestep = 0.1
        self.sim_timestep = 0.001
        self.viewer_cam_distance = 2.0
        self.view_cam_lookat = [0.2, -0.2, 0.2]
        self.viewer_fps = 50
        self.disable_gravity = False

        # hyper-parameters for the robot
        self.n_arm_joints = 7
        self.arm_joint_names = [f"panda_joint{i + 1}" for i in range(self.n_arm_joints)]
        self.arm_actuator_names = [f"panda_actuator_{i + 1}" for i in range(self.n_arm_joints)]
        self.n_hand_joints = 16
        self.hand_joint_names = [f"joint_{i}" for i in range(self.n_hand_joints)]
        self.hand_actuator_names = [f"actuator_{i}" for i in range(self.n_hand_joints)]
        self.fingertip_names = [
            "thumb_fingertip_new",
            "fingertip_new",
            "fingertip_2_new",
            "fingertip_3_new",
        ]
        self.fingertip_center_names = [
            "thumb_tip_center",
            "finger1_tip_center",
            "finger2_tip_center",
            "finger3_tip_center",
        ]
        self.fingertip_geom_names = [f"{fingertip_name}_g" for fingertip_name in self.fingertip_names]

        self.n_joints = self.n_arm_joints + self.n_hand_joints
        self.joint_names = self.arm_joint_names + self.hand_joint_names
        self.actuator_names = self.arm_actuator_names + self.hand_actuator_names

        # variable
        self.n_step = 0
        self.target_joint_pos = np.zeros((self.n_joints))
        self.joint_torques = np.zeros((self.n_joints))

        self.model = mujoco.MjModel.from_xml_path(robot_file_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_timestep
        if self.disable_gravity:
            self.model.opt.gravity = [0.0, 0.0, 0.0]  # disable gravity
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = self.viewer_cam_distance
        self.viewer.cam.lookat = self.view_cam_lookat
        self.viewer.cam.azimuth = -90
        self.viewer.cam.elevation = -25
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        self.initial_robot_config()

    def initial_robot_config(self):
        joint_pos = np.zeros((self.n_joints))
        joint_pos[:7] = np.array([0, -np.pi / 4, 0.0, -3.0 / 4.0 * np.pi, 0, np.pi / 2.0, 1.0 / 4.0 * np.pi])
        self.set_joint_pos(joint_pos)
        self.sim_step(refresh=True)

    def step(self):
        for i in range(int(self.timestep / self.sim_timestep)):
            self.sim_step()

    def sim_step(self, refresh=False):
        """
        One low-level simulation step.
        """
        mujoco.mj_step(self.model, self.data)
        if self.n_step % (int(1.0 / self.sim_timestep) / self.viewer_fps) == 0 or refresh:
            self.viewer.sync()
        self.n_step += 1

    def set_joint_pos(self, qpos):
        """
        Force set the joint positions (ignoring physics)

        """
        assert len(qpos) == self.n_joints
        for i, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos = qpos[i]
        self.ctrl_joint_pos(qpos)

    def get_joint_pos(self):
        joint_pos = np.zeros((self.n_joints))
        for i, joint_name in enumerate(self.joint_names):
            joint_pos[i] = self.data.joint(joint_name).qpos[0]
        return joint_pos

    def get_target_joint_pos(self):
        return self.target_joint_pos.copy()

    def get_joint_torques(self):
        for i, actuator_name in enumerate(self.actuator_names):
            self.joint_torques[i] = self.data.actuator(actuator_name).force[0]
        return self.joint_torques.copy()

    def ctrl_joint_pos(self, target_joint_pos):
        assert len(target_joint_pos) == self.n_joints
        for i, actuator_name in enumerate(self.actuator_names):
            self.data.actuator(actuator_name).ctrl = target_joint_pos[i]
        self.target_joint_pos[:] = target_joint_pos[:]

    def get_object_pose(self, body_name):
        name = body_name
        if not body_name_exists(self.model, name):
            raise NameError(f"Body {name} does not exist.")

        pos = self.data.body(name).xpos.copy()
        rot_mat = self.data.body(name).xmat.copy().reshape(3, 3)
        pose = posRotMat2Isometry3d(pos, rot_mat)
        return pose


def test_env():
    robot_mujoco = RobotMujoco(robot_file_path="./assets/scenes/scene.xml")
    while True:
        step_start = time.time()
        robot_mujoco.step()
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = robot_mujoco.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    test_env()
