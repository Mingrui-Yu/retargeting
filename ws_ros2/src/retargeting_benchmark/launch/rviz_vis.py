#!/usr/bin/python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory("my_robot_description"),
        "urdf",
        "panda_leap_paxini.xacro",
    )
    panda_mesh_dir = "package://my_robot_description/urdf/panda/meshes/"
    leap_mesh_dir = "package://my_robot_description/urdf/leap_hand/meshes/"
    rviz_config_file = os.path.join("src/retargeting_benchmark", "rviz", "vis.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_description",
                default_value=urdf_file,
                description="Path to URDF file",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace="visualize",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(
                                [
                                    "xacro ",
                                    str(urdf_file),
                                    f" panda_mesh_dir:={panda_mesh_dir}",
                                    f" leap_mesh_dir:={leap_mesh_dir}",
                                ]
                            ),
                            value_type=str,
                        )
                    },
                    {"frame_prefix": "/visualize/"},
                ],
                remappings=[("/joint_states", "joint_states")],
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                namespace="visualize",
                name="joint_state_publisher",
                output="screen",
                parameters=[{"source_list": ["robot_joint_states"]}],
                remappings=[("/joint_states", "joint_states")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config_file],
            ),
        ]
    )
