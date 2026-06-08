#!/usr/bin/env python3
"""
sanity_check.launch.py — DP7 Pre-Flight Verification
=====================================================
Runs ONLY the sanity check node for 10 seconds, prints
pass/fail report, then exits cleanly.

Usage:
  ros2 launch vio_pipeline sanity_check.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():

    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulated clock')

    return LaunchDescription([
        arg_sim_time,
        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),

        LogInfo(msg=''),
        LogInfo(msg='🔍 DP7 PRE-FLIGHT SANITY CHECK — running 10 s observation window...'),
        LogInfo(msg='   Make sure Gazebo is running with the drone spawned first.'),
        LogInfo(msg=''),

        Node(
            package='vio_pipeline',
            executable='sanity_check_node',
            name='sanity_check_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
