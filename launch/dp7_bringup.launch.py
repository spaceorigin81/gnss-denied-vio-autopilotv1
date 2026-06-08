#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('vio_pipeline')

    arg_config = DeclareLaunchArgument('config_path', default_value=PathJoinSubstitution([pkg_share, 'config', 'openvins_config.yaml']))
    arg_tunnel_len = DeclareLaunchArgument('tunnel_length', default_value='50.0')
    arg_height = DeclareLaunchArgument('cruise_height', default_value='1.5')
    arg_speed = DeclareLaunchArgument('max_speed', default_value='1.0')
    arg_sanity_only = DeclareLaunchArgument('sanity_check_only', default_value='false')
    arg_log = DeclareLaunchArgument('log_level', default_value='info')
    arg_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')

    config_path = LaunchConfiguration('config_path')
    tunnel_length = LaunchConfiguration('tunnel_length')
    cruise_height = LaunchConfiguration('cruise_height')
    max_speed = LaunchConfiguration('max_speed')
    sanity_only = LaunchConfiguration('sanity_check_only')
    log_level = LaunchConfiguration('log_level')
    use_sim_time = LaunchConfiguration('use_sim_time')

    set_sim_time = SetParameter(name='use_sim_time', value=use_sim_time)

    # 1. CORE VIO (Active Flight Estimator)
    openvins_node = Node(
        package='vio_pipeline',
        executable='openvins_node',
        name='openvins_node',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        condition=UnlessCondition(sanity_only),
        parameters=[
            {'config_path': config_path},
            {'use_sim_time': use_sim_time},
            {'imu_topic': '/simple_drone/imu/out'},
            {'cam0_topic': '/simple_drone/left_camera/image_raw'},
            {'cam1_topic': '/simple_drone/right_camera/image_raw'},
        ],
        remappings=[
            ('/ov_msckf/odomimu', '/ov_msckf/odomimu'),
        ]
    )

    # 2. ANOMALY MONITOR
    acs_node = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='vio_pipeline',
                executable='acs_node',
                name='acs_node',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                condition=UnlessCondition(sanity_only),
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'nis_window': 30},
                    {'scale_max': 8.0},
                    {'scale_min': 0.5},
                    {'inflate_rate': 1.25},
                    {'deflate_rate': 0.98},
                ]
            )
        ]
    )

    # 3. FOURIER VIO (Shadow Mode - Gagged cmd_vel)
    fourier_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='vio_pipeline',
                executable='fourier_vio_node',
                name='fourier_vio_node',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                condition=UnlessCondition(sanity_only),
                parameters=[
                    {'use_sim_time': use_sim_time},
                ],
                remappings=[
                    ('/ov_msckf/odomimu', '/acs/odometry'),
                    ('/simple_drone/left_camera/image_raw', '/simple_drone/left_camera/image_raw'),
                    ('/simple_drone/imu/out', '/simple_drone/imu/out'),
                    ('/simple_drone/cmd_vel', '/shadow/fourier/cmd_vel'), 
                ]
            )
        ]
    )

    # 4. RATSLAM (Shadow Mode - Gagged cmd_vel)
    ratslam_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='vio_pipeline',
                executable='ratslam_node',
                name='ratslam_node',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                condition=UnlessCondition(sanity_only),
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'match_threshold': 0.88},
                    {'min_template_spacing': 0.5},
                    {'descriptor_size': 64},
                ],
                remappings=[
                    ('/fourier_vio/odometry', '/fourier_vio/odometry'),
                    ('/simple_drone/left_camera/image_raw', '/simple_drone/left_camera/image_raw'),
                    ('/simple_drone/cmd_vel', '/shadow/ratslam/cmd_vel'), 
                ]
            )
        ]
    )

    # 5. AUTOPILOT (Pointed securely at High-Precision VIO)
    autopilot_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='vio_pipeline',
                executable='autopilot_node',
                name='autopilot_node',
                output='screen',
                arguments=['--ros-args', '--log-level', log_level],
                condition=UnlessCondition(sanity_only),
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'tunnel_length': tunnel_length},
                    {'cruise_height': cruise_height},
                    {'max_speed': max_speed},
                    {'kp': 1.2},
                    {'ki': 0.05},
                    {'kd': 0.3},
                ],
                remappings=[
                    ('/tunnel_nav/odometry', '/ov_msckf/odomimu'), 
                    ('/simple_drone/odom', '/simple_drone/odom'),
                    ('/ov_msckf/initialized', '/ov_msckf/initialized'),
                    ('/simple_drone/cmd_vel', '/simple_drone/cmd_vel'),
                ]
            )
        ]
    )

    sanity_node = Node(
        package='vio_pipeline',
        executable='sanity_check_node',
        name='sanity_check_node',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
        condition=IfCondition(sanity_only),
        parameters=[
            {'use_sim_time': use_sim_time},
        ]
    )

    return LaunchDescription([
        arg_config,
        arg_tunnel_len,
        arg_height,
        arg_speed,
        arg_sanity_only,
        arg_log,
        arg_use_sim_time,
        set_sim_time,
        LogInfo(msg='Starting DP7 Autonomous Navigator — Shadow Mode Architecture'),
        openvins_node,
        acs_node,
        fourier_node,
        ratslam_node,
        autopilot_node,
        sanity_node,
    ])