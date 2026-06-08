import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('vio_pipeline')
    world_file = os.path.join(pkg, 'worlds', 'tunnel_world.world')
    model_file = os.path.join(pkg, 'models', 'model.sdf')

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    spawn_drone = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-file', model_file, '-entity', 'sjtu_drone', '-robot_namespace', 'simple_drone', '-z', '1.5'],
                output='screen'
            )
        ]
    )

    tfs = [
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0.25', '0.06', '0', '0', '0', '0', 'base_link', 'left_camera_link']),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0.25', '-0.06', '0', '0', '0', '0', 'base_link', 'right_camera_link']),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'])
    ]

    return LaunchDescription([LogInfo(msg='Starting Gazebo & Spawning Drone...'), gazebo, spawn_drone] + tfs)
