from setuptools import setup
import os
from glob import glob

package_name = 'vio_pipeline'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # Install marker file
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        # Install all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Install config files (YAMLs)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Install world files (Gazebo environments)
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DP7 Engineering Team',
    maintainer_email='engineer@honeywell.com',
    description='DP7 Autonomous Navigator: Redundant GNSS-Denied Architecture',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # The Core Architecture
            'openvins_node = vio_pipeline.openvins_node:main',
            'autopilot_node = vio_pipeline.autopilot_node:main',
            
            # The Shadow/Redundant Nodes
            'acs_node = vio_pipeline.acs_node:main',
            'fourier_vio_node = vio_pipeline.fourier_vio_node:main',
            'ratslam_node = vio_pipeline.ratslam_node:main',
            
            # Utilities
            'sanity_check_node = vio_pipeline.sanity_check_node:main',
            'dp7_mission_control = vio_pipeline.dp7_mission_control:main',
        ],
    },
)