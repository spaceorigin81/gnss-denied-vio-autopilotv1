cmake_minimum_required(VERSION 3.8)
project(vio_pipeline)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS vio_pipeline/openvins_node.py DESTINATION lib/${PROJECT_NAME} RENAME openvins_node)
install(PROGRAMS vio_pipeline/fourier_vio_node.py DESTINATION lib/${PROJECT_NAME} RENAME fourier_vio_node)
install(PROGRAMS vio_pipeline/ratslam_node.py DESTINATION lib/${PROJECT_NAME} RENAME ratslam_node)
install(PROGRAMS vio_pipeline/autopilot_node.py DESTINATION lib/${PROJECT_NAME} RENAME autopilot_node)
install(PROGRAMS vio_pipeline/acs_node.py DESTINATION lib/${PROJECT_NAME} RENAME acs_node)

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)

ament_package()