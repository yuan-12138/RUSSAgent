cmake_minimum_required(VERSION 3.0.2)
project(image_processing)

# Install the ROS-facing trajectory projection and execution scripts.

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/path_stroke_preview.py
  scripts/path_stroke_preview_uv.py
  scripts/path_execute_server.py
  scripts/path_execute.py
  scripts/ribline_snapshot_to_path_yaml.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
