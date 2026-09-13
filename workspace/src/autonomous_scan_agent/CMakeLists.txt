cmake_minimum_required(VERSION 3.0.2)
project(autonomous_scan_agent)

# The launcher invokes project scripts directly. Catkin installs the ROS-facing
# executables while the Python library is loaded from the workspace source tree.

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/rgbpair_skel_pipeline.py
  scripts/iiwa_pose_converter.py
  scripts/capture_local_mover.py
  scripts/calib_local_mover.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
