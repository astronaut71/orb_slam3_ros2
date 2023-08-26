cmake_minimum_required(VERSION 3.5)
project(orbslam3)

# You should set the PYTHONPATH to your own python site-packages path
set(ENV{PYTHONPATH} "/opt/ros/humble/lib/python3.10/site-packages")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(ORB_SLAM3 REQUIRED)

include_directories(
  include
  ${ORB_SLAM3_ROOT_DIR}/include
  ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
  ${ORB_SLAM3_ROOT_DIR}/Thirdparty/Sophus
)

link_directories(
  include
)

add_executable(mono-inertial
   #src/monocular-inertial/mono-inertial-no-synchr.cpp
   #src/monocular-inertial/mono-inertial-node.cpp
   #src/monocular-inertial/mono-inertial-node-17-8-v1.cpp
   #src/monocular-inertial/mono-inertial-node-17-8-v2.cpp
   #src/monocular-inertial/mono-inertial-node-17-8-v3.cpp
   #src/monocular-inertial/mono-inertial-node-17-8-v4.cpp

   #src/monocular-inertial/converter.cpp
   #src/monocular-inertial/mono-inertial-node1.cpp
   #src/monocular-inertial/mono-inertial1.cpp
   #src/monocular-inertial/mono-inertial-node_13_aug.cpp
   #src/monocular-inertial/mono-inertial-13_aug.cpp
   #src/monocular-inertial/kopija.cpp
   src/monocular-inertial/monocular-inertial-slam-node4.cpp



)
ament_target_dependencies(mono-inertial rclcpp sensor_msgs tf2_ros tf2_geometry_msgs tf2 nav_msgs cv_bridge ORB_SLAM3 Pangolin OpenCV )

add_executable(mono
  src/monocular/mono.cpp
  src/monocular/monocular-slam-node.cpp
  #src/monocular/mono1.cpp
  #src/monocular/monocular-slam-node1.cpp
  #src/monocular/monocular-slam-node2.cpp
  #src/monocular/monocular-slam-node3.cpp
  #src/monocular/monocular-slam-node4.cpp


)
ament_target_dependencies(mono rclcpp sensor_msgs  tf2_ros tf2_geometry_msgs tf2 nav_msgs cv_bridge ORB_SLAM3 Pangolin OpenCV)

add_executable(rgbd
  src/rgbd/rgbd.cpp
  src/rgbd/rgbd-slam-node.cpp
)
ament_target_dependencies(rgbd rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM3 Pangolin OpenCV)

add_executable(stereo
  src/stereo/stereo.cpp
  src/stereo/stereo-slam-node.cpp
)
ament_target_dependencies(stereo rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM3 Pangolin OpenCV)

add_executable(stereo-inertial
  src/stereo-inertial/stereo-inertial.cpp
  src/stereo-inertial/stereo-inertial-node.cpp
)
ament_target_dependencies(stereo-inertial rclcpp sensor_msgs cv_bridge ORB_SLAM3 Pangolin OpenCV)

install(TARGETS mono-inertial mono rgbd stereo stereo-inertial DESTINATION lib/${PROJECT_NAME})

# Install launch files.
#install(DIRECTORY launch config vocabulary
#  DESTINATION share/${PROJECT_NAME}/)

ament_package()

