project(pipeline)

message(STATUS "=================================")
message(STATUS "${CMAKE_CURRENT_LIST_FILE}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "=================================")

#find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")
find_package(OpenGL)
find_package(assimp  REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

find_package(COLMAP REQUIRED)
find_package(Boost REQUIRED filesystem system)
find_package(Ceres REQUIRED)
find_package(CUDA REQUIRED)

link_directories(
  ${COLMAP_LINK_DIRS}
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}

  SYSTEM
  ${STB_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
)

add_subdirectory(modules)
add_subdirectory(colmap)


# Initial Poses

add_library(
  initial_pose_providers_lib
  initial_pose_provider.cc
)
target_link_libraries (
  initial_pose_providers_lib
  PRIVATE
  colmap_utils
  colmap_initial_pose_provider

  absl::check
)


add_library(
  configs
  configs.cc
)
target_link_libraries(
  configs
  yaml-cpp
  absl::log
)

add_library(
  ios_utils
  ios_utils.cc
)

add_library(
  utils 
  keypoints_utils.cc
  pose_utils.cc
)


add_library(
  ubipose_controller 
  ubipose_controller.cc
)
target_link_libraries (
  ubipose_controller
  PUBLIC
  initial_pose_providers_lib
  colmap_reconstruction
  utils

  absl::absl_check
  absl::absl_log
)

add_library(
  matching 
  matching.cc
)
target_link_libraries(
  matching

  PUBLIC
  absl::absl_log
)

add_library(
  plotting 
  plotting.cc
)

add_library(
  ubipose 
  ubipose.cc
)
target_link_libraries(
  ubipose
  PUBLIC
  unprojector

  absl::absl_check
  absl::absl_log
)

# Executables
add_executable(
  ubipose_pipeline_main_ios_data 
  ubipose_pipeline_main_ios_data.cc
)
target_link_libraries(
  ubipose_pipeline_main_ios_data
  PRIVATE
  ubipose_controller
  renderer
  unprojector
  image_registrator
  superpointglue
  utils
  matching
  plotting
  ubipose
  configs
  ios_utils

  absl::absl_check
  absl::absl_log
  absl::log_initialize
  absl::log_flags
  absl::flags
  absl::flags_parse

  ${COLMAP_LIBRARIES}
  Boost::boost
  Boost::filesystem
  ${Boost_FILESYSTEM_LIBRARY}
)
