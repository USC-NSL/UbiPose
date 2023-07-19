
add_library(
  colmap_reconstruction
  colmap_reconstruction.cc
)
target_link_libraries(
  colmap_reconstruction
  PRIVATE

  ${COLMAP_LIBRARIES}
)

add_library(
  colmap_utils
  colmap_utils.cc
)

add_library(
  colmap_initial_pose_provider
  colmap_initial_pose_provider.cc
)
target_link_libraries(
  colmap_initial_pose_provider

  PRIVATE
  colmap_utils
  colmap_reconstruction
)
