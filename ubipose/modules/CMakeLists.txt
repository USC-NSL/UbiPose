
include_directories(
  ${CMAKE_SOURCE_DIR}/pipeline

  SYSTEM
  ${OpenCV_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
)

add_library(
  image_registrator
  image_registrator.cc
  image_registration_impl.cc
)

add_library(renderer renderer.cc)
target_link_libraries(
  renderer
  PRIVATE
  EGL
  ${OPENGL_LIBRARY}
  ${OpenCV_LIBS}
  assimp
  Eigen3::Eigen
  absl::log
)

add_library(
  unprojector 
  unprojector.cc
)
target_link_libraries(
  unprojector
  PRIVATE
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

add_library(
  superpointglue
  superpointglue.cc
)
target_include_directories(
  superpointglue
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/thirdparty/SuperPoint-SuperGlue-TensorRT/include/
  ${CMAKE_SOURCE_DIR}/thirdparty/SuperPoint-SuperGlue-TensorRT/
)
target_link_libraries(
  superpointglue
  superpointglue_lib
  yaml-cpp
  Eigen3::Eigen
)

