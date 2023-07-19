cmake_minimum_required(VERSION 3.5)
project(ubipose)

# set(CMAKE_BUILD_TYPE "release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fstack-protector-all -g -Wall")
SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty/abseil-cpp)
add_subdirectory(thirdparty/SuperPoint-SuperGlue-TensorRT)
add_subdirectory(ubipose)
