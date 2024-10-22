### Fabrizio Genilotti

## CMake setup

# CMake version
cmake_minimum_required(VERSION 2.8)
# Set everything up for C++ 17 features
set(CMAKE_CXX_STANDARD 17)

## Project setup

# Project name
project(billiard)
# OpenCV package
find_package(OpenCV REQUIRED)
# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    system/include
    metrics/include)

## System libraries

# video_utils library
add_library(system_video_utils
    system/include/video_utils.h
    system/src/video_utils.cpp)
# filesystem_utils library
add_library(system_filesystem_utils
    system/include/filesystem_utils.h
    system/src/filesystem_utils.cpp)
# object_detection library
add_library(system_object_detection
    system/include/object_detection.h
    system/src/object_detection.cpp)
# border_detection library
add_library(system_border_detection
    system/include/border_detection.h
    system/src/border_detection.cpp)
# segmentation library
add_library(system_segmentation
    system/include/segmentation.h
    system/src/segmentation.cpp)
# minimap library
add_library(system_minimap
    system/include/minimap.h
    system/src/minimap.cpp)
# billiard_metric library
add_library(system_billiard_metric
    system/include/billiard_metric.h
    system/src/billiard_metric.cpp)

## System executable

add_executable(system system/src/main.cpp)
target_link_libraries(system
${OpenCV_LIBS}
system_video_utils
system_filesystem_utils
system_object_detection
system_border_detection
system_segmentation
system_minimap
system_billiard_metric)