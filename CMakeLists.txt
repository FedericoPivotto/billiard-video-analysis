## Initial setup

# CMake version
cmake_minimum_required(VERSION 2.8)
# Set everything up for C++ 17 features
set(CMAKE_CXX_STANDARD 17)

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
# edge_detection library
add_library(system_edge_detection
    system/include/edge_detection.h
    system/src/edge_detection.cpp)
# segmentation library
add_library(system_segmentation
    system/include/segmentation.h
    system/src/segmentation.cpp)
# minimap library
add_library(system_minimap
    system/include/minimap.h
    system/src/minimap.cpp)

# TODO: to remove
# lourdes library
add_library(system_lourdes
    system/include/lourdes.h
    system/src/lourdes.cpp)

## Metrics libraries

# billiard_metric library
add_library(metrics_billiard_metric
    metrics/include/billiard_metric.h
    metrics/src/billiard_metric.cpp)

## System executable
add_executable(system system/src/main.cpp)
target_link_libraries(system
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
    system_edge_detection
    system_segmentation
    system_minimap)

## Metrics executable
add_executable(metrics metrics/src/main.cpp)
target_link_libraries(metrics
    ${OpenCV_LIBS}
    metrics_billiard_metric
    system_filesystem_utils
    system_object_detection)

## Federico system executable
add_executable(federico system/src/federico.cpp)
target_link_libraries(federico
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
    system_edge_detection
    system_segmentation
    system_minimap)

## Fabrizio system executable
add_executable(fabrizio system/src/fabrizio.cpp)
target_link_libraries(fabrizio
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
    system_edge_detection
    system_segmentation
    system_minimap)

## Leonardo system executable
add_executable(leonardo system/src/leonardo.cpp)
target_link_libraries(leonardo
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
    system_edge_detection
    system_segmentation
    system_minimap)

# TODO: to remove
## Lourdes system executable
add_executable(main system/src/main_lourdes.cpp)
target_link_libraries(main
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
    system_edge_detection
    system_segmentation
    system_minimap
    system_lourdes)