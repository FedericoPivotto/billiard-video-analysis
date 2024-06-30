## CMake setup

# CMake versione
cmake_minimum_required(VERSION 2.8)

# Project name
project(billiard)

## Common operations

# Load OpenCV package
find_package(OpenCV REQUIRED)

# Libraries include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    system/include
    metrics/include
)

## System module build

# System utils video_utils
add_library(system_video_utils
    system/include/video_utils.h
    system/src/video_utils.cpp
)

# System utils filesystem_utils
add_library(system_filesystem_utils
    system/include/filesystem_utils.h
    system/src/filesystem_utils.cpp
)

# System executable
add_executable(system
    system/src/main.cpp
)

# Link libraries and system executable
target_link_libraries(system
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
)

## Federico system module build

# Federico system utils library
add_library(system_object_detection
    system/include/object_detection.h
    system/src/object_detection.cpp
)

# Federico system executable
add_executable(federico
    system/src/federico.cpp
)

# Link libraries and system executable
target_link_libraries(federico
    ${OpenCV_LIBS}
    system_video_utils
    system_filesystem_utils
    system_object_detection
)

## Fabrizio system module build

# Fabrizio system utils library
add_library(system_edge_detection
    system/include/edge_detection.h
    system/src/edge_detection.cpp
)

# Fabrizio system executable
add_executable(fabrizio
    system/src/fabrizio.cpp
)

# Link libraries and system executable
target_link_libraries(fabrizio
    ${OpenCV_LIBS}
    system_edge_detection
    system_video_utils  
)

## Leonardo system module build

# Leonardo system utils library
add_library(system_segmentation
    system/include/segmentation.h
    system/src/segmentation.cpp
)

# Leonardo system executable
add_executable(leonardo
    system/src/leonardo.cpp
)

# Link libraries and system executable
target_link_libraries(leonardo
    ${OpenCV_LIBS}
    system_segmentation
)

## Metrics module build

# Metrics utils library
add_library(metrics_library
    metrics/include/library.h
    metrics/src/library.cpp
)

# Metrics executable
add_executable(metrics
    metrics/src/main.cpp
)

# Link libraries and metrics executable
target_link_libraries(metrics
    ${OpenCV_LIBS}
    metrics_library
)