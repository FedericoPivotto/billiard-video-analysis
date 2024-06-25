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

# System utils library
add_library(system_library
    system/include/library.h
    system/src/library.cpp
)

# System executable
add_executable(system
    system/src/main.cpp
)

# Link libraries and system executable
target_link_libraries(system
    ${OpenCV_LIBS}
    system_library
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