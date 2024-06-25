# CMake versione
cmake_minimum_required(VERSION 2.8)

# Project name
project(task)

# Load OpenCV package
find_package(OpenCV REQUIRED)

# Libraries include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Panoramic utils library
add_library(library
    include/library.h
    src/library.cpp
)

# Main executable
add_executable(${PROJECT_NAME}
    src/task.cpp
)

# Link libraries and main executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} library)