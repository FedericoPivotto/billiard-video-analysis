#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

// libraries required in this source file

// iostream: std::string
#include <iostream>

// iostream: std::vector
#include <vector>

namespace fsu {
    // create function declarations
    void create_video_result_dir(const std::string video_path, std::vector<std::string>& video_result_subdirs);
}

#endif // FILESYSTEM_UTILS_H