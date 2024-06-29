#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

// libraries required in this source file

// iostream: std::string
#include <iostream>
// vector: std::vector
#include <vector>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

namespace fsu {
    // create function declarations
    void create_video_result_dir(const std::string video_path, std::vector<std::string>& video_result_subdirs);
    void create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file);
}

#endif // FILESYSTEM_UTILS_H