#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

// libraries required in this source file

// iostream: std::string
#include <iostream>
// vector: std::vector
#include <vector>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// fstream: std::ofstream
#include <fstream>

// user-defined libraries required in this source file

// object detection library
#include <object_detection.h>

namespace fsu {
    // create function declarations
    void create_video_result_dir(const std::string video_path, std::vector<std::string>& video_result_subdirs);
    void create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file);

    // get function declarations
    void get_bboxes_frame_file_path(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file_path);

    // write function declarations
    void write_ball_bbox(std::ofstream& bboxes_frame_file, od::Ball ball);

    // read function declarations
    void read_ball_bboxes(const std::string bboxes_frame_file_path, std::vector<od::Ball>& balls);
}

#endif // FILESYSTEM_UTILS_H