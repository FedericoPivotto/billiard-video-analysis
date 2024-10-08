/* Federico Pivotto */

#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

/* Libraries required */
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <filesystem>

/* User-defined libraries required */
#include <object_detection.h>

/* Filesystem utils namespace */
namespace fsu {
    // Create function declarations
    void create_video_result_dir(const std::string video_path, std::string& video_result_path, std::vector<std::string>& video_result_subdirs);
    void create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, std::string& bboxes_frame_file);

    // Get function declarations
    void get_video_dataset_dir(const std::string video_path, std::vector<std::string>& video_dataset_subdirs);
    void get_bboxes_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, std::string& bboxes_frame_file_path);
    void get_video_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, std::string& bboxes_frame_file_path);
    void get_metrics_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string metrics_video_path, std::string& metrics_frame_file_path);

    // Write function declarations
    void write_ball_bbox(std::ofstream& bboxes_frame_file, od::Ball ball);

    // Read function declarations
    void read_ball_bboxes(const std::string bboxes_frame_file_path, std::vector<od::Ball>& balls, const bool confidence_flag = false);

    // Save function declarations
    void save_video_frame(const std::vector<cv::Mat>& video_frames, const int n_frame, const cv::Mat& frame, const std::string& video_result_subdir);
    void save_video_metrics(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string metrics_result, const std::string& video_result_subdir);
}

#endif // FILESYSTEM_UTILS_H