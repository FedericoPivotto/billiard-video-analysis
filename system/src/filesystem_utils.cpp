#include <filesystem_utils.h>

// librarires required in this source file and not already included in video_utils.h

// filesystem: std::filesystem::exists(), std::filesystem::create_directory()
#include <filesystem>

// fstream: std::ofstream
#include <fstream>

void fsu::create_video_result_dir(const std::string video_path, std::vector<std::string>& video_result_subdirs) {
    // create result directory if not exists
    std::string result_path = "../system/result/";
    if(! std::filesystem::exists(result_path))
        std::filesystem::create_directory(result_path);

    // create video result directory or delete if already exists
    std::string video_result_dir = std::filesystem::path(video_path).parent_path().filename();
    std::string video_result_path = result_path + video_result_dir;
    // delete existing video result directory
    if(std::filesystem::exists(video_result_path))
        std::filesystem::remove_all(video_result_path);
    // create video result directory
    std::filesystem::create_directory(video_result_path);

    // create video bounding_boxes, frames, mask directories
    video_result_subdirs = {video_result_path + "/bounding_boxes", video_result_path + "/frames", video_result_path + "/masks"};
    for(std::string video_result_subdir : video_result_subdirs)
    std::filesystem::create_directory(video_result_subdir);
}

void fsu::create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file) {
    // create frame bboxes text file
    bboxes_frame_file = bboxes_video_path + "/frame_";
    if(nframe == 0)
        bboxes_frame_file += "first";
    else if (nframe == video_frames.size()-1)
        bboxes_frame_file += "last";
    else
        bboxes_frame_file += std::to_string(nframe + 1);
    bboxes_frame_file += ".txt";
    std::ofstream bboxes_file(bboxes_frame_file);

    // close frame bboxes text file
    bboxes_file.close();
}