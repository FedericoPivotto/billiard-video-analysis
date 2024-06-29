#include <filesystem_utils.h>

// librarires required in this source file and not already included in video_utils.h

// filesystem: std::filesystem::exists(), std::filesystem::create_directory()
#include <filesystem>

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