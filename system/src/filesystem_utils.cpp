#include <filesystem_utils.h>

/* Librarires required in this source file and not already included in video_utils.h */

// filesystem: std::filesystem::exists(), std::filesystem::create_directory()
#include <filesystem>
// string: std::getline()
#include <string>

/* Create video result directory */
void fsu::create_video_result_dir(const std::string video_path, std::vector<std::string>& video_result_subdirs) {
    // Create result directory if not exists
    std::string result_path = "../system/result/";
    if(! std::filesystem::exists(result_path))
        std::filesystem::create_directory(result_path);

    // Create video result directory or delete if already exists
    std::string video_result_dir = std::filesystem::path(video_path).parent_path().filename();
    std::string video_result_path = result_path + video_result_dir;
    // Delete existing video result directory
    if(std::filesystem::exists(video_result_path))
        std::filesystem::remove_all(video_result_path);
    // Create video result directory
    std::filesystem::create_directory(video_result_path);

    // Create video bounding_boxes, frames, mask directories
    video_result_subdirs = {video_result_path + "/bounding_boxes", video_result_path + "/frames", video_result_path + "/masks"};
    for(std::string video_result_subdir : video_result_subdirs)
    std::filesystem::create_directory(video_result_subdir);
}

/* Create a bounding box file for the given video frame */
void fsu::create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // Create frame bboxes text file
    fsu::get_bboxes_frame_file_path(video_frames, nframe, bboxes_video_path, bboxes_frame_file_path);
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // Close frame bboxes text file
    bboxes_frame_file.close();
}

/* Get bounding box file path for the given video frame */
void fsu::get_bboxes_frame_file_path(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // Set frame bboxes text filename
    bboxes_frame_file_path = bboxes_video_path + "/frame_";
    if(nframe == 0)
        bboxes_frame_file_path += "first";
    else if (nframe == video_frames.size()-1)
        bboxes_frame_file_path += "last";
    else
        bboxes_frame_file_path += std::to_string(nframe + 1);
    bboxes_frame_file_path += "_bbox.txt";
}

/* Get file path for the given video frame */
void fsu::get_video_frame_file_path(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // Set segmentation mask frame filename
    bboxes_frame_file_path = bboxes_video_path + "/frame_";
    if(nframe == 0)
        bboxes_frame_file_path += "first";
    else if (nframe == video_frames.size()-1)
        bboxes_frame_file_path += "last";
    else
        bboxes_frame_file_path += std::to_string(nframe + 1);
    bboxes_frame_file_path += ".png";
}

/* Write ball in the given opened bounding box file stream */
void fsu::write_ball_bbox(std::ofstream& bboxes_frame_file, od::Ball ball) {
    // Write ball bounding box
    bboxes_frame_file << ball << " " << ball.confidence << std::endl;
}

/* Read balls from the bounding box file path with/without confidence value */
void fsu::read_ball_bboxes(const std::string bboxes_frame_file_path, std::vector<od::Ball>& balls, const bool confidence_flag) {
    // Open frame bboxes text file
    std::ifstream bboxes_frame_file(bboxes_frame_file_path);

    // Read frame bboxes text file lines
    std::string line;
    while(std::getline(bboxes_frame_file, line)) {
        // Stream for line parsing
        std::istringstream iss(line);
        unsigned int x, y, width, height, ball_class;
        double confidence;

        // Read ball row with/without confidence
        bool read_status;
        if(confidence_flag)
            read_status = ! (iss >> x >> y >> width >> height >> ball_class >> confidence);
        else
            read_status = ! (iss >> x >> y >> width >> height >> ball_class);

        // Skip and proceed to next line if error
        if(read_status)
            continue;

        // Create and add ball to vector
        balls.push_back(od::Ball(x, y, width, height, ball_class, confidence));
    }

    // Close frame bboxes text file
    bboxes_frame_file.close();
}

/* Get video result directory */
void fsu::get_video_dataset_dir(const std::string video_path, std::vector<std::string>& video_dataset_subdirs) {
    // Get dataset directory if not exists
    std::string dataset_path = "../dataset/";

    // Get video dataset directory
    std::string video_dataset_dir = std::filesystem::path(video_path).parent_path().filename();
    std::string video_dataset_path = dataset_path + video_dataset_dir;

    // Get video bounding_boxes, frames, mask directories
    video_dataset_subdirs = {video_dataset_path + "/bounding_boxes", video_dataset_path + "/frames", video_dataset_path + "/masks"};
}

/* Save video frame in given directory */
void fsu::save_video_frame(const std::vector<cv::Mat>& video_frames, const cv::Mat& frame, const int nframe, const std::string& video_result_subdir) {
    // Save segmentation mask frame
    std::string video_frame_file_path;
    fsu::get_video_frame_file_path(video_frames, nframe, video_result_subdir, video_frame_file_path);
    cv::imwrite(video_frame_file_path, frame);
}