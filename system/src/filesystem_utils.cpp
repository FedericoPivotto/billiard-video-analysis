/* Federico Pivotto */

#include <filesystem_utils.h>

/* Librarires required and not yet included in filesystem_utils.h */
#include <string>

/* Create video result directory */
void fsu::create_video_result_dir(const std::string video_path, std::string& video_result_path, std::vector<std::string>& video_result_subdirs) {
    // Create result directory if not exists
    std::string result_path = "../system/result/";
    if(! std::filesystem::exists(result_path))
        std::filesystem::create_directory(result_path);

    // Create video result directory or delete if already exists
    std::string video_result_dir = std::filesystem::path(video_path).parent_path().filename();
    video_result_path = result_path + video_result_dir;

    // Delete existing video result directory
    if(std::filesystem::exists(video_result_path))
        std::filesystem::remove_all(video_result_path);
    // Create video result directory
    std::filesystem::create_directory(video_result_path);

    // Create video sub-directories
    video_result_subdirs = {video_result_path + "/bounding_boxes", video_result_path + "/frames", video_result_path + "/masks", video_result_path + "/edge_detection", video_result_path + "/object_detection", video_result_path + "/segmentation", video_result_path + "/output", video_result_path + "/metrics", video_result_path + "/minimap"};
    for(std::string video_result_subdir : video_result_subdirs)
    std::filesystem::create_directory(video_result_subdir);
}

/* Create a bounding box file for the given video frame */
void fsu::create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // Create frame bboxes text file
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // Close frame bboxes text file
    bboxes_frame_file.close();
}

/* Get video dataset directory */
void fsu::get_video_dataset_dir(const std::string video_path, std::vector<std::string>& video_dataset_subdirs) {
    // Get dataset directory if not exists
    std::string dataset_path = "../dataset/";

    // Get video dataset directory
    std::string video_dataset_dir = std::filesystem::path(video_path).parent_path().filename();
    std::string video_dataset_path = dataset_path + video_dataset_dir;

    // Get video bounding_boxes, frames, mask directories
    video_dataset_subdirs = {video_dataset_path + "/bounding_boxes", video_dataset_path + "/frames", video_dataset_path + "/masks"};
}

/* Get bounding box file path for the given video frame */
void fsu::get_bboxes_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // Set frame bboxes text filename
    bboxes_frame_file_path = bboxes_video_path + "/frame_";
    if(n_frame == 0)
        bboxes_frame_file_path += "first";
    else if (n_frame == video_frames.size()-1)
        bboxes_frame_file_path += "last";
    else
        bboxes_frame_file_path += std::to_string(n_frame + 1);
    bboxes_frame_file_path += "_bbox.txt";
}

/* Get file path for the given video frame */
void fsu::get_video_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string frame_video_path, std::string& video_frame_file_path) {
    // Set segmentation mask frame filename
    video_frame_file_path = frame_video_path + "/frame_";
    if(n_frame == 0)
        video_frame_file_path += "first";
    else if (n_frame == video_frames.size()-1)
        video_frame_file_path += "last";
    else
        video_frame_file_path += std::to_string(n_frame + 1);
    video_frame_file_path += ".png";
}

/* Get metrics file path for the given video frame */
void fsu::get_metrics_frame_file_path(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string metrics_video_path, std::string& metrics_frame_file_path) {
    // Set frame metrics text filename
    metrics_frame_file_path = metrics_video_path + "/frame_";
    if(n_frame == 0)
        metrics_frame_file_path += "first";
    else if (n_frame == video_frames.size()-1)
        metrics_frame_file_path += "last";
    else
        metrics_frame_file_path += std::to_string(n_frame + 1);
    metrics_frame_file_path += ".txt";
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

/* Save video frame in given directory */
void fsu::save_video_frame(const std::vector<cv::Mat>& video_frames, const int n_frame, const cv::Mat& frame, const std::string& video_result_subdir) {
    // Save video frame
    std::string video_frame_file_path;
    fsu::get_video_frame_file_path(video_frames, n_frame, video_result_subdir, video_frame_file_path);
    cv::imwrite(video_frame_file_path, frame);
}

/* Save video frame metrics in given directory */
void fsu::save_video_metrics(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string metrics_result, const std::string& video_result_subdir) {
    // Video frame metrics text file
    std::string video_metrics_file_path;
    fsu::get_metrics_frame_file_path(video_frames, n_frame, video_result_subdir, video_metrics_file_path);
    // Open metrics text file
    std::ofstream video_metrics_file(video_metrics_file_path);
    // Write metrics
    video_metrics_file << metrics_result;
    // Close metrics text file
    video_metrics_file.close();
}