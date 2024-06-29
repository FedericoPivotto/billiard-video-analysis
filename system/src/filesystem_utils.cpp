#include <filesystem_utils.h>

// librarires required in this source file and not already included in video_utils.h

// filesystem: std::filesystem::exists(), std::filesystem::create_directory()
#include <filesystem>

// string: std::getline()
#include <string>

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

void fsu::create_bboxes_frame_file(const std::vector<cv::Mat>& video_frames, const int nframe, const std::string bboxes_video_path, std::string& bboxes_frame_file_path) {
    // create frame bboxes text file
    bboxes_frame_file_path = bboxes_video_path + "/frame_";
    if(nframe == 0)
        bboxes_frame_file_path += "first";
    else if (nframe == video_frames.size()-1)
        bboxes_frame_file_path += "last";
    else
        bboxes_frame_file_path += std::to_string(nframe + 1);
    bboxes_frame_file_path += ".txt";
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // close frame bboxes text file
    bboxes_frame_file.close();
}

void fsu::write_ball_bbox(std::ofstream& bboxes_frame_file, od::Ball ball) {
    // write ball bounding box
    bboxes_frame_file << ball.x << " " << ball.y << " " << ball.width << " " << ball.height << " " << ball.ball_class << std::endl;
}

void fsu::read_ball_bboxes(const std::string bboxes_frame_file_path, std::vector<od::Ball>& balls) {
    // open frame bboxes text file
    std::ifstream bboxes_frame_file(bboxes_frame_file_path);

    // read frame bboxes text file lines
    std::string line;
    while(std::getline(bboxes_frame_file, line)) {
        // stream for line parsing
        std::istringstream iss(line);
        unsigned int x, y, width, height, ball_class;

        // skip and proceed to next line if error
        if (! (iss >> x >> y >> width >> height >> ball_class))
            continue;

        // create and add ball to vector
        balls.push_back(od::Ball(x, y, width, height, ball_class));
    }

    // close frame bboxes text file
    bboxes_frame_file.close();
}