#include <video_utils.h>

// librarires required in this source file and not already included in video_utils.h

// iostream: std::cerr
#include <iostream>

// filesystem: glob()
#include <opencv2/core/utils/filesystem.hpp>

void vu::get_video_paths(std::vector<cv::String>& video_paths) {
    // benchmark dataset video paths
    const cv::String dataset_dir("../dataset/"), video_path("game*_clip*.mp4");
    bool nested_dir = true;
    cv::utils::fs::glob(dataset_dir, video_path, video_paths, nested_dir);
}

void vu::get_video_captures(const std::vector<cv::String> video_paths, std::vector<cv::VideoCapture>& captures) {
    // read videos from dataset
    for(cv::String video_path : video_paths)
        captures.push_back(cv::VideoCapture(video_path));
}

void vu::read_video_frames(cv::VideoCapture capture, std::vector<cv::Mat>& video_frames) {
    // safety check on video
    if(! capture.isOpened()){
        std::cerr << "Error: The video cannot be read"<< std::endl;
        exit(vu::VIDEO_READ_ERROR);
    }
    
    // read video frames
    for(cv::Mat frame; capture.read(frame);)
        video_frames.push_back(frame.clone());

    // release video
    capture.release();
}

void vu::show_video_frames(const std::vector<cv::Mat> video_frames) {
    // show video frames
    for(cv::Mat frame : video_frames) {
        cv::namedWindow("Frame");
        cv::imshow("Frame", frame);
        cv::waitKey(0);
    }
}