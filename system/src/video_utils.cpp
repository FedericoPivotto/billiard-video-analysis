/* Fabrizio Genilotti */

#include <video_utils.h>

/* Librarires required in this source file and not already included in video_utils.h */

// iostream: std::cerr
#include <iostream>
// filesystem: glob()
#include <opencv2/core/utils/filesystem.hpp>

/* Get dataset video paths */
void vu::get_video_paths(std::vector<cv::String>& video_paths) {
    // Benchmark dataset video paths
    const cv::String dataset_dir("../dataset/"), video_path("game*_clip*.mp4");
    bool nested_dir = true;
    cv::utils::fs::glob(dataset_dir, video_path, video_paths, nested_dir);
}

/* Get video captures from video paths given */
void vu::get_video_captures(const std::vector<cv::String> video_paths, std::vector<cv::VideoCapture>& captures) {
    // Read videos from dataset
    for(cv::String video_path : video_paths)
        captures.push_back(cv::VideoCapture(video_path));
}

/* Read video frames of the given video capture */
void vu::read_video_frames(cv::VideoCapture capture, std::vector<cv::Mat>& video_frames) {
    // Safety check on video
    if(! capture.isOpened()) {
        std::cerr << "Error: The video cannot be read." << std::endl;
        exit(vu::VIDEO_READ_ERROR);
    }
    
    // Read video frames
    for(cv::Mat frame; capture.read(frame);)
        video_frames.push_back(frame.clone());

    // Release video
    capture.release();
}

/* Show video frames given */
void vu::show_video_frames(const std::vector<cv::Mat> video_frames) {
    // Show video frames
    for(cv::Mat video_frame : video_frames) {
        cv::namedWindow("Video frame");
        cv::imshow("Video frame", video_frame);
        cv::waitKey(0);
    }
}

/* Create and save a video given a vector of frames */
void vu::save_video(std::vector<cv::Mat>& video_frames, const cv::VideoCapture capture, const std::string video_path) {
    // Video writer
    cv::VideoWriter video_writer;

    // Video parameters
    double fps = capture.get(cv::CAP_PROP_FPS);
    int width  = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Open video writer
    video_writer.open(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if(! video_writer.isOpened()) {
        std::cerr << "Error: The video cannot be written." << std::endl;
        exit(vu::VIDEO_WRITE_ERROR);
    }

    // Write video frames
    for(cv::Mat& video_frame : video_frames)
        video_writer.write(video_frame);

    // Close video writer
    video_writer.release();
}