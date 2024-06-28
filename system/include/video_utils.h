#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

// libraries required in this source file

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// videoio: cv::VideoCapture
#include <opencv2/videoio.hpp>
// error constants
#define VIDEO_READ_ERROR -1

// get function declarations
void get_video_paths(std::vector<cv::String>& video_paths);
void get_video_captures(const std::vector<cv::String> video_paths, std::vector<cv::VideoCapture>& captures);

// read function declarations
void read_video_frames(cv::VideoCapture capture, std::vector<cv::Mat>& video_frames);

// show function declarations
void show_video_frames(const std::vector<cv::Mat> video_frames);

#endif // VIDEO_UTILS_H