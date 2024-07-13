#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

/* Libraries required in this source file */

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// videoio: cv::VideoCapture
#include <opencv2/videoio.hpp>

/* Video utils namespace */
namespace vu {
    // Error constants
    const int VIDEO_READ_ERROR = -1;
    const int VIDEO_WRITE_ERROR = -2;

    // Get function declarations
    void get_video_paths(std::vector<cv::String>& video_paths);
    void get_video_captures(const std::vector<cv::String> video_paths, std::vector<cv::VideoCapture>& captures);

    // Read function declarations
    void read_video_frames(cv::VideoCapture capture, std::vector<cv::Mat>& video_frames);

    // Show function declarations
    void show_video_frames(const std::vector<cv::Mat> video_frames);

    // Save function declarations
    void save_video(std::vector<cv::Mat>& video_frames, const int fourcc, const double fps, const int width, const int height, const std::string video_path);
}

#endif // VIDEO_UTILS_H