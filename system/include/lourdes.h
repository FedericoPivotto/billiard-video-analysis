#ifndef LOURDES_H
#define LOURDES_H

/* Libraries required in this source file */
// utility: std::pair
#include <utility>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// For ball class
#include <object_detection.h>

/* User-defined libraries required in this source file */

/* LOURDES namespace */
namespace lrds {
    // Object detection main declaration
    void lrds_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame);
}

#endif // LOURDES_H