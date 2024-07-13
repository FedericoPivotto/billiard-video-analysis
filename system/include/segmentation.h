#ifndef SEGMENTATION_H
#define SEGMENTATION_H

/* Libraries required in this source file */

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// object_detection: od::Ball
#include <object_detection.h>

/* Segmentation namespace */
namespace sg {
    // Auxiliary function declarations
    void points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points);

    // Segmentation function declarations
    void field_segmentation(const std::vector<cv::Point2f>& corners, cv::Mat& frame, const bool white_flag = false);
    void ball_segmentation(od::Ball ball_bbox, cv::Mat& frame);

    // Mask creation function declarations
    void change_color(cv::Mat& frame, cv::Scalar color, cv::Scalar new_color);
    void segmentation_mask(const cv::Mat segmentation_frame, cv::Mat& segmentation_mask);

    // Main function declaration
    void segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& frame_segmentation, bool test_flag = false);
}

#endif // SEGMENTATION_H