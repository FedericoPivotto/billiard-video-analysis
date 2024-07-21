/* Leonardo Egidati */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

/* Libraries required */
#include <opencv2/highgui.hpp>

/* User-defined libraries required */
#include <object_detection.h>

/* Segmentation namespace */
namespace sg {
    // Segmentation color constants
    const std::pair<int, cv::Scalar> BACKGROUND_BGR(0, cv::Scalar(0, 0, 0));
    const std::pair<int, cv::Scalar> WHITE_BALL_BGR(1, cv::Scalar(255, 255, 255));
    const std::pair<int, cv::Scalar> BLACK_BALL_BGR(2, cv::Scalar(0, 0, 0));
    const std::pair<int, cv::Scalar> SOLID_BALL_BGR(3, cv::Scalar(255, 185, 35));
    const std::pair<int, cv::Scalar> STRIPE_BALL_BGR(4, cv::Scalar(0, 0, 255));
    const std::pair<int, cv::Scalar> FIELD_BGR(5, cv::Scalar(35, 125, 55));

    // Segmentation mask color constants
    const std::pair<int, cv::Scalar> BACKGROUND_MASK(0, cv::Scalar(0, 0, 0));
    const std::pair<int, cv::Scalar> WHITE_BALL_MASK(1, cv::Scalar(1, 1, 1));
    const std::pair<int, cv::Scalar> BLACK_BALL_MASK(2, cv::Scalar(2, 2, 2));
    const std::pair<int, cv::Scalar> SOLID_BALL_MASK(3, cv::Scalar(3, 3, 3));
    const std::pair<int, cv::Scalar> STRIPE_BALL_MASK(4, cv::Scalar(4, 4, 4));
    const std::pair<int, cv::Scalar> FIELD_MASK(5, cv::Scalar(5, 5, 5));  

    // Segmentation function declarations
    void segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, bool test_flag = false);
    void segmentation_mask(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, bool test_flag = false);

    // Auxiliary function declarations
    void field_segmentation(const std::vector<cv::Point2f>& corners, cv::Mat& frame, const cv::Scalar field_color = FIELD_BGR.second);
    void ball_segmentation(od::Ball ball_bbox, cv::Mat& frame, const bool mask_flag = false);
    
    // Utility function declaration
    void points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points);
}

#endif // SEGMENTATION_H