#ifndef MINIMAP_H
#define MINIMAP_H

/* Libraries required in this source file */

// vector: std::vector
#include <vector>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

/* User-defined libraries */
#include<object_detection.h>

/* Minimap namespace */
namespace mm {
    // Minimap color constants
    const std::pair<int, cv::Scalar> WHITE_BALL_BGR(1, cv::Scalar(255, 255, 255));
    const std::pair<int, cv::Scalar> BLACK_BALL_BGR(2, cv::Scalar(0, 0, 0));
    const std::pair<int, cv::Scalar> SOLID_BALL_BGR(3, cv::Scalar(255, 232, 184));
    const std::pair<int, cv::Scalar> STRIPE_BALL_BGR(4, cv::Scalar(179, 179, 255));
    const std::pair<int, cv::Scalar> FIELD_BGR(5, cv::Scalar(255, 255, 255));
    const cv::Scalar HOLE_BGR(50, 50, 50);
    const cv::Scalar BALL_BORDER(58, 58, 58);
    const cv::Scalar BALL_POSITION(0, 0, 0);

    // Create function declarations
    void create_map_view(const cv::Mat& image, cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<cv::Point2f>& corners, const bool is_distorted);

    // Auxiliary function declarations
    double compute_slope(const double theta);
    void warped_pixel(const cv::Point2f& point, const cv::Mat& map_perspective, cv::Point2f& warped_point);
    void check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted);
    void overlay_map_view_trajectories(cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<od::Ball> ball_bboxes);
    void overlay_map_view_balls(cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<od::Ball> ball_bboxes);
    void overlay_map_view_background(cv::Mat& map_view);

    // Main function declarations
    void compute_map_view(cv::Mat& map_view, cv::Mat& field_frame, cv::Mat& map_perspective, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);
    void overlay_map_view(cv::Mat& frame, cv::Mat& map_view);
}

#endif // MINIMAP_H