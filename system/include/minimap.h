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
    // Create function declarations
    void create_map_view(const cv::Mat& image, cv::Mat& map_view, const std::vector<cv::Point2f>& corners, const bool is_distorted, const std::vector<od::Ball> ball_bboxes);

    // Auxiliary function declarations
    double compute_slope(const double theta);
    void warped_pixel(const cv::Point2f& point, const cv::Mat& map_perspective, cv::Point2f& warped_point);
    void check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted);
    void draw_map_view_details(cv::Mat& map_view, const int ball_radius);

    // Main function declarations
    void compute_map_view(cv::Mat& map_view, cv::Mat& field_frame, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners, const std::vector<od::Ball> ball_bboxes);
    void overlay_map_view(cv::Mat& frame, const cv::Mat& map_view);
}

#endif // MINIMAP_H