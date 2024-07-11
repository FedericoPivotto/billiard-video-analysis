#ifndef MINIMAP_H
#define MINIMAP_H

// libraries required in this source file

// vector: std::vector
#include <vector>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

namespace mm {
    void sort_corners(std::vector<cv::Point2f>& corners);

    void create_map_view(const cv::Mat& image, cv::Mat& map_view, const std::vector<cv::Point2f>& corners, const bool is_distorted);

    double compute_slope(const double theta);

    void warped_pixel(const cv::Point2f& point, const cv::Mat& map_perspective, cv::Point2f& warped_point);

    void check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted);

    void overlay_map_view(cv::Mat& frame, const cv::Mat& map_view);

    void compute_map_view(cv::Mat& map_view, const cv::Mat& first_frame, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);
}

#endif // MINIMAP_H