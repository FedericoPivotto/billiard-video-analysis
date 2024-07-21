/* Fabrizio Genilotti */

#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

/* Libraries required */
#include <opencv2/highgui.hpp>

/* Edge detection namespace */
namespace ed {
    // Color constant
    const cv::Scalar BORDER_BGR(0, 255, 255);

    // Border detection function delcarations
    void edge_detection(cv::Mat& first_frame, std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);
    void draw_borders(cv::Mat& image, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);
    
    // Border detection auxiliary function declarations
    void find_corners(const std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);
    void sort_corners(std::vector<cv::Point2f>& corners);
    void select_borders(const std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f>& borders);
    void find_borders(const cv::Mat& edge_map, std::vector<cv::Vec2f>& borders);
    void borders_intersection(const cv::Vec2f& first_line, const cv::Vec2f& second_line, cv::Point2f& corner);

    // Auxiliary function declarations
    void hsv_mask(const cv::Mat& hsv_frame, cv::Mat& mask, cv::Scalar lower_hsv, cv::Scalar upper_hsv);
    void negative_lines(std::vector<cv::Vec2f>& lines);
}

#endif // EDGE_DETECTION_H