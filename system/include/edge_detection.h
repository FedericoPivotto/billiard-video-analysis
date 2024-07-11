#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

// libraries required in this source file

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace ed {

    void negative_lines(std::vector<cv::Vec2f>& lines);

    void select_borders(const std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f>& borders);

    void find_borders(const cv::Mat& edge_map, std::vector<cv::Vec2f>& borders);

    void borders_intersection(const cv::Vec2f& first_line, const cv::Vec2f& second_line, cv::Point2f& corner);

    void find_corners(const std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);

    void draw_borders(cv::Mat& image, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);

    void hsv_mask(const cv::Mat& hsv_frame, cv::Mat& mask, cv::Scalar lower_hsv, cv::Scalar upper_hsv);

    void edge_detection(cv::Mat& first_frame, std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);
}

#endif // EDGE_DETECTION_H