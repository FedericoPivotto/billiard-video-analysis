#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// libraries required in this source file

void negative_lines(std::vector<cv::Vec2f>& lines);

void select_borders(const std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f>& borders);

void find_borders(const cv::Mat& edge_map, std::vector<cv::Vec2f>& borders);

void borders_intersection(const cv::Vec2f& first_line, const cv::Vec2f& second_line, cv::Point2f& corner);

void find_corners(const std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);

void draw_borders(cv::Mat& image, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);

void hsv_mask(const cv::Mat& hsv_frame, cv::Mat& mask, cv::Scalar lower_hsv, cv::Scalar upper_hsv);

void sort_corners(std::vector<cv::Point2f>& corners);

void create_map_view(const cv::Mat& image, cv::Mat& map_view, const std::vector<cv::Point2f>& corners, const bool is_distorted);

double compute_slope(const double theta);

void check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted);

void overlay_map_view(cv::Mat& frame, const cv::Mat& map_view);

#endif // EDGE_DETECTION_H