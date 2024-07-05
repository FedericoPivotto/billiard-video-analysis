#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// libraries required in this source file

void negativeLines(std::vector<cv::Vec2f>& lines);

void findBorders(const std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f>& borders);

void findLines(const cv::Mat& edge_map, std::vector<cv::Vec2f>& borders);

void bordersIntersection(const cv::Vec2f& first_line, const cv::Vec2f& second_line, cv::Point2f& corner);

void findCorners(const std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners);

void drawBorders(cv::Mat& image, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners);

void hsvMask(const cv::Mat& hsv_frame, cv::Mat& mask, cv::Scalar lower_hsv, cv::Scalar upper_hsv);

void sortCorners(std::vector<cv::Point2f>& corners);

void createMapView(const cv::Mat& image, cv::Mat& map_view, const std::vector<cv::Point2f>& corners);

#endif // EDGE_DETECTION_H