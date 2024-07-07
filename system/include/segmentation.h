#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// libraries required in this source file

void ball_segmentation(od::Ball ball_bbox, cv::Mat& frame);

std::vector<cv::Point> convertToIntegerPoints(const std::vector<cv::Point2f>& floatPoints);

#endif // SEGMENTATION_H