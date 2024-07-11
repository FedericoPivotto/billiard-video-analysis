#ifndef SEGMENTATION_H
#define SEGMENTATION_H

// libraries required in this source file

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// object_detection: od::Ball
#include <object_detection.h>

namespace sg {
    void field_segmentation(const std::vector<cv::Point2f> corners, cv::Mat& frame);
    void ball_segmentation(od::Ball ball_bbox, cv::Mat& frame);
    std::vector<cv::Point> convertToIntegerPoints(const std::vector<cv::Point2f>& floatPoints);
}

#endif // SEGMENTATION_H