#ifndef SEGMENTATION_H
#define SEGMENTATION_H

// libraries required in this source file

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// object_detection: od::Ball
#include <object_detection.h>

namespace sg {
    void ball_segmentation(od::Ball ball_bbox, const cv::Mat& frame);
    std::vector<cv::Point> convertToIntegerPoints(const std::vector<cv::Point2f>& floatPoints);
}

#endif // SEGMENTATION_H