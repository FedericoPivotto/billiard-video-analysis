#ifndef SEGMENTATION_H
#define SEGMENTATION_H

// libraries required in this source file

// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// object_detection: od::Ball
#include <object_detection.h>

namespace sg {
    void points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points);
    void field_segmentation(std::vector<cv::Point2f>& corners, cv::Mat& frame);
    void ball_segmentation(od::Ball ball_bbox, cv::Mat& frame);
    void segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, 
const std::vector<cv::Point2f> corners, cv::Mat& frame_segmentation);
}

#endif // SEGMENTATION_H