#ifndef LOURDES_H
#define LOURDES_H

/* Libraries required in this source file */
// utility: std::pair
#include <utility>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>
// For ball class
#include <object_detection.h>

/* User-defined libraries required in this source file */

/* LOURDES namespace */
namespace lrds {
    // Object detection main declaration
    //void lrds_sift_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame);
//
    //void lrds_template_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame);

    void lrds_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, const bool is_distorted);
    
    //void frame_feature_extraction(cv::Mat frame, const std::vector<cv::Point2f>& corners, std::vector<cv::KeyPoint>& table_keypoints, cv::Mat& table_descriptors);
//
    //void ball_feature_extraction(cv::Mat frame, std::vector<cv::KeyPoint>& table_keypoints, cv::Mat& table_descriptors);
//
    //void feature_matching(cv::Mat frame, const std::vector<cv::KeyPoint>& table_keypoints, const cv::Mat& table_descriptors);

    void points_float_to_point(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points);

    void find_dominant_colors(const cv::Mat& frame, const cv::Mat& mask, int& dominant_hue, int& dominant_saturation);

    void suppress_billiard_holes(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f> corners, const bool is_distorted);

    void suppress_close_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big);

    void suppress_small_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_small);
    
    void normalize_circles_radius(std::vector<cv::Vec3f>& circles);

    //    void select_circles(std::vector<cv::Vec4f>& circles);

    void preprocess_bgr_frame(const cv::Mat& frame, cv::Mat& preprocessed_video_frame);
}

#endif // LOURDES_H