/* Leonardo Egidati */

#include <segmentation.h>

/* Librarires required and not yet included in segmentation.h */
#include <opencv2/imgproc.hpp>
#include <filesystem_utils.h>
#include <algorithm>

/* User-defined librarires required and not yet included in segmentation.h */
#include <edge_detection.h>
#include <minimap.h>

/* Table and balls segmentation */
void sg::segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, bool test_flag) {
    // Read frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Read ball bounding box from frame bboxes text file
    std::vector<od::Ball> ball_bboxes;
    bool confidence_flag = ! test_flag;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes, confidence_flag);

    // Color table pixels within the table borders
    sg::field_segmentation(corners, video_frame);
    
    // Scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // Color balls according to class
        sg::ball_segmentation(ball_bbox, video_frame);
    }
}

/* Table, balls and background mask segmentation */
void sg::segmentation_mask(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, bool test_flag) {
    // Read frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Read ball bounding box from frame bboxes text file
    std::vector<od::Ball> ball_bboxes;
    bool confidence_flag = ! test_flag;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes, confidence_flag);

    // Color table pixels within the table borders
    video_frame = cv::Mat::zeros(video_frame.size(), video_frame.type());
    sg::field_segmentation(corners, video_frame, sg::FIELD_MASK.second);
    
    // Scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // Color balls according to class
        bool mask_flag = true;
        sg::ball_segmentation(ball_bbox, video_frame, mask_flag);
    }
}

/* Field segmentation */
void sg::field_segmentation(const std::vector<cv::Point2f>& corners, cv::Mat& frame, const cv::Scalar field_color) {
    // Sorted float corners
    std::vector<cv::Point2f> sorted_corners(corners);
    ed::sort_corners(sorted_corners);

    // Round float corners to int
    std::vector<cv::Point> int_corners;
    sg::points_float_to_int(sorted_corners, int_corners);

    // Field coloring
    std::vector<std::vector<cv::Point>> fill_corners = {int_corners};
    cv::fillPoly(frame, fill_corners, field_color);
}

/* Ball segmentation */
void sg::ball_segmentation(od::Ball ball_bbox, cv::Mat& frame, const bool mask_flag) {
    // Get ball center
    cv::Point center(ball_bbox.center().first, ball_bbox.center().second);
    
    // Ball color palette
    std::vector<cv::Scalar> ball_colors_segmentation = {sg::WHITE_BALL_BGR.second, sg::BLACK_BALL_BGR.second, sg::SOLID_BALL_BGR.second, sg::STRIPE_BALL_BGR.second};
    std::vector<cv::Scalar> ball_colors_mask = {sg::WHITE_BALL_MASK.second, sg::BLACK_BALL_MASK.second, sg::SOLID_BALL_MASK.second, sg::STRIPE_BALL_MASK.second};
    std::vector<cv::Scalar>& ball_colors = mask_flag ? ball_colors_mask : ball_colors_segmentation;
    
    // Ball coloring
    cv::circle(frame, center, ball_bbox.radius(), ball_colors[ball_bbox.ball_class-1], -1);
}

/* Convert points from float to int */
void sg::points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        int_points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
}