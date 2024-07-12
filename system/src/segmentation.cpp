#include <segmentation.h>

/* Librarires required in this source file and not already included in segmentation.h */

// imgproc: cv::circle
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::get_bboxes_frame_file_path(), fsu::read_ball_bboxes()
#include <filesystem_utils.h>
// edge_detection: ed::sort_corners()
#include <edge_detection.h>

/* Field segmentation */
void sg::field_segmentation(const std::vector<cv::Point2f>& corners, cv::Mat& frame, const bool white_flag) {
    // Sorted float corners
    std::vector<cv::Point2f> sorted_corners(corners);
    ed::sort_corners(sorted_corners);

    // Round float corners to int
    std::vector<cv::Point> int_corners;
    sg::points_float_to_int(sorted_corners, int_corners);

    // Field coloring
    std::vector<std::vector<cv::Point>> fill_corners = {int_corners};
    cv::Scalar field_color = white_flag ? cv::Scalar(190, 190, 190) : cv::Scalar(35, 125, 55);
    cv::fillPoly(frame, fill_corners, field_color);
}

/* Ball segmentation */
void sg::ball_segmentation(od::Ball ball_bbox, cv::Mat& frame) {
    // Get ball center
    cv::Point center(ball_bbox.center().first, ball_bbox.center().second);
    // Ball color palette
    std::vector<cv::Scalar> ball_colors = {cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0), cv::Scalar(255, 185, 35), cv::Scalar(0, 0, 255)};
    // Ball coloring
    cv::circle(frame, center, ball_bbox.radius(), ball_colors[ball_bbox.ball_class-1], -1);
}

/* Convert points from float to int */
void sg::points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& int_points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        int_points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
}

/* Table and balls segmentation */
void sg::segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, 
const std::vector<cv::Point2f> corners, cv::Mat& video_frame, bool test_flag) {
    // Read frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Read ball bounding box from frame bboxes text file
    std::vector<od::Ball> ball_bboxes;
    if(test_flag)
        fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);
    else
        fsu::read_ball_bboxes_with_confidence(bboxes_frame_file_path, ball_bboxes);

    // Color table pixels within the table borders
    sg::field_segmentation(corners, video_frame);
    
    // Scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // Color balls according to class
        sg::ball_segmentation(ball_bbox, video_frame);
    }
}