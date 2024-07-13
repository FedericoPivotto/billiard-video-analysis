#include <segmentation.h>

/* Librarires required in this source file and not already included in segmentation.h */

// imgproc: cv::circle
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::get_bboxes_frame_file_path(), fsu::read_ball_bboxes()
#include <filesystem_utils.h>
// edge_detection: ed::sort_corners()
#include <edge_detection.h>
// algorithm: std::find()
#include <algorithm>

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

/* Replace a specified color with another one */
void sg::change_color(cv::Mat& frame, const cv::Scalar color, const cv::Scalar new_color) {
    // Replace color with new color
    cv::Mat mask;
    cv::inRange(frame, color, color, mask);
    frame.setTo(new_color, mask);
}

/* Segmentation mask */
void sg::segmentation_mask(const cv::Mat segmentation_frame, cv::Mat& segmentation_mask) {
    // Clone segmentation frame
    segmentation_mask = segmentation_frame.clone();

    // Mask for color replace
    cv::Mat mask;

    // Replace white ball (class 1) color BGR (255, 255, 255) with gray (1, 1, 1)
    sg::change_color(segmentation_mask, cv::Scalar(255, 255, 255), cv::Scalar(1, 1, 1));
    // Replace black ball (class 2) color BGR (0, 0, 0) with gray (2, 2, 2)
    sg::change_color(segmentation_mask, cv::Scalar(0, 0, 0), cv::Scalar(2, 2, 2));
    // Replace solid balls (class 3) color BGR (255, 185, 35) with gray (3, 3, 3)
    sg::change_color(segmentation_mask, cv::Scalar(255, 185, 35), cv::Scalar(3, 3, 3));
    // Replace stripe balls (class 4) color BGR (0, 0, 255) with gray (4, 4, 4)
    sg::change_color(segmentation_mask, cv::Scalar(0, 0, 255), cv::Scalar(4, 4, 4));
    // Replace field (class 5) color BGR (35, 125, 55) with gray (5, 5, 5)
    sg::change_color(segmentation_mask, cv::Scalar(35, 125, 55), cv::Scalar(5, 5, 5));

    // Gray colors list
    std::vector<cv::Vec3b> gray_colors = {cv::Vec3b(1, 1, 1), cv::Vec3b(2, 2, 2), cv::Vec3b(3, 3, 3), cv::Vec3b(4, 4, 4), cv::Vec3b(5, 5, 5)};

    // Replace remaining pixels of background (class 0) to gray (0, 0, 0)
    for(int i = 0; i < segmentation_mask.rows; i++) {
        for(int j = 0; j < segmentation_mask.cols; j++) {
            // Is present flag
            bool is_present = false;

            // Check if pixel color is not in gray colors list
            for(cv::Vec3b gray_color : gray_colors) {
                if(segmentation_mask.at<cv::Vec3b>(i, j) == gray_color) {
                    is_present = true;
                    break;
                }
            }

            // Replace pixel color with (0, 0, 0) if not in gray colors list
            if(! is_present)
                segmentation_mask.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        }
    }
}