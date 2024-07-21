/* Fabrizio Genilotti */

#include <minimap.h>

/* Librarires required in this source file and not already included in minimap.h */

// iostream: std::cout, std::endl
#include <iostream>
// calib3d: findHomography()
#include <opencv2/calib3d.hpp>
// imgproc: warpPerspective(), cv::circle
#include <opencv2/imgproc.hpp>
// edge_detection: ed::sort_corners()
#include <edge_detection.h>
// segmentation: sg::field_segmentation()
#include <segmentation.h>

/* Compute the slope of a line expressed in polar representation (rho, theta) */
double mm::compute_slope(const double theta) {
    const double epsilon = 1e-1;

    // Check if line is vertical, otherwise compute slope    
    if((std::abs(theta) < epsilon) || std::abs(theta - CV_PI) < epsilon) {
        return std::numeric_limits<double>::infinity();
    } else {
        return - 1.0 / std::tan(theta);
    }
}

/* Check whether the video point of view is affected by distortion */
void mm::check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted) {
    is_distorted = false;
    double slope_sum = 0.0;

    // Sum the slopes of the borders 
    for(const cv::Vec2f& border : borders) {
        double slope = mm::compute_slope(border[1]);

        if(slope == std::numeric_limits<double>::infinity()) {
            slope_sum += 1000;
        } else {
            slope_sum += slope;
        }
    }

    // Set threshold for distortion check
    const double threshold = 100.0;

    // Check for presence of distortion
    if(slope_sum <= threshold) {
        is_distorted = true;
    }
}

/* Compute the pixel location in the warped image w.r.t. the original image */
void mm::warped_pixel(const cv::Point2f& point, const cv::Mat& map_perspective, cv::Point2f& warped_point) {
    // Convert original point into homogeneous coordinates
    cv::Mat homogeneous_point(3, 1, CV_64F);
    homogeneous_point.at<double>(0,0) = static_cast<double>(point.x);
    homogeneous_point.at<double>(1,0) = static_cast<double>(point.y);
    homogeneous_point.at<double>(2,0) = 1.0;
    
    // Compute warped point in homogeneous coordinates
    cv::Mat homogeneous_warped_point = map_perspective * homogeneous_point;

    // Store warped point
    warped_point.x = static_cast<float>(homogeneous_warped_point.at<double>(0,0) / homogeneous_warped_point.at<double>(2,0));
    warped_point.y = static_cast<float>(homogeneous_warped_point.at<double>(1,0) / homogeneous_warped_point.at<double>(2,0));
}

/* Generate map view of the area inside the borders */
void mm::create_map_view(const cv::Mat& image, cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<cv::Point2f>& corners, const bool is_distorted) {
    std::vector<cv::Point2f> dst;

    // Check table orientation
    if(!is_distorted) {
        dst = {cv::Point2f(0, 0), cv::Point2f(350, 0), cv::Point2f(350, 175), cv::Point2f(0, 175)};
    } else {
        dst = {cv::Point2f(0, 175), cv::Point2f(0, 0), cv::Point2f(350, 0), cv::Point2f(350, 175)};
    }

    // Get perspective transform matrix
    map_perspective = cv::findHomography(corners, dst);

    // Generate map view
    cv::warpPerspective(image, map_view, map_perspective, cv::Size(350, 175));
}

/* Overlay the balls trajectories on the map_view */
void mm::overlay_map_view_trajectories(cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<od::Ball> ball_bboxes) {
    // Warp each ball center and draw mini-map balls position
    cv::Point2f warped_point;

    // Balls position drawing
    for(const od::Ball& ball : ball_bboxes) {
        mm::warped_pixel(cv::Point2f(ball.x + ball.width / 2, ball.y + ball.height / 2), map_perspective, warped_point);
        
        // Draw ball position
        int point_radius = 0;
        cv::circle(map_view, warped_point, point_radius, mm::BALL_BORDER, cv::FILLED);
    }
}

/* Overlay the balls on the map-view */
void mm::overlay_map_view_balls(cv::Mat& map_view, cv::Mat& map_perspective, const std::vector<od::Ball> ball_bboxes) {
    // Warp each ball center and draw mini-map balls
    const int ball_radius = 6;
    cv::Point2f warped_point;

    // Balls drawing
    for(const od::Ball& ball : ball_bboxes) {
        mm::warped_pixel(cv::Point2f(ball.x + ball.width / 2, ball.y + ball.height / 2), map_perspective, warped_point);
        
        // Draw ball inside
        if(ball.ball_class == 1) {
            cv::circle(map_view, warped_point, ball_radius, mm::WHITE_BALL_BGR.second, cv::FILLED);
        } else if(ball.ball_class == 2) {
            cv::circle(map_view, warped_point, ball_radius, mm::BLACK_BALL_BGR.second, cv::FILLED);
        } else if(ball.ball_class == 3) {
            cv::circle(map_view, warped_point, ball_radius, mm::SOLID_BALL_BGR.second, cv::FILLED);
        } else if(ball.ball_class == 4) {
            cv::circle(map_view, warped_point, ball_radius, mm::STRIPE_BALL_BGR.second, cv::FILLED);
        }

        // Draw ball border
        int ball_border_thickness = 1;
        cv::circle(map_view, warped_point, ball_radius, mm::BALL_POSITION, ball_border_thickness);
    }
}

/* Overlay the map-view on the map-view background */
void mm::overlay_map_view_background(cv::Mat& map_view) {
    // Read billiard minimap background image
    cv::Mat map_view_background = cv::imread("../system/img/billiard_minimap.png");

    // Resize map-view background according to scale
    double scale = 0.255;
    cv::resize(map_view_background, map_view_background, cv::Size(), scale, scale);

    // Consider offsets for the coordinates
    const int x = (map_view_background.cols - map_view.cols) / 2;
    const int y = (map_view_background.rows - map_view.rows) / 2;

    // Set the region of interest and to overlay on it
    cv::Rect roi(x, y, map_view.cols, map_view.rows);
    map_view.copyTo(map_view_background(roi));

    // Update map-view
    map_view = map_view_background;
}

/* Overlay the map-view into the current frame */
void mm::overlay_map_view(cv::Mat& frame, cv::Mat& map_view) {
    // Resize map-view
    double scale = 0.85;
    cv::resize(map_view, map_view, cv::Size(), scale, scale);

    // Consider offsets for the coordinates
    const int x = 10;
    const int y = frame.rows - map_view.rows - 10;

    // Set the region of interest and to overlay on it
    cv::Rect roi(x, y, map_view.cols, map_view.rows);
    map_view.copyTo(frame(roi));
}

/* Computes map-view of the current frame */
void mm::compute_map_view(cv::Mat& map_view, cv::Mat& field_frame, cv::Mat& map_perspective, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners) {
    // Sorted float corners
    std::vector<cv::Point2f> sorted_corners(corners);
    ed::sort_corners(sorted_corners);

    // Field frame white table segmentation
    bool white_flag = true;
    sg::field_segmentation(sorted_corners, field_frame, mm::FIELD_BGR.second);
    
    // Check for presence of distortion
    bool is_distorted = false;
    mm::check_perspective_distortion(borders, is_distorted);

    // Create map-view
    mm::create_map_view(field_frame, map_view, map_perspective, sorted_corners, is_distorted);
}