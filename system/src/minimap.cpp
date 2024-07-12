#include <minimap.h>

// librarires required in this source file and not already included in minimap.h

// iostream: std::cout, std::endl
#include <iostream>
// calib3d: findHomography()
#include <opencv2/calib3d.hpp>
// imgproc: warpPerspective(), cv::circle
#include <opencv2/imgproc.hpp>

/* Compute the slope of a line expressed in polar representation (rho, theta) */
double mm::compute_slope(const double theta){
    const double epsilon = 1e-1;

    // Check if line is vertical, otherwise compute slope    
    if((std::abs(theta) < epsilon) || std::abs(theta - CV_PI) < epsilon){
        return std::numeric_limits<double>::infinity();
    } else {
        return - 1.0 / std::tan(theta);
    }
}


/* Check whether the video point of view is affected by distortion */
void mm::check_perspective_distortion(const std::vector<cv::Vec2f>& borders, bool& is_distorted){
    is_distorted = false;
    double slope_sum = 0.0;

    // Sum the slopes of the borders 
    for(const cv::Vec2f& border : borders){
        double slope = mm::compute_slope(border[1]);

        if(slope == std::numeric_limits<double>::infinity()){
            slope_sum += 1000;
        } else {
            slope_sum += slope;
        }
    }

    // Set threshold for distortion check
    const double threshold = 100.0;

    // Check for presence of distortion
    if(slope_sum <= threshold){
        is_distorted = true;
    }
}


/* Compute the pixel location in the warped image w.r.t. the original image */
void mm::warped_pixel(const cv::Point2f& point, const cv::Mat& map_perspective, cv::Point2f& warped_point){
    
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
void mm::create_map_view(const cv::Mat& image, cv::Mat& map_view, const std::vector<cv::Point2f>& corners, const bool is_distorted){
    std::vector<cv::Point2f> dst;

    // Check table orientation
    if(!is_distorted){
        dst = {cv::Point2f(0, 0), cv::Point2f(350, 0), cv::Point2f(350, 200), cv::Point2f(0, 200)};
    } else {
        dst = {cv::Point2f(0, 200), cv::Point2f(0, 0), cv::Point2f(350, 0), cv::Point2f(350, 200)};
    }

    // Get perspective transform matrix
    cv::Mat map_perspective = cv::findHomography(corners, dst);

    // Generate map view
    cv::warpPerspective(image, map_view, map_perspective, cv::Size(350, 200));

    cv::Point2f warped_point;

    // TEST---------------------------
    mm::warped_pixel(cv::Point2f(268, 317), map_perspective, warped_point);
    // TODO: print to remove
    std::cout<<"WARPED POINT: "<< warped_point << std::endl;
    cv::circle(map_view, warped_point, 6, cv::Scalar(0,0,255));
}


/* Overlay the map-view into the current frame */
void mm::overlay_map_view(cv::Mat& frame, const cv::Mat& map_view){
    // Consider offsets for the coordinates
    const int x = 10;
    const int y = frame.rows - map_view.rows - 10;

    // Set the region of interest and to overlay on it
    cv::Rect roi(x, y, map_view.cols, map_view.rows);
    map_view.copyTo(frame(roi));
}


/* Computes map-view of the current frame */
void mm::compute_map_view(cv::Mat& map_view, const cv::Mat& first_frame, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners){
    
    // Check for presenceof distortion
    bool is_distorted = false;
    mm::check_perspective_distortion(borders, is_distorted);

    // Create map-view
    mm::create_map_view(first_frame, map_view, corners, is_distorted);
}