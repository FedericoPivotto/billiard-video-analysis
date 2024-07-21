/* Fabrizio Genilotti */

#include <border_detection.h>

/* Librarires required and not yet included in border_detection.h */
#include <iostream>
#include <opencv2/imgproc.hpp>

/* Perform border detection on the given frame */
void bd::border_detection(cv::Mat& video_frame, std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners) {
    // Frame pre-processing
    cv::Mat preprocessed_video_frame;
    cv::bilateralFilter(video_frame, preprocessed_video_frame, 9, 100.0, 75.0);
    cv::cvtColor(preprocessed_video_frame, preprocessed_video_frame, cv::COLOR_BGR2HSV);

    // Mask generation by ranged HSV color segmentation
    cv::Mat mask;
    cv::Scalar lower_hsv(60, 150, 110);
    cv::Scalar upper_hsv(120, 255, 230); 
    bd::hsv_mask(preprocessed_video_frame, mask, lower_hsv, upper_hsv);
  
    // Compute edge map of the mask by Canny edge detection
    cv::Mat edge_map;
    double upper_th = 100.0;
    double lower_th = 10.0;
    cv::Canny(mask, edge_map, lower_th, upper_th);

    // Line detection using hough lines
    bd::find_borders(edge_map, borders);
    bd::find_corners(borders, corners);
}

/* Draw the borders on the current frame */
void bd::draw_borders(cv::Mat& image, const std::vector<cv::Vec2f>& borders, const std::vector<cv::Point2f>& corners) {
    double distance_th = 5.0;

    // Draw the borders
    for(size_t i = 0; i < borders.size(); i++) {
        float rho = borders[i][0], theta = borders[i][1];
        double a = std::cos(theta), b = std::sin(theta);

        // Collect corners belonging to the current border
        std::vector<cv::Point2f> matched_corners;

        // Check what corners belong to the current border
        for(size_t j = 0; j < corners.size(); j++) {
            if(std::fabs(corners[j].x * a + corners[j].y * b - rho) <= distance_th)
                matched_corners.push_back(corners[j]);
        }

        // Check if correct number of corners
        if(matched_corners.size() == 2)
            cv::line(image, matched_corners[0], matched_corners[1], bd::BORDER_BGR, 3, cv::LINE_AA);
    }
}

/* Find the corners of the borders */
void bd::find_corners(const std::vector<cv::Vec2f>& borders, std::vector<cv::Point2f>& corners) {
    // Compute the borders by finding lines intersections
    for(size_t i = 0; i < borders.size(); i++) {
        for(size_t j = i + 1; j < borders.size(); j++) {
            // Check if there are all four corners
            if(corners.size() == 4)
                return;

            // Find corner candidate
            cv::Point2f corner;
            bd::borders_intersection(borders[i], borders[j], corner);

            // Check corner feasibility
            if((corner.x != -1.0 && corner.y != -1.0) && (corner.x >= 0 && corner.y >= 0))
                corners.push_back(corner);
        }
    }
}

/* Sort corners in top-left, top-right, bottom-right, bottom-left */
void bd::sort_corners(std::vector<cv::Point2f>& corners) {
    // Sort by y coordinate
    for(size_t i = 0; i < corners.size(); i++) {
        for(size_t j = i + 1; j < corners.size(); j++) {
            if(corners[i].y > corners[j].y)
                std::swap(corners[i], corners[j]);
        }
    }

    // Sort by x coordinate
    if(corners[0].x >= corners[1].x)
        std::swap(corners[0], corners[1]);
    if(corners[2].x <= corners[3].x)
        std::swap(corners[2], corners[3]);
}

/* Find the possible four borders among the candidate lines */
void bd::select_borders(const std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f>& borders) {
    // List of already visited candidates (similar to already selected borders)
    std::vector<bool> visited(lines.size(), false);

    // Check number of candidates
    if(lines.size() < 4) {
        std::cerr << "Not enough lines to find a border" << std::endl;
        return;
    }

    // Find borders (by means of rho and theta comparisons)
    for(int i = 0; i < lines.size(); i++) {
        float rho_i = lines[i][0], theta_i = lines[i][1];
        
        if(!visited[i] && borders.size() < 4) {
            borders.push_back(lines[i]);
            
            for(int j = i + 1; j < lines.size(); j++) {
                float rho_j = lines[j][0], theta_j = lines[j][1];
                
                if((std::abs(rho_i - rho_j) <= 100 && std::abs(theta_i - theta_j) <= (CV_PI / 36)) && !visited[j]) {
                    visited[j] = true;    
                }
            }
        } else if(borders.size() == 4) {
            return;
        }
    }
}

/* Find the borders of the billiard table */
void bd::find_borders(const cv::Mat& edge_map, std::vector<cv::Vec2f>& borders) {
    // Find line candidates and select the four borders
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edge_map, lines, 1, CV_PI / 180, 95, 0, 0);
    bd::negative_lines(lines);
    bd::select_borders(lines, borders);
}

/* Find the intersection of two lines */
void bd::borders_intersection(const cv::Vec2f& first_line, const cv::Vec2f& second_line, cv::Point2f& corner) {
    // Compute line intersection by solving a linear system of two equations
    // The two equations are considered with the following notation:
    // a*x + b*y + c = 0
    // d*x + e*y + f = 0
    double rho_first = first_line[0], theta_first = first_line[1];
    double rho_second = second_line[0], theta_second = second_line[1];

    double a = std::cos(theta_first), b = std::sin(theta_first);
    double d = std::cos(theta_second), e = std::sin(theta_second);
    double c = rho_first * (-1.0), f = rho_second * (-1.0);

    // Check lines parallelism, if so return
    double det = d*b - e*a;
    if(std::abs(det) < 1e-1) {
        corner.x = -1.0;
        corner.y = -1.0;
        return;
    }

    // Compute intersection
    corner.x = (e*c - b*f) / det;
    corner.y = (a*f - d*c) / det;
}

/* Generate mask by ranged HSV color segmentation */
void bd::hsv_mask(const cv::Mat& hsv_frame, cv::Mat& mask, cv::Scalar lower_hsv, cv::Scalar upper_hsv) {
    // Color segmentation
    cv::inRange(hsv_frame, lower_hsv, upper_hsv, mask);

    // Dilate and erosion set operations on mask 
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::dilate(mask, mask, kernel);
    cv::erode(mask, mask, kernel);
}

/* Manage candidate lines with negative rho to make them comparable */
void bd::negative_lines(std::vector<cv::Vec2f>& lines) {
    for(int i = 0; i < lines.size() - 1; i++) {
        if(lines[i][0] < 0) {
            lines[i][0] *= -1.0;
            lines[i][1] -= CV_PI;
        }
    }
}