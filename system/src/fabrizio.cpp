// Setup CMake: mkdir build && cd build && cmake ..
// Compile with CMake: cd build && make

// Compile: g++ fabrizio.cpp -o fabrizio -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_core -lopencv_imgcodecs
// Execute: ./fabrizio

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

// edge_detection detection library
#include <edge_detection.h>

// error constants
#define INVALID_ARGUMENTS_ERROR -1
#define IMAGE_READ_ERROR -2

using namespace cv;
using namespace std;

/* Manage candidate lines with negative rho to make them comparable */
void negativeLines(vector<Vec2f>& lines){
    for(int i = 0; i < lines.size() - 1; i++ ){
        if(lines[i][0] < 0){
            lines[i][0] *= -1.0;
            lines[i][1] -= CV_PI;
        }
    }
}

/* Find the possible four borders among the candidate lines */
void findBorders(const vector<Vec2f> lines, vector<Vec2f>& borders){
    
    // List of already visited candidates (similar to already selected borders)
    vector<bool> visited(lines.size(), false);

    // Check number of candidates
    if(lines.size() < 4){
        cerr << "Not enough lines to find a border"<< endl;
        return;
    }

    // Find borders (by means of rho and theta comparisons)
    for(int i = 0; i < lines.size(); i++ ){
        float rho_i = lines[i][0], theta_i = lines[i][1];
        
        if(!visited[i] && borders.size() < 4){
            borders.push_back(lines[i]);
            
            for(int j = i + 1; j < lines.size(); j++ ){
                float rho_j = lines[j][0], theta_j = lines[j][1];
                
                if( (abs(rho_i - rho_j) <= 100 && abs(theta_i - theta_j) <= (CV_PI / 36)) && !visited[j] ){
                    visited[j] = true;    
                }
            }
        } else if( borders.size() == 4 ){
            return;
        }
    }

}

/* Find the borders of the billiard table */
void findLines(const Mat& edge_map, vector<Vec2f>& borders){

    // Find line candidates and select the four borders
    vector<Vec2f> lines;
    HoughLines(edge_map, lines, 1, CV_PI / 180, 95, 0, 0);
    negativeLines(lines);
    findBorders(lines, borders);
}

/* Find the intersection of two lines */
void bordersIntersection(const Vec2f& first_line, const Vec2f& second_line, Point2f& corner){

    // Compute line intersection by solving a linear system of two equations
    // The two equations are considered with the following notation:
    // a*x + b*y + c = 0
    // d*x + e*y + f = 0
    double rho_first = first_line[0], theta_first = first_line[1];
    double rho_second = second_line[0], theta_second = second_line[1];

    double a = cos(theta_first), b = sin(theta_first);
    double d = cos(theta_second), e = sin(theta_second);
    double c = rho_first * (-1.0), f = rho_second * (-1.0);

    // Check lines parallelism, if so return
    double det = d*b - e*a;
    if(abs(det) < 1e-1){
        corner.x = -1.0;
        corner.y = -1.0;
        return;
    }

    // Compute intersection
    corner.x = (e*c - b*f) / det;
    corner.y = (a*f - d*c) / det;
}

/* Find the corners of the borders */
void findCorners(const vector<Vec2f>& borders, vector<Point2f>& corners){

    // Compute the borders by finding lines intersections
    for( size_t i = 0; i < borders.size(); i++ ){
        for( size_t j = i + 1; j < borders.size(); j++ ){
            // Check if there are all four corners
            if(corners.size() == 4){
                return;
            }

            // Find corner candidate
            Point2f corner;
            bordersIntersection(borders[i], borders[j], corner);

            // Check corner feasibility
            if((corner.x != -1.0 && corner.y != -1.0) && (corner.x >= 0 && corner.y >= 0)){
                corners.push_back(corner);
            }
        }
    }
}

/* Draw the borders on the current frame */
void drawBorders(Mat& image, const vector<Vec2f>& borders, const vector<Point2f>& corners){
    double distance_th = 5.0;

    // Draw the borders
    for( size_t i = 0; i < borders.size(); i++ ){
        float rho = borders[i][0], theta = borders[i][1];
        double a = cos(theta), b = sin(theta);

        // Collect corners belonging to the current border
        vector<Point2f> matched_corners;

        // Check what corners belong to the current border
        for( size_t j = 0; j < corners.size(); j++ ){
            if(fabs(corners[j].x * a + corners[j].y * b - rho) <= distance_th){
                matched_corners.push_back(corners[j]);
            }
        }

        // Check if correct number of corners
        if(matched_corners.size() == 2){
            line(image, matched_corners[0], matched_corners[1], Scalar(0,0,255), 3, LINE_AA);
        }
    }
}

/* Generate mask by ranged HSV color segmentation */
void hsvMask(const Mat& hsv_frame, Mat& mask, Scalar lower_hsv, Scalar upper_hsv){

    // Color segmentation
    inRange(hsv_frame, lower_hsv, upper_hsv, mask);

    // Dilate and erosion set operations on mask 
    Mat kernel = getStructuringElement(MORPH_RECT, Size(21, 21));
    dilate(mask, mask, kernel);
    erode(mask, mask, kernel);
}

/* Sort corners in top-left, top-right, bottom-right, bottom-left */
void sortCorners(vector<Point2f>& corners){
        // Sort by y coordinate
        for( size_t i = 0; i < corners.size(); i++ ){
            for( size_t j = i + 1; j < corners.size(); j++ ){
                if(corners[i].y > corners[j].y){
                    swap(corners[i], corners[j]);
                }
            }
        }

        // Sort by x coordinate
        if(corners[0].x >= corners[1].x){
            swap(corners[0], corners[1]);
        }
        if(corners[2].x <= corners[3].x){
            swap(corners[2], corners[3]);
        }
}


/* Generate map view of the area inside the borders */
void createMapView(const Mat& image, Mat& map_view, const vector<Point2f>& corners){
    //vector<Point2f> dst = {Point2f(0, 0), Point2f(400, 0), Point2f(0, 250), Point2f(400, 250)};
    //vector<Point2f> dst = {Point2f(400, 250), Point2f(0, 250), Point2f(400, 0), Point2f(0, 0)};
    vector<Point2f> dst;

    cout<<corners<<endl;
    // Check table orientation
    if(norm(corners[0] - corners[3]) <= norm(corners[0] - corners[1])){
        dst = {Point2f(0, 0), Point2f(400, 0), Point2f(400, 250), Point2f(0, 250)};
    } else {
        dst = {Point2f(0, 250), Point2f(0, 0), Point2f(400, 0), Point2f(400, 250)};
    }

    // Get perspective transform matrix
    Mat map_perspective = findHomography(corners, dst);

    // Generate map view
    warpPerspective(image, map_view, map_perspective, Size(400, 250));
}


int main(int argc, char** argv) {
    // get videos paths
    std::vector<cv::String> video_paths;
    vu::get_video_paths(video_paths);

    // get video captures
    std::vector<cv::VideoCapture> captures;
    vu::get_video_captures(video_paths, captures);
    
    // for each video read frames
    for(int i = 0; i < captures.size(); ++i) {
        // read video frames
        std::vector<cv::Mat> video_frames;
        vu::read_video_frames(captures[i], video_frames);

        // create video result directory
        std::vector<std::string> video_result_subdirs;
        fsu::create_video_result_dir(video_paths[i], video_result_subdirs);
        
        // TODO: object detection (Federico)

        // TODO: edge detection (Fabrizio)

        // First frame extraction
        Mat first_frame = video_frames[0];

        if (first_frame.empty()) {
            cout << "Could not open the frame!" << endl;
            return -1;
        }

        // Frame pre-processing
        Mat preprocessed_first_frame;
        bilateralFilter(first_frame, preprocessed_first_frame, 9, 100.0, 75.0);
        cvtColor(preprocessed_first_frame, preprocessed_first_frame, COLOR_BGR2HSV);

        // Mask generation by ranged HSV color segmentation
        Mat mask;
        Scalar lower_hsv(60, 150, 110);
        Scalar upper_hsv(120, 255, 230); 
        hsvMask(preprocessed_first_frame, mask, lower_hsv, upper_hsv);
  
        // Compute edge map of the mask by canny edge detection
        Mat edge_map;
        double upper_th = 100.0;
        double lower_th = 10.0;
        Canny(mask, edge_map, lower_th, upper_th);

        // Line detection using hough lines
        vector<Vec2f> borders;
        vector<Point2f> corners;
        findLines(edge_map, borders);
        findCorners(borders, corners);
        drawBorders(first_frame, borders, corners);

        // Show frame with borders
        namedWindow("Billiard video frame");
        imshow("Billiard video frame", first_frame);

        // Compute map view of the billiard table
        Mat map_view;
        sortCorners(corners);
        createMapView(first_frame, map_view, corners);

        // Show map view
        namedWindow("Billiard map view");
        imshow("Billiard map view", map_view);

        waitKey(0);

        // TODO: segmentation (Leonardo)
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        //show_video_frames(video_frames);
    }

    return 0;
}