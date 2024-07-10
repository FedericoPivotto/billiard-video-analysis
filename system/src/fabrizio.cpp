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

        // TODO: edge detection (Fabrizio) ----------------------------------------------------------------

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
        hsv_mask(preprocessed_first_frame, mask, lower_hsv, upper_hsv);
  
        // Compute edge map of the mask by canny edge detection
        Mat edge_map;
        double upper_th = 100.0;
        double lower_th = 10.0;
        Canny(mask, edge_map, lower_th, upper_th);

        // Line detection using hough lines
        vector<Vec2f> borders;
        vector<Point2f> corners;
        find_borders(edge_map, borders);
        find_corners(borders, corners);
        draw_borders(first_frame, borders, corners);

        // Show frame with borders
        namedWindow("Billiard video frame");
        imshow("Billiard video frame", first_frame);

        // Compute map view of the billiard table
        Mat map_view;
        sort_corners(corners);
        create_map_view(first_frame, map_view, corners);

        // Show map view
        namedWindow("Billiard map view");
        imshow("Billiard map view", map_view);

        waitKey(0);

        // TODO: segmentation (Leonardo) ------------------------------------------------------------------
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        //show_video_frames(video_frames);
    }

    return 0;
}