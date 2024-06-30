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

int lowTh, highTh;

/* Callback low threshold */
void callBackLow(int lTh, void* userdata){
    Mat dst;
    Mat image = *(static_cast<Mat*>(userdata));
    
    lowTh = lTh;
    Canny(image, dst, lTh, highTh);
    imshow("Show street", dst);
}

/* Callback high threshold */
void callBackHigh(int hTh, void* userdata){
    Mat dst;
    Mat image = *(static_cast<Mat*>(userdata));

    highTh = hTh;
    Canny(image, dst, lowTh, hTh);
    imshow("Show street", dst);
}

/* Display canny image */
void display_canny(Mat& src){
    createTrackbar("Low threshold", "Show street", NULL, 1000, callBackLow, &src);
    createTrackbar("High Threshold", "Show street", NULL, 1000, callBackHigh, &src);
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
        // Ideas: bilateral filter, histogram equalization, tv_bergman filter

        // first frame extraction
        Mat first_frame = video_frames[0];
        //cvtColor(video_frames[0], first_frame, COLOR_BGR2GRAY);

        if (first_frame.empty()) {
            cout << "Could not open or find the image!" << endl;
            return -1;
        }

        // frame pre-processing
        Mat preprocess_first_frame;
        //GaussianBlur(first_frame, first_frame, Size(5,5), 2, 0);
        //cvtColor(preprocess_first_frame, preprocess_first_frame, COLOR_BGR2GRAY);
        bilateralFilter(first_frame, preprocess_first_frame, 9, 100.0, 75.0);
        //equalizeHist(first_frame, first_frame);

        /* Show image */
        namedWindow("Show street");
        imshow("Show street", preprocess_first_frame);

        /* Display canny */
        display_canny(preprocess_first_frame);

        waitKey(0);

        // TODO: segmentation (Leonardo)
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        //show_video_frames(video_frames);
    }

    return 0;
}