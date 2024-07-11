// user-defined libraries
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

// edge_detection detection library
#include <edge_detection.h>

// segmentation library
#include <segmentation.h>

void segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, 
const std::vector<cv::Point2f> corners, cv::Mat frame_segmentation) {
    // read frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // read ball bounding box from frame bboxes text file
    std::vector<od::Ball> ball_bboxes;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);
    for(od::Ball ball : ball_bboxes)
        std::cout << "Ball: " << ball << std::endl;

    for(cv::Point2f corner : corners)
        std::cout << "Corner: " << corner << std::endl;

    frame_segmentation = video_frames[n_frame].clone();

    // color table pixels within the table borders
    sg::field_segmentation(corners, frame_segmentation);
    
    // scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // color balls according to class
        sg::ball_segmentation(ball_bbox, frame_segmentation);
    }
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
        cv::Mat first_frame = video_frames[0];

        if (first_frame.empty()) {
            std::cout << "Could not open the frame!" << std::endl;
            return -1;
        }

        // Frame pre-processing
        cv::Mat preprocessed_first_frame;
        bilateralFilter(first_frame, preprocessed_first_frame, 9, 100.0, 75.0);
        cvtColor(preprocessed_first_frame, preprocessed_first_frame, cv::COLOR_BGR2HSV);

        // Mask generation by ranged HSV color segmentation
        cv::Mat mask;
        cv::Scalar lower_hsv(60, 150, 110);
        cv::Scalar upper_hsv(120, 255, 230); 
        hsvMask(preprocessed_first_frame, mask, lower_hsv, upper_hsv);
  
        // Compute edge map of the mask by canny edge detection
        cv::Mat edge_map;
        double upper_th = 100.0;
        double lower_th = 10.0;
        Canny(mask, edge_map, lower_th, upper_th);

        // Line detection using hough lines
        std::vector<cv::Vec2f> borders;
        std::vector<cv::Point2f> corners;
        findLines(edge_map, borders);
        findCorners(borders, corners);
        drawBorders(first_frame, borders, corners);

        // Show frame with borders
        //cv::namedWindow("Billiard video frame");
        //cv::imshow("Billiard video frame", first_frame);

        // Compute map view of the billiard table
        cv::Mat map_view;
        sortCorners(corners);
        createMapView(first_frame, map_view, corners);

        // Show map view
        //cv::namedWindow("Billiard map view");
        //cv::imshow("Billiard map view", map_view);

        // TODO: segmentation (Leonardo)
        // ATTENTION: dataset path is specified just for testing
        // ATTENTION: if added just to avoid repetitions while testing
        if(i == 0) {
            // first and last video frame segmentation
            std::string bboxes_test_dir = "../dataset/game1_clip1/bounding_boxes";
            std::vector<cv::Mat> frame_segmentation(2);

            segmentation(video_frames, 0, bboxes_test_dir, corners, frame_segmentation[0]);
            segmentation(video_frames, video_frames.size()-1, bboxes_test_dir, corners, frame_segmentation[1]);
            
            vu::show_video_frames(frame_segmentation);
            cv::waitKey(0);
        }

        // TODO: instruction replacing that above when segmentation is fine
        // segmentation(video_frames, 0, video_result_subdirs[0]);
        // segmentation(video_frames, video_frames.size()-1, video_result_subdirs[0]);

        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        // ATTENTION: commented video frame visualization to speed up testing
        // vu::show_video_frames(video_frames);
    }

    return 0;
}