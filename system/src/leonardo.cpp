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

// segmentation library
#include <segmentation.h>

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

        // Edge detection made on the first frame
        cv::Mat first_frame = video_frames[0];
        std::vector<cv::Vec2f> borders;
        std::vector<cv::Point2f> corners;

        if (first_frame.empty()) {
            std::cout << "Could not open the frame!" << std::endl;
            return -1;
        } else {
            ed::edge_detection(first_frame, borders, corners);
            ed::draw_borders(first_frame, borders, corners);
        }

        // TODO: edge detection (Fabrizio) ----------------------------------------------------------------

        // Store bounding boxes centers and compute average ray (TODO: generalize to our results)
        std::vector<od::Ball> ball_boxes;
        std::string ball_boxes_dir_path = "../dataset/game1_clip1/bounding_boxes";
        std::string ball_boxes_path;

        fsu::get_bboxes_frame_file_path(video_frames, 0, ball_boxes_dir_path, ball_boxes_path);
        fsu::read_ball_bboxes(ball_boxes_path, ball_boxes);

        // TODO: segmentation (Leonardo)
        // ATTENTION: dataset path is specified just for testing
        // ATTENTION: if added just to avoid repetitions while testing
        if(i == 0) {
            // first and last video frame segmentation
            std::string bboxes_test_dir = "../dataset/game1_clip1/bounding_boxes";
            std::vector<cv::Mat> frame_segmentation(2);

            sg::segmentation(video_frames, 0, bboxes_test_dir, corners, frame_segmentation[0]);
            sg::segmentation(video_frames, video_frames.size()-1, bboxes_test_dir, corners, frame_segmentation[1]);
            
            vu::show_video_frames(frame_segmentation);
            cv::waitKey(0);
        }

        // TODO: instruction replacing that above when segmentation is fine
        // sg::segmentation(video_frames, 0, video_result_subdirs[0]);
        // sg::segmentation(video_frames, video_frames.size()-1, video_result_subdirs[0]);

        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        // ATTENTION: commented video frame visualization to speed up testing
        // vu::show_video_frames(video_frames);
    }

    return 0;
}