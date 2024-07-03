// user-defined libraries

//AGGIUNTE QUESTE 4 LIBRERIE NON SO SE SIA GIUSTO
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// segmentation library
#include <segmentation.h>

void segmentation(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path) {
    // read frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // read ball bounding box from frame bboxes text file
    std::vector<od::Ball> ball_bboxes;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);
    for(od::Ball ball : ball_bboxes)
        std::cout << "Ball: " << ball << std::endl;

    // scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // TODO: color balls according to class
        
        if (ball_bbox.ball_class == 1)
        {
            std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
            cv::Point center(center_point.first, center_point.second);
            cv::Scalar color(255, 255, 255);
            cv::circle( video_frames[n_frame], center, ball_bbox.radius(), color, -1 );
        }
        else if (ball_bbox.ball_class == 2)
        {
            std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
            cv::Point center(center_point.first, center_point.second);
            cv::Scalar color(0, 0, 0);
            cv::circle( video_frames[n_frame], center, ball_bbox.radius(), color, -1 );
        }
        else if (ball_bbox.ball_class == 3)
        {
            std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
            cv::Point center(center_point.first, center_point.second);
            cv::Scalar color(255, 0, 0);
            cv::circle( video_frames[n_frame], center, ball_bbox.radius(), color, -1 );
        }
        else if(ball_bbox.ball_class == 4)
        {
            std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
            cv::Point center(center_point.first, center_point.second);
            cv::Scalar color(0, 0, 255);
            cv::circle( video_frames[n_frame], center, ball_bbox.radius(), color, -1 );
        }
    }

    // color table pixels within the table borders except those with ball class colors

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

        // TODO: segmentation (Leonardo)
        // ATTENTION: dataset path is specified just for testing
        // ATTENTION: if added just to avoid repetitions while testing
        if(i == 0) {
            // first and last video frame segmentation
            std::string bboxes_test_dir = "../dataset/game1_clip1/bounding_boxes";
            segmentation(video_frames, 0, bboxes_test_dir);
            segmentation(video_frames, video_frames.size()-1, bboxes_test_dir);
            
            vu::show_video_frames(video_frames);
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