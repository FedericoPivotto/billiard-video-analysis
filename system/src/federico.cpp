#include <iostream>

// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>

void object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path) {
    // create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::create_bboxes_frame_file(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // open frame bboxes text file
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);
    
    // vector of bounding boxes
    std::vector<od::Ball> ball_bboxes;

    // video frame clone
    cv::Mat frame(video_frames[n_frame].clone());

    // bgr to hsv
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    // hsv to grayscale
    cv::Mat hsv_channels[3];
    cv::split(frame_hsv, hsv_channels);
    cv::Mat frame_gs = hsv_channels[2];

    // frame preprocess
    cv::GaussianBlur(frame_gs, frame_gs, cv::Size(5, 5), 2, 2);
	// cv::medianBlur(frame_gs, frame_gs, 5);

    // hough circle transform
    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(frame_gs, circles, cv::HOUGH_GRADIENT, 1,
		5, // distance between circles
		100, 10, // canny edge detector parameters and circles center detection 
		7, 15); // min_radius & max_radius of circles to detect
    
    // show detected circles:
	for(size_t i = 0; i < circles.size(); i++) {
		// circle data
        cv::Vec3i c = circles[i];
		cv::Point center = cv::Point(c[0], c[1]);
		int radius = c[2];
        // circle ball
        ball_bboxes.push_back(od::Ball(c[0]-c[2], c[1]-c[2], c[2]*2, c[2]*2, 0));
        
		// show circle center
		cv::circle(frame, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
		// show circle outline
		cv::circle(frame, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
	}

	// show the detected circles
	cv::imshow("Detected circles", frame);
	// wait key before going ahead
	cv::waitKey(0);

    // TODO: detect ball bounding boxes using Viola and Jones approach
    // TODO: update ball vector with bounding box x, y, width, height
    // SEE: notes p.122 for extract ball image

    // scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // TODO: ball class detection
        od::detect_ball_class(ball_bbox, video_frames[n_frame]);

        // write ball bounding box in frame bboxes text file
        fsu::write_ball_bbox(bboxes_frame_file, ball_bbox);
    }

    // close frame bboxes text file
    bboxes_frame_file.close();
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
        // first and last video frame object detection
        object_detection(video_frames, 0, video_result_subdirs[0]);
        object_detection(video_frames, video_frames.size()-1, video_result_subdirs[0]);
        
        // TODO: edge detection (Fabrizio)
        // TODO: segmentation (Leonardo)
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        vu::show_video_frames(video_frames);
    }

    return 0;
}