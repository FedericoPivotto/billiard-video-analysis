// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

// edge_detection detection library
#include <edge_detection.h>

// minimap detection library
#include <minimap.h>

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
        
        // Compute map view of the billiard table-----------------------
        cv::Mat map_view;
        ed::sort_corners(corners);   
        mm::compute_map_view(map_view, first_frame, borders, corners);

        // Overlay the map-view in the current frame
        mm::overlay_map_view(first_frame, map_view);

        // Compute map view of the billiard table-----------------------

        // Show frame with borders
        cv::namedWindow("Billiard video frame");
        cv::imshow("Billiard video frame", first_frame);

        cv::waitKey(0);

        // TODO: segmentation (Leonardo) ------------------------------------------------------------------
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        //show_video_frames(video_frames);
    }

    return 0;
}