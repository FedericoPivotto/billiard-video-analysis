#include <iostream>

// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// edge_detection detection library
#include <edge_detection.h>

// object detection library
#include <object_detection.h>

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
    for(size_t i = 0; i < captures.size(); ++i) {
        // read video frames
        std::vector<cv::Mat> video_frames;
        vu::read_video_frames(captures[i], video_frames);

        // create video result directory
        std::vector<std::string> video_result_subdirs;
        fsu::create_video_result_dir(video_paths[i], video_result_subdirs);
        
        // skip video if empty
        if(video_frames.empty())
            continue;

        // first and last video frame indexes if exists
        std::vector<size_t> frame_indexes = {video_frames.size()-1};
        if(video_frames.size() > 1)
            frame_indexes.push_back(0);
        // sort frame indexes
        std::sort(frame_indexes.begin(), frame_indexes.end());

        // for each video frame of interest
        std::vector<cv::Mat> video_frames_cv;
        std::vector<cv::Vec2f> first_borders;
        std::vector<cv::Point2f> first_corners;
        for(size_t j = 0; j < frame_indexes.size(); ++j) {
            // frame of interes index
            size_t k = frame_indexes[j];

            // skip video frame if empty
            if(video_frames[k].empty())
                continue;

            // video frame clone
            cv::Mat video_frame_cv = video_frames[k].clone();
            video_frames_cv.push_back(video_frame_cv);

            // check if first frame analyzed
            bool is_first = video_frames_cv.size() == 1 ? true : false;

            // edge detection (Fabrizio)
            std::vector<cv::Vec2f> borders = is_first ? first_borders : std::vector<cv::Vec2f>();
            std::vector<cv::Point2f> corners = is_first ? first_corners : std::vector<cv::Point2f>();
            ed::edge_detection(video_frame_cv, borders, corners);

            // TODO: object detection (Federico)
            od::object_detection(video_frames, k, video_result_subdirs[0], corners, video_frame_cv);

            // segmentation (Leonardo)
            sg::segmentation(video_frames, k, video_result_subdirs[0], corners, video_frame_cv);
            // draw field borders
            ed::draw_borders(video_frame_cv, borders, corners);
        }

        // assuming field corners of the first video frame
        // for each video frame
        std::vector<cv::Mat> video_game_frames_cv;
        for(size_t j = 0; j < video_frames.size(); ++j) {
            // skip game video frame if empty
            if(video_frames[j].empty())
                continue;

            // video game frame clone
            cv::Mat video_game_frame_cv = video_frames[j].clone();
            video_game_frames_cv.push_back(video_game_frame_cv);

            // TODO: 2D top-view minimap (Fabrizio)

            // TODO: trajectory tracking
        }

        // show computer vision video frames
        vu::show_video_frames(video_frames_cv);

        // show video game frames
        vu::show_video_frames(video_game_frames_cv);
    }

    return 0;
}