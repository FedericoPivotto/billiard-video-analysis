// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// edge_detection detection library
#include <edge_detection.h>

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

        // scan frames of interest
        std::vector<cv::Mat> video_frames_cv;
        for(size_t j : frame_indexes) {
            // skip frame if empty
            if(video_frames[j].empty())
                continue;

            // video frame clone
            cv::Mat video_frame_cv = video_frames[j].clone();
            video_frames_cv.push_back(video_frame_cv);

            // TODO: edge detection (Fabrizio)
            std::vector<cv::Vec2f> borders;
            std::vector<cv::Point2f> corners;
            ed::edge_detection(video_frame_cv, borders, corners);

            // TODO: object detection (Federico)
            // TODO: segmentation (Leonardo)

            // draw field borders
            ed::draw_borders(video_frame_cv, borders, corners);
            
            // TODO: 2D top-view minimap
            // TODO: trajectory tracking
        }

        // show computer vision video frames
        vu::show_video_frames(video_frames_cv);

        // show video frames
        // vu::show_video_frames(video_frames);
    }

    return 0;
}