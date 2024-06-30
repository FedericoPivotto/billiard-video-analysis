// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

int main(int argc, char** argv) {
    // get videos paths
    std::vector<cv::String> video_paths;
    vu::get_video_paths(video_paths);

    // get video captures
    std::vector<cv::VideoCapture> captures;
    vu::get_video_captures(video_paths, captures);
    
    // for each video read frames
    for(cv::VideoCapture capture : captures) {
        // read video frames
        std::vector<cv::Mat> video_frames;
        vu::read_video_frames(capture, video_frames);
        
        // TODO: object detection (Federico)
        // TODO: edge detection (Fabrizio)
        // TODO: segmentation (Leonardo)
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        vu::show_video_frames(video_frames);
    }

    return 0;
}