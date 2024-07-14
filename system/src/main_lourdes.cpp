/* User-defined libraries */

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// edge_detection detection library
#include <edge_detection.h>

// object detection library
#include <lourdes.h>

// segmentation library
#include <segmentation.h>

// minimap detection library
#include <minimap.h>

/* Computer vision system main */
int main(int argc, char** argv) {
    // Get videos paths
    std::vector<cv::String> video_paths;
    vu::get_video_paths(video_paths);

    // Get video captures
    std::vector<cv::VideoCapture> captures;
    vu::get_video_captures(video_paths, captures);
    
    // For each video read frames
    for(size_t i = 0; i < captures.size(); ++i) {
        // Read video frames
        std::vector<cv::Mat> video_frames;
        vu::read_video_frames(captures[i], video_frames);

        // Create video result directory
        std::string video_result_path;
        std::vector<std::string> video_result_subdirs;
        fsu::create_video_result_dir(video_paths[i], video_result_path, video_result_subdirs);
        
        // Skip video if empty
        if(video_frames.empty())
            continue;

        // First and last video frame indexes if exists
        std::vector<size_t> frame_indexes = {video_frames.size()-1};
        if(video_frames.size() > 1)
            frame_indexes.push_back(0);
        // Sort frame indexes
        std::sort(frame_indexes.begin(), frame_indexes.end());

        // For each video frame of interest
        std::vector<cv::Mat> video_frames_cv;
        std::vector<cv::Vec2f> first_borders;
        std::vector<cv::Point2f> first_corners;
        for(size_t j = 0; j < frame_indexes.size(); ++j) {
            // Frame of interes index
            size_t k = frame_indexes[j];

            // Skip video frame if empty
            if(video_frames[k].empty())
                continue;

            // Video frame clone
            cv::Mat video_frame_cv = video_frames[k].clone();
            video_frames_cv.push_back(video_frame_cv);

            // Edge detection (Fabrizio)
            std::vector<cv::Vec2f> borders;
            std::vector<cv::Point2f> corners;
            ed::edge_detection(video_frame_cv, borders, corners);

            // Store borders and corners if first frame analyzed
            bool is_first = video_frames_cv.size() == 1 ? true : false;
            if(is_first) {
                first_borders = borders;
                first_corners = corners;
            }

            // TODO: object detection (Federico)

            // Segmentation (Leonardo)
            // Get video dataset directory
            // TODO: change with result directory
            std::vector<std::string> video_dataset_subdirs;
            fsu::get_video_dataset_dir(video_paths[i], video_dataset_subdirs);

            ed::sort_corners(first_corners);
            lrds::lrds_object_detection(video_frames, k, video_dataset_subdirs[0], first_corners, video_frame_cv);
            // TODO: when object detection is fine, the flag must be sat to false
            // ATTENTION: test_flag is used just to do test with a dataset bounding box file
            //bool test_flag = true;
            //sg::segmentation(video_frames, k, video_dataset_subdirs[0], first_corners, video_frame_cv, test_flag);
            
            // Draw field borders
            ed::draw_borders(video_frame_cv, first_borders, first_corners);
        }

        // Show computer vision video frames
        vu::show_video_frames(video_frames_cv);
    }

    return 0;
}