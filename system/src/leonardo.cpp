/* User-defined libraries */

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// edge_detection detection library
#include <edge_detection.h>

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
            // Save video frame
            fsu::save_video_frame(video_frames, k, video_frame_cv, video_result_subdirs[1]);

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

            // Save edge detection frame
            cv::Mat edge_video_frame_cv = video_frames[k].clone();
            ed::draw_borders(edge_video_frame_cv, borders, corners);
            fsu::save_video_frame(video_frames, k, edge_video_frame_cv, video_result_subdirs[3]);

            // TODO: object detection (Federico)

            // Segmentation (Leonardo)
            // Get video dataset directory
            // TODO: change with result directory
            std::vector<std::string> video_dataset_subdirs;
            fsu::get_video_dataset_dir(video_paths[i], video_dataset_subdirs);

            // TODO: when object detection is fine, the flag must be sat to false
            // ATTENTION: test_flag is used just to do test with a dataset bounding box file
            bool test_flag = true;
            sg::segmentation(video_frames, k, video_dataset_subdirs[0], corners, video_frame_cv, test_flag);
            // Save segmentation
            fsu::save_video_frame(video_frames, k, video_frame_cv, video_result_subdirs[5]);

            // Segmentation mask
            cv::Mat video_frame_cv_mask = video_frames[k].clone();
            sg::segmentation_mask(video_frames, k, video_dataset_subdirs[0], corners, video_frame_cv_mask, test_flag);
            // Save segmentation mask
            fsu::save_video_frame(video_frames, k, video_frame_cv_mask, video_result_subdirs[2]);

            // Save output frame
            ed::draw_borders(video_frame_cv, borders, corners);
            fsu::save_video_frame(video_frames, k, video_frame_cv, video_result_subdirs[6]);
        }

        // Assuming field corners of the first video frame
        
        // For each video frame
        std::vector<cv::Mat> video_game_frames_cv;
        for(size_t j = 0; j < video_frames.size(); ++j) {
            // Skip game video frame if empty
            if(video_frames[j].empty())
                continue;

            // Video game frame clone
            cv::Mat video_game_frame_cv = video_frames[j].clone();
            video_game_frames_cv.push_back(video_game_frame_cv);

            // Get video dataset directory
            // TODO: change with result directory
            std::vector<std::string> video_dataset_subdirs;
            fsu::get_video_dataset_dir(video_paths[i], video_dataset_subdirs);

            // Get bounding box file path
            // TODO: change with result directory
            std::string bboxes_frame_file_path;
            fsu::get_bboxes_frame_file_path(video_frames, j, video_dataset_subdirs[0], bboxes_frame_file_path);

            // TODO: 2D top-view minimap (Fabrizio)

            // Read balls from bounding box file
            std::vector<od::Ball> ball_bboxes;
            fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);
            
            // Create map-view
            cv::Mat map_view, field_frame = video_frames[j].clone();
            mm::compute_map_view(map_view, field_frame, first_borders, first_corners, ball_bboxes);
            // Overlay the map-view in the current frame
            mm::overlay_map_view(video_game_frame_cv, map_view);

            // TODO: trajectory tracking
            // NOTE: required to update minimap
        }
        
        // Show computer vision video frames
        // vu::show_video_frames(video_frames_cv);

        // Show video game frames
        // vu::show_video_frames(video_game_frames_cv);

        // Game video filename
        std::string result_video_name = std::filesystem::path(video_paths[i]).parent_path().filename();
        std::string result_video_path = video_result_path + "/" + result_video_name + ".mp4";
        // Video parameters
        double fps = captures[i].get(cv::CAP_PROP_FPS);
        int width  = captures[i].get(cv::CAP_PROP_FRAME_WIDTH);
        int height = captures[i].get(cv::CAP_PROP_FRAME_HEIGHT);
        // Create and save video
        vu::save_video(video_game_frames_cv, fps, width, height, result_video_path);
    }

    return 0;
}