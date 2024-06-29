#include <iostream>

// fstream: std::ofstream
#include <fstream>

// user-defined libraries

// video_captures: video utilities
#include <video_utils.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

void object_detection(const std::vector<cv::Mat>& video_frames, int nframe, const std::string bboxes_video_path) {
    // create frame bboxes text file
    std::string bboxes_file_path = bboxes_video_path + "/frame_";
    if(nframe == 0)
        bboxes_file_path += "first";
    else if (nframe == video_frames.size()-1)
        bboxes_file_path += "last";
    else
        bboxes_file_path += std::to_string(nframe + 1);
    bboxes_file_path += ".txt";
    std::ofstream bboxes_file(bboxes_file_path);

    // identify each ball using Viola and Jones approach

    // for each bounding box, save the identified portion of the image

    // for each portion:
    // 1. make the background black
    // 2. identify the ball class:
    //  - 1:white ball - white is the predominant color
    //  - 2:black ball - black is the predominant color
    //  - 3:solid ball - color, except white and black, is the predominant color
    //  - 4:stripe ball - both white and color are predominant colors
    // 3. append a row [x, y, width, height, ball category ID] to the frame text file

    // close the frame text file
}

int main(int argc, char** argv) {
    // get videos paths
    std::vector<cv::String> video_paths;
    vu::get_video_paths(video_paths);

    // get video captures
    std::vector<cv::VideoCapture> captures;
    vu::get_video_captures(video_paths, captures);
    
    // for each video read frames
    // for(cv::VideoCapture capture : captures) {
    for(int i = 0; i < captures.size(); ++i) {
        // read video frames
        std::vector<cv::Mat> video_frames;
        vu::read_video_frames(captures[i], video_frames);

        // create video result directory
        std::vector<std::string> video_result_subdirs;
        fsu::create_video_result_dir(video_paths[i], video_result_subdirs);
        
        // TODO: object detection (Federico)

        // first video frame object detection
        object_detection(video_frames, 0, video_result_subdirs[0]);
        // last video frame object detection
        object_detection(video_frames, video_frames.size()-1, video_result_subdirs[0]);

        // TODO: edge detection (Fabrizio)
        // TODO: segmentation (Leonardo)
        // TODO: 2D top-view minimap
        // TODO: trajectory tracking

        // show video frames
        // vu::show_video_frames(video_frames);
    }

    return 0;
}

/*
Input: first video frame.
Output: frame text file (as frame_number_bbox.txt), object detection for the first video frame without the table borders.

Procedure the first frame:
1. Create in a proper directory a text file for the first frame
2. Identify each ball using Viola and Jones approach
3. For each bounding box, save the identified portion of the image
4. For each portion:
   1. Make the background black
   2. Identify the ball class:
      - 1: White ball - white is the predominant color
      - 2: Black ball - black is the predominant color
      - 3: Solid ball - color, except white and black, is the predominant color
      - 4: Stripe ball - both white and color are predominant colors 
   3. Append a row ```[x, y, width, height, ball category ID]``` to the frame text file
5. Close the frame text file

For the first video frame:
1. Open the filled frame text file for first frame, copy the first frame and superimpose on it all the identified bounding boxes
*/