#include <lourdes.h>

/* Librarires required in this source file and not already included in lourdes.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>


/* Balls detection in given a video frame */
void lrds::lrds_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame) {
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Vector of bounding boxes
    std::vector<od::Ball> ball_bboxes;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);

    // Video frame clone
    cv::Mat frame(video_frames[n_frame].clone());
    
    // Gray frame
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}