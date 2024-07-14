#include <lourdes.h>

/* Librarires required in this source file and not already included in lourdes.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>
// SIFT
#include <opencv2/features2d.hpp>

/* Feature extraction from the single bgr channels */
void lrds::frame_feature_extraction(cv::Mat frame, const std::vector<cv::Point2f>& corners, std::vector<cv::KeyPoint>& table_keypoints, cv::Mat& table_descriptors){
    // Keypoints and descriptors
    std::vector<cv::KeyPoint> frame_keypoints;
    cv::Mat frame_descriptors;
    
    // SIFT ball detector
    cv::Ptr<cv::SIFT> ball_detector = cv::SIFT::create();
    ball_detector -> detectAndCompute(frame, cv::noArray(), frame_keypoints, frame_descriptors);
    
    // Filter features outside the table
    for(size_t i = 0; i < frame_keypoints.size(); i++){
        cv::KeyPoint kp = frame_keypoints[i];
        cv::Point2f point = kp.pt;

        if(cv::pointPolygonTest(corners, point, false) == 1.0){
            table_keypoints.push_back(kp);
            table_descriptors.push_back(frame_descriptors.row(i));
        }
    }
}

/* Matching the extracted features w.r.t. the ball samples */
void lrds::feature_matching(cv::Mat frame, const std::vector<cv::KeyPoint>& table_keypoints, const cv::Mat& table_descriptors){

}

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

    // Frame preprocessing 
    cv::Mat preprocessed_video_frame;
    cv::bilateralFilter(video_frame, preprocessed_video_frame, 9, 50.0, 25.0);

    // Gray frame
    cv::Mat frame_gray;
    cv::cvtColor(preprocessed_video_frame, frame_gray, cv::COLOR_BGR2GRAY);
    
    // HSV space
    //cv::cvtColor(preprocessed_video_frame, preprocessed_video_frame, cv::COLOR_BGR2HSV);

    // Split in BGR channels
    std::vector<cv::Mat> bgr_channels;
    cv::split(preprocessed_video_frame, bgr_channels);

    // SIFT feature extraction
    std::vector<cv::KeyPoint> table_keypoints;
    cv::Mat table_descriptors;
    
    lrds::frame_feature_extraction(bgr_channels[0], corners, table_keypoints, table_descriptors);
    lrds::frame_feature_extraction(bgr_channels[1], corners, table_keypoints, table_descriptors);
    lrds::frame_feature_extraction(bgr_channels[2], corners, table_keypoints, table_descriptors);

    // Draw features keypoints
    cv::Mat frame_key;
    cv::drawKeypoints(frame_gray, table_keypoints, frame_key, cv::Scalar::all(-1));
    cv::imshow("SIFT frame", frame_key);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}