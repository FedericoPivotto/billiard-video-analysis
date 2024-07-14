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
    cv::Ptr<cv::SIFT> ball_detector = cv::SIFT::create(400);
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

/* Feature extraction from the single bgr channels */
void lrds::ball_feature_extraction(cv::Mat frame, std::vector<cv::KeyPoint>& table_keypoints, cv::Mat& table_descriptors){
    // Convert to grayscale
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // SIFT ball detector
    cv::Ptr<cv::SIFT> ball_detector = cv::SIFT::create();
    ball_detector -> detectAndCompute(frame, cv::noArray(), table_keypoints, table_descriptors);
}

/* Matching the extracted features w.r.t. the ball samples */
void lrds::feature_matching(cv::Mat frame, const std::vector<cv::KeyPoint>& table_keypoints, const cv::Mat& table_descriptors){
    // Read ball template
    cv::Mat ball_solid = cv::imread("../dataset/billiard_balls/white_ball.png");
    //cv::resize(ball_solid, ball_solid, cv::Size(50, 50), 0, 0, cv::INTER_NEAREST);

    // Ball template feature extraction
    std::vector<cv::KeyPoint> ball_keypoints;
    cv::Mat ball_descriptors;
    ball_feature_extraction(ball_solid, ball_keypoints, ball_descriptors);

    // Show keypoints in ball template
    cv::drawKeypoints(ball_solid, ball_keypoints, ball_solid, cv::Scalar::all(-1));
    cv::imshow("Ball", ball_solid);

    // Ball matching
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> matches;

    matcher->knnMatch(table_descriptors, ball_descriptors, matches, 2);

    // Keep good matches
    std::vector<cv::DMatch> good_matches;
    const float ratio_th = 0.75;
    for(size_t i = 0; i < matches.size(); i++){
        if(matches[i][0].distance < ratio_th * matches[i][1].distance){
            good_matches.push_back(matches[i][0]);
        }
    }

    // Show matching
    cv::Mat img_matches;
    cv::drawMatches(frame, table_keypoints, ball_solid, ball_keypoints, good_matches, img_matches);
    cv::imshow("Ball", img_matches);
}

/* Balls detection in given a video frame */
void lrds::lrds_sift_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame) {
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

    // Feature matching w.r.t. ball templates
    feature_matching(frame, table_keypoints, table_descriptors);

    // Draw features keypoints
    //cv::Mat frame_key;
    //cv::drawKeypoints(frame_gray, table_keypoints, frame_key, cv::Scalar::all(-1));
    //cv::imshow("SIFT frame", frame_key);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/* Convert points from float to int */
void lrds::points_float_to_point(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
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
    cv::bilateralFilter(video_frame, preprocessed_video_frame, 9, 150.0, 75.0);
    
    // HSV space
    //cv::cvtColor(preprocessed_video_frame, preprocessed_video_frame, cv::COLOR_BGR2HSV);

    // Mask image to consider only the billiard table
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    std::vector<cv::Point> table_corners;
    points_float_to_point(corners, table_corners);
    cv::fillConvexPoly(mask, table_corners, cv::Scalar(255, 255, 255));


    // Filter out the background of the billiard table
    cv::Mat frame_masked;
    cv::bitwise_and(preprocessed_video_frame, mask, frame_masked);

    // Gray frame
    cv::Mat frame_gray;
    cv::cvtColor(frame_masked, frame_gray, cv::COLOR_BGR2GRAY);

    // Split in BGR channels
    std::vector<cv::Mat> bgr_channels;
    cv::split(frame_masked, bgr_channels);

    // Define ball templates
    std::vector<std::string> ball_templates_names = {"white_ball.png", "black_ball.png", "solid_ball.png", "stripe_ball.png" };
    //std::vector<std::string> ball_templates_names = {"white_ball.png" };
    std::string templates_path = "../dataset/billiard_balls/";

    // Multiple scales
    std::vector<double> scales = {0.67, 0.8, 0.9, 1.0, 1.1, 1.2};

    // Template matching of balls
    for(const std::string& ball_template_file : ball_templates_names){
        cv::Mat ball_template = cv::imread(templates_path + ball_template_file);
        //cv::bilateralFilter(ball_template.clone(), ball_template, 9, 25.0, 25.0);
        
        // Check template existence
        if(ball_template.empty()){
            continue;
        }

        // Convert to gray
        cv::Mat ball_gray;
        cv::cvtColor(ball_template, ball_gray, cv::COLOR_BGR2GRAY);

        for(const double& scale : scales){
            cv::Mat ball_scaled;
            cv::resize(ball_gray, ball_scaled, cv::Size(), scale, scale);
            
            // Matching
            cv::Mat matching;
            cv::matchTemplate(frame_gray, ball_scaled, matching, cv::TM_CCOEFF_NORMED);

            //double min_val, max_val;
            //cv::Point min_point, max_point;
            //cv::normalize(matching, matching, 0, 1, cv::NORM_MINMAX);
            //cv::minMaxLoc(matching, &min_val, &max_val, &min_point, &max_point);

            std::vector<cv::Rect> boxes;
            std::vector<double> scores;

            const double max_th = 0.9;

            for(size_t i = 0; i < matching.rows; i++){
                for(size_t j = 0; j < matching.cols; j++){
                    if(matching.at<float>(i, j) > max_th){
                        cv::Rect match_rect(j, i, ball_scaled.rows, ball_scaled.cols);
                        boxes.push_back(match_rect);
                        scores.push_back(matching.at<float>(i, j));
                    }
                }
            }

            for(const cv::Rect& match_rect : boxes){
                cv::rectangle(frame, match_rect, cv::Scalar(0, 0, 255), 2);
            }

            cv::imshow("Template", frame);
        }
    }


    // Haar cascade vs template matching

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}