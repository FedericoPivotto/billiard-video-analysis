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

//-----------------------------------------------------------------------------------------------

/* Convert points from float to int */
void lrds::points_float_to_point(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
}

/* Balls detection in given a video frame */
void lrds::lrds_template_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame) {
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
    std::vector<std::string> ball_templates_names = {"white.jpg"};
    //std::vector<std::string> ball_templates_names = {"white_ball.png" };
    std::string templates_path = "../dataset/billiard_balls/";

    // Multiple scales
    std::vector<double> scales = {0.16, 0.25, 0.28};

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

            const double max_th = 0.8;

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

//-----------------------------------------------------------------------------------------

/* Callback function parameters */
typedef struct {
	// Window name
	const char* window_name;

	// Pointers to images
	cv::Mat* src;

    // HSV parameters
    int iLowH;
    int iHighH;
    int iLowS;
    int iHighS;
    int iLowV;
    int iHighV;

    // Bilateral filter
    int color_std;
    int space_std;
    int max_th;

    // Max threshold
    int maxH;
    int maxS;
    int maxV;
} ParameterHoughHSV;

/* Callback function */
static void hsv_hough_callback(int pos, void* userdata) {
    // Get Canny parameters from userdata
	ParameterHoughHSV params = *((ParameterHoughHSV*) userdata);

    // Dereference images
    cv::Mat frame = (*params.src).clone();
	cv::Mat frame_hsv, frame_bilateral, mask;
    cv::bilateralFilter(frame, frame_bilateral, 9, params.color_std, params.space_std);

    // Threshold the image in hsv
    cv::cvtColor(frame_bilateral, frame_hsv, cv::COLOR_BGR2HSV);

    //int hbins = 30, sbins = 32;
    //int histSize[] = { hbins, sbins };
    //float hranges[] = { 0, 180 };
    //float sranges[] = { 0, 256 };
    //const float* ranges[] = { hranges, sranges };
    //cv::Mat hist;
    //int channels[] = { 0, 1 };

    //cv::calcHist(&frame_hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    //cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    //double maxVal = 0;
    //int maxIdx[] = { 0, 0 };
    //minMaxIdx(hist, 0, &maxVal, 0, maxIdx);
    //int dominantHue = maxIdx[0] * (180 / hbins);
    //int dominantSaturation = maxIdx[1] * (256 / sbins); 

    //int hueRange = 150;
    //cv::inRange(frame_hsv, 
    //        cv::Scalar(dominantHue - hueRange, dominantSaturation - 150, 100),
    //        cv::Scalar(dominantHue + hueRange, dominantSaturation + 100, 255), 
    //        mask);
    cv::inRange(frame_hsv, cv::Scalar(params.iLowH, params.iLowS, params.iLowV), cv::Scalar(params.iHighH, params.iHighS, params.iHighV), mask);

    // Dilate and erosion set operations on mask 
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(mask, mask, kernel);
    //cv::erode(mask, mask, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);

    //std::vector<cv::Vec3f> circles;
	//cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
	//	10, // distance between circles
	//	100, 9, // canny edge detector parameters and circles center detection 
	//	3, 20); // min_radius & max_radius of circles to detect
    //
    //// Show detected circles
    //for(size_t i = 0; i < circles.size(); i++) {
    //    // Circle data
    //    cv::Vec3i c = circles[i];
    //    cv::Point center(c[0], c[1]);
    //    unsigned int radius = c[2];
//
    //    // Show circle center
    //    cv::circle(frame, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
    //    // Show circle outline
    //    cv::circle(frame, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    //}

    // Compute edge map of the frame by canny edge detection
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > 10) { 
            //cv::Rect bounding_rect = cv::boundingRect(contours[i]);
            //cv::rectangle(frame, bounding_rect, cv::Scalar(0, 255, 0), 2);

            // Fit an ellipse and check the ratio of the axes
            if (contours[i].size() >= 5) { 
                cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
                float aspect_ratio = (float)ellipse.size.width / (float)ellipse.size.height;
                if (aspect_ratio > 0.9 && aspect_ratio < 1.1) { 
                    cv::ellipse(frame, ellipse, cv::Scalar(0, 255, 0), 2);
                }
            }

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contours[i], center, radius);
            double circle_area = CV_PI * std::pow(radius, 2);
            if (std::abs(circle_area - area) / area < 0.2) { // Allowable area difference
                cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
            }
        }
    }


    // Display our result
	cv::imshow(params.window_name, frame);
    cv::imshow("Mask of change", mask);
}

static void mouseCallBack(int event, int x, int y, int flags, void* userdata){
    cv::Mat image;
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat* imgPointer = (cv::Mat*)userdata;
        image = *imgPointer;
        int meanB = 0;
        int meanG = 0;
        int meanR = 0;
        for(int i = x - 4; i <= x + 4; i++){
            for(int j = y - 4; j <= y + 4; j++){
                if((i <= image.cols - 1 && j <= image.rows -1) && (i >= 0 && j >= 0)){
                    meanB += (int)image.at<cv::Vec3b>(j,i)[0];
                    meanG += (int)image.at<cv::Vec3b>(j,i)[1];
                    meanR += (int)image.at<cv::Vec3b>(j,i)[2];
                }
            }
        }
        std::cout<<"MeanB: "<<(meanB / 81)<<std::endl;
        std::cout<<"MeanG: "<<(meanG / 81)<<std::endl;
        std::cout<<"MeanR: "<<(meanR / 81)<<std::endl;
    }
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
    cv::bilateralFilter(video_frame, preprocessed_video_frame, 9, 25.0, 25.0);

    // Mask image to consider only the billiard table
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    std::vector<cv::Point> table_corners;
    points_float_to_point(corners, table_corners);
    cv::fillConvexPoly(mask, table_corners, cv::Scalar(255, 255, 255));

    // Filter out the background of the billiard table
    cv::Mat frame_masked;
    cv::bitwise_and(preprocessed_video_frame, mask, frame_masked);

    // HSV parameters
    //int iLowH = 20, iHighH = 179, maxH = 179;
    //int iLowS = 15, iHighS = 160, maxS = 255;
    //int iLowV = 20, iHighV = 180, maxV = 255;
//
    //// Bilateral filter parameters
    //int max_th = 300;
    //int color_std = 100;
    //int space_std = 75;
//
    //// HSV window trackbars
    //// ParameterHSV hsvp = {"Control HSV", &frame_hsv, &mask, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, maxH, maxS, maxV};
    //ParameterHoughHSV hsvp = {"Control HSV", &frame_masked, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, color_std, space_std, max_th, maxH, maxS, maxV};
    //
    //// Control window
    //cv::namedWindow(hsvp.window_name);
//
    //// Create trackbar for Hue (0 - 179)
    //cv::createTrackbar("LowH", hsvp.window_name, &hsvp.iLowH, hsvp.maxH, hsv_hough_callback, &hsvp);
    //cv::createTrackbar("HighH", hsvp.window_name, &hsvp.iHighH, hsvp.maxH, hsv_hough_callback, &hsvp);
    //// Create trackbar for Saturation (0 - 255)
    //cv::createTrackbar("LowS", hsvp.window_name, &hsvp.iLowS, hsvp.maxS, hsv_hough_callback, &hsvp);
    //cv::createTrackbar("HighS", hsvp.window_name, &hsvp.iHighS, hsvp.maxS, hsv_hough_callback, &hsvp);
    //// Create trackbar for Value (0 - 255)
    //cv::createTrackbar("LowV", hsvp.window_name, &hsvp.iLowV, hsvp.maxV, hsv_hough_callback, &hsvp);
    //cv::createTrackbar("HighV", hsvp.window_name, &hsvp.iHighV, hsvp.maxV, hsv_hough_callback, &hsvp);
    //// Create trackbar for color and space std
    //cv::createTrackbar("Color std", hsvp.window_name, &hsvp.color_std, hsvp.max_th, hsv_hough_callback, &hsvp);
    //cv::createTrackbar("Space std", hsvp.window_name, &hsvp.space_std, hsvp.max_th, hsv_hough_callback, &hsvp);
    
    //cv::imshow("Frame", frame_hsv);

    /* Mouse callback */
    cv::imshow("HSV", frame_masked);
    cv::setMouseCallback("HSV", mouseCallBack, (void*)&frame_masked);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}

static void to_CMYK(const cv::Mat& frame_bgr, cv::Mat& frame_cmyk){
    // To RGB
    cv::Mat frame_rgb;
    cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

    // Color normalization
    frame_rgb.convertTo(frame_rgb, CV_32F, 1.0 / 255);

    // Convert to CMY
    cv::Mat frame_cmy = 1 - frame_rgb;

    // Split in channels
    std::vector<cv::Mat> channels_cmy;
    cv::split(frame_cmy, channels_cmy);

    // Compute additional channel K
    cv::Mat k_cmyk = cv::min(channels_cmy[0], cv::min(channels_cmy[1], channels_cmy[2]));

    // Compute channels and normalize
    cv::Mat c_cmyk = (channels_cmy[0] - k_cmyk)/(1 - k_cmyk + 1e-10);
    cv::Mat m_cmyk = (channels_cmy[1] - k_cmyk)/(1 - k_cmyk + 1e-10);
    cv::Mat y_cmyk = (channels_cmy[2] - k_cmyk)/(1 - k_cmyk + 1e-10);

    c_cmyk = cv::min(cv::max(c_cmyk, 0), 1);  
    m_cmyk = cv::min(cv::max(m_cmyk, 0), 1); 
    y_cmyk = cv::min(cv::max(y_cmyk, 0), 1); 
    k_cmyk = cv::min(cv::max(k_cmyk, 0), 1);

    c_cmyk.convertTo(c_cmyk, CV_8U, 255);
    m_cmyk.convertTo(m_cmyk, CV_8U, 255);
    y_cmyk.convertTo(y_cmyk, CV_8U, 255);
    k_cmyk.convertTo(k_cmyk, CV_8U, 255);

    // Merge into cmyk frame
    std::vector<cv::Mat> channels_cmyk = {c_cmyk, m_cmyk, y_cmyk, k_cmyk};
    cv::merge(channels_cmyk, frame_cmyk);
}