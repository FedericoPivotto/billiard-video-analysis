#include <lourdes.h>

/* Librarires required in this source file and not already included in lourdes.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>


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

    // Corners
    std::vector<cv::Point2f> corners;

    // Distortion
    bool is_distorted;
} ParameterHoughHSV;


/* Convert points from float to int */
void lrds::points_float_to_point(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
}

/* Preprocessing of frame in bgr */
void lrds::preprocess_bgr_frame(const cv::Mat& frame, cv::Mat& preprocessed_video_frame){
    //// Apply median to slightly remove noise
    //cv::medianBlur(frame, preprocessed_video_frame, 3);
    //
    //// Frame sharpening 
    //cv::Mat gaussian_frame;
    //cv::GaussianBlur(preprocessed_video_frame, gaussian_frame, cv::Size(3,3), 1.0);
    //cv::addWeighted(preprocessed_video_frame, 1.5, gaussian_frame, -0.5, 0, preprocessed_video_frame);
    
    // Keep color information 
    cv::bilateralFilter(frame.clone(), preprocessed_video_frame, 9, 75.0, 50.0);
}

/* Suppress circles too close to billiard holes */
void lrds::suppress_billiard_holes(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f> corners, const bool is_distorted){
    // The ratio between billiard hole and a short border the table 
    const double ratio_hole_border = 0.08;
    std::vector<cv::Vec3f> circles_filtered;

    // Check if billiard table is distorted
    if(!is_distorted){
        // Compute short border length 
        const int border_length = cv::norm(corners[0] - corners[3]);
        const int ray = border_length * ratio_hole_border;

        // Compute holes positions along long borders
        const float cx_one = (corners[0].x + corners[1].x) / 2, cx_two = (corners[2].x + corners[3].x) / 2;
        const float cy_one = (corners[0].y + corners[1].y) / 2, cy_two = (corners[2].y + corners[3].y) / 2;
        std::vector<cv::Point2f> mid_holes = {cv::Point2f(cx_one, cy_one), cv::Point2f(cx_two, cy_two)};
        
        // Check if hough circle is not too close to a billiard hole
        for(size_t i = 0; i < circles.size(); i++){
            // Checking circle closeness to corner holes
            bool is_close = false;
            for(size_t j = 0; j < corners.size(); j++){
                if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray){
                    is_close = true;
                }
            }

            // Check holes in the long borders
            for(size_t j = 0; j < 2; j++){
                if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray){
                    is_close = true;
                }
            }

            // Check if circle not too close to a billiard hole
            if(!is_close){
                circles_filtered.push_back(circles[i]);
            }
        }

    } else {
        // Compute short borders lengths 
        const int border_length_one = cv::norm(corners[0] - corners[1]), border_length_two = cv::norm(corners[2] - corners[3]);
        const int ray_one = border_length_one * ratio_hole_border, ray_two = border_length_two * ratio_hole_border;

        // Check holes in the long borders
        const double border_length_ratio = static_cast<double>(border_length_one) / border_length_two;
        double interpolation_weight = (border_length_ratio >= 0.70) ? 0.40 : (border_length_ratio >= 0.55) ? 0.38 : 0.3;

        std::vector<cv::Point2f> mid_holes = {corners[1] + interpolation_weight * (corners[2] - corners[1]), corners[0] + interpolation_weight * (corners[3] - corners[0])};

        // Check if hough circle is not too close to a billiard hole
        for(size_t i = 0; i < circles.size(); i++){
            // Checking circle closeness to holes
            bool is_close = false;
            for(size_t j = 0; j < corners.size(); j++){
                if(j <= 1){
                    if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray_one){
                        is_close = true;
                    }
                } else {
                    if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray_two){
                        is_close = true;
                    }
                }
            }

            // Check holes in the long borders
            for(size_t j = 0; j < 2; j++){
                if(j <= 1){
                    if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray_one){
                        is_close = true;
                    }
                } else {
                    if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= ray_two){
                        is_close = true;
                    }
                }
            }

            // Check if circle not too close to a billiard hole
            if(!is_close){
                circles_filtered.push_back(circles[i]);
            }
        }
    }

    circles = circles_filtered;
}

/* Trackbars for hough circles */
static void hsv_hough_callback(int pos, void* userdata) {
    // Get Canny parameters from userdata
	ParameterHoughHSV params = *((ParameterHoughHSV*) userdata);

    // Dereference images
    cv::Mat frame = (*params.src).clone();
	cv::Mat mask;

    // Color hsv segmentation
    cv::inRange(frame, cv::Scalar(params.iLowH, params.iLowS, params.iLowV), cv::Scalar(params.iHighH, params.iHighS, params.iHighV), mask);

    // Dilate and erosion set operations on mask 
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    //cv::dilate(mask, mask, kernel);
    //cv::erode(mask, mask, kernel);
    //cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);
    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);

    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
		6, // distance between circles
		100, 9, // canny edge detector parameters and circles center detection 
		3, 20); // min_radius & max_radius of circles to detect
    
    lrds::suppress_billiard_holes(circles, params.corners, params.is_distorted);
    
    // Show detected circles
    for(size_t i = 0; i < circles.size(); i++) {
        // Circle data
        cv::Vec3i c = circles[i];
        cv::Point center(c[0], c[1]);
        unsigned int radius = c[2];

        // Show circle center
        cv::circle(frame, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        // Show circle outline
        cv::circle(frame, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    }

    // Display our result
	cv::imshow(params.window_name, frame);
    cv::imshow("Mask of change", mask);
}

void lrds::lrds_object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame, const bool is_distorted) {
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::get_bboxes_frame_file_path(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Vector of bounding boxes
    std::vector<od::Ball> ball_bboxes;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);

    // Video frame clone
    cv::Mat frame(video_frames[n_frame].clone());

    // Mask image to consider only the billiard table
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    std::vector<cv::Point> table_corners;
    points_float_to_point(corners, table_corners);
    cv::fillConvexPoly(mask, table_corners, cv::Scalar(255, 255, 255));

    // Filter out the background of the billiard table
    cv::Mat frame_masked;
    cv::bitwise_and(video_frame, mask, frame_masked);

    // Masked frame preprocessing 
    cv::Mat preprocessed_video_frame;
    preprocess_bgr_frame(frame_masked, preprocessed_video_frame);

    // Convert to HSV color space
    cv::Mat frame_hsv;
    cv::cvtColor(preprocessed_video_frame, frame_hsv, cv::COLOR_BGR2HSV);

    // HSV parameters
    int iLowH = 60, iHighH = 120, maxH = 179;
    int iLowS = 150, iHighS = 255, maxS = 255;
    int iLowV = 115, iHighV = 255, maxV = 255;

    // Bilateral filter parameters
    int max_th = 300;
    int color_std = 100;
    int space_std = 75;

    // HSV window trackbars
    // ParameterHSV hsvp = {"Control HSV", &frame_hsv, &mask, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, maxH, maxS, maxV};
    ParameterHoughHSV hsvp = {"Control HSV", &frame_hsv, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, color_std, space_std, max_th, maxH, maxS, maxV, corners, is_distorted};
    
    // Control window
    cv::namedWindow(hsvp.window_name);

    // Create trackbar for Hue (0 - 179)
    cv::createTrackbar("LowH", hsvp.window_name, &hsvp.iLowH, hsvp.maxH, hsv_hough_callback, &hsvp);
    cv::createTrackbar("HighH", hsvp.window_name, &hsvp.iHighH, hsvp.maxH, hsv_hough_callback, &hsvp);
    // Create trackbar for Saturation (0 - 255)
    cv::createTrackbar("LowS", hsvp.window_name, &hsvp.iLowS, hsvp.maxS, hsv_hough_callback, &hsvp);
    cv::createTrackbar("HighS", hsvp.window_name, &hsvp.iHighS, hsvp.maxS, hsv_hough_callback, &hsvp);
    // Create trackbar for Value (0 - 255)
    cv::createTrackbar("LowV", hsvp.window_name, &hsvp.iLowV, hsvp.maxV, hsv_hough_callback, &hsvp);
    cv::createTrackbar("HighV", hsvp.window_name, &hsvp.iHighV, hsvp.maxV, hsv_hough_callback, &hsvp);
    // Create trackbar for color and space std
    cv::createTrackbar("Color std", hsvp.window_name, &hsvp.color_std, hsvp.max_th, hsv_hough_callback, &hsvp);
    cv::createTrackbar("Space std", hsvp.window_name, &hsvp.space_std, hsvp.max_th, hsv_hough_callback, &hsvp);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
}






//-------------------------------------------------------

    // Compute edge map of the frame by canny edge detection
    //std::vector<std::vector<cv::Point>> contours;
    //std::vector<cv::Vec4i> hierarchy;
    //cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
//
    //for (size_t i = 0; i < contours.size(); i++) {
    //    double area = cv::contourArea(contours[i]);
    //    if (area > 10) { 
    //        //cv::Rect bounding_rect = cv::boundingRect(contours[i]);
    //        //cv::rectangle(frame, bounding_rect, cv::Scalar(0, 255, 0), 2);
//
    //        // Fit an ellipse and check the ratio of the axes
    //        if (contours[i].size() >= 5) { 
    //            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
    //            float aspect_ratio = (float)ellipse.size.width / (float)ellipse.size.height;
    //            if (aspect_ratio > 0.9 && aspect_ratio < 1.1) { 
    //                cv::ellipse(frame, ellipse, cv::Scalar(0, 255, 0), 2);
    //            }
    //        }
//
    //        cv::Point2f center;
    //        float radius;
    //        cv::minEnclosingCircle(contours[i], center, radius);
    //        double circle_area = CV_PI * std::pow(radius, 2);
    //        if (std::abs(circle_area - area) / area < 0.2) { // Allowable area difference
    //            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
    //        }
    //    }
    //}
//