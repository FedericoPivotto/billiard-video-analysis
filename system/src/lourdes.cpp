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

/* Suppress too much close circles */
void lrds::suppress_close_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big){
    const float min_distance = 15.0;
    std::vector<cv::Vec3f> circles_filtered;
    std::vector<bool> visited(circles.size(), false);

    // Suppress close circles and keep big circles
    for(size_t i = 0; i < circles.size(); i++){
        int count = 1;
        float sum_radius = circles[i][2];
        cv::Point2f center_i(circles[i][0], circles[i][1]);

        if(!visited[i] && circles[i][2] <= 14.0){
            visited[i] = true;

            for(size_t j = i + 1; j < circles.size(); j++){
                if(!visited[j]){    
                    // Compute distance between centers of circle i and circle j
                    cv::Point2f center_j(circles[j][0], circles[j][1]);
                    float distance = cv::norm(center_i - center_j);

                    if(distance <= min_distance){
                        count++;
                        sum_radius += circles[j][2];
                        visited[j] = true;
                    }
                }
            }

            // Compute circle to represent the ball
            float avg_radius = sum_radius / count;
            cv::Vec3f circle_i(center_i.x, center_i.y, avg_radius);
            circles_filtered.push_back(circle_i);

        } else if(!visited[i] && circles[i][2] >= 15.0){
            visited[i] = true;
            circles_big.push_back(circles[i]);
        }
    }

    circles = circles_filtered;
}

/* Suppress too much small circles */
void lrds::suppress_small_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_small){
    const float radius_min = 5.0;
    std::vector<cv::Vec3f> circles_filtered;
    
    for(size_t i = 0; i < circles.size(); i++){
        if(circles[i][2] >= radius_min){
            circles_filtered.push_back(circles[i]);
        } else {
            circles_small.push_back(circles[i]);
        }
    }

    circles = circles_filtered;
}

/* Normalize too much small or large circles */
void lrds::normalize_circles_radius(std::vector<cv::Vec3f>& circles){
    float radius_sum = 0.0, radius_avg = 0.0;
    std::vector<cv::Vec3f> circles_filtered;
    
    // Compute average radius only on not large circles
    for(size_t i = 0; i < circles.size(); i++){
        radius_sum += circles[i][2];
    }

    radius_avg = radius_sum / circles.size();

    // Resize small circles
    for(size_t i = 0; i < circles.size(); i++){
        if(circles[i][2] <= 10.0){
            circles[i][2] = radius_avg;
        }
    }
}

/*
void lrds::select_circles(std::vector<cv::Vec4f>& circles){
    const float votes_th = 10.0; 
    std::vector<cv::Vec4f> circles_filtered;
    
    // Compute average radius only on not large circles
    for(size_t i = 0; i < circles.size(); i++){
        if(circles[i][3] > votes_th){
            circles_filtered.push_back(circles[i]);
        }
    }

    circles = circles_filtered;
}*/

/* Perform color segmentation based on dominant hue and saturation */
void lrds::find_dominant_colors(const cv::Mat& frame, const cv::Mat& mask, int& dominant_hue, int& dominant_saturation){
    // Settings of hsv histogram
    int hbins = 30, sbins = 32;
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};

    const int hist_size[] = {hbins, sbins};
    const float* hist_ranges[] = {hranges, sranges};
    int channels[] = {0, 1};
    
    // Compute hsv histogram only on H and S channels
    cv::Mat histogram;
    cv::calcHist(&frame, 1, channels, mask, histogram, 2, hist_size, hist_ranges, true, false);
    cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX);

    // Find dominant hue and saturation by the max value of the histogram
    double max_hist = 0.0; 
    int max_index[] = {0, 0};
    cv::minMaxIdx(histogram, 0, &max_hist, 0, max_index);

    dominant_hue = max_index[0] * (180 / hbins);
    dominant_saturation = max_index[1] * (256 / sbins);
}

/* Trackbars for hough circles */
static void hsv_hough_callback(int pos, void* userdata) {
    // Get Canny parameters from userdata
	ParameterHoughHSV params = *((ParameterHoughHSV*) userdata);

    // Dereference images
    cv::Mat frame = (*params.src).clone();
	cv::Mat mask, hand_mask, glove_mask, frame_bgr;

    // Compute bgr frame
    cv::cvtColor(frame, frame_bgr, cv::COLOR_HSV2BGR);

    // Color hsv segmentation
    //int hue_range = 50;
    //int dominant_hue = 0, dominant_saturation = 0;
    //std::vector<cv::Point> table_corners;
    //cv::Mat hist_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    //
    //lrds::points_float_to_point(params.corners, table_corners);
    //cv::fillConvexPoly(hist_mask, table_corners, cv::Scalar(255));
    //
    //lrds::find_dominant_colors(frame, hist_mask, dominant_hue, dominant_saturation);
    //cv::inRange(frame, cv::Scalar(dominant_hue - params.iLowH, dominant_saturation - params.iLowS, params.iLowV), 
    //                   cv::Scalar(dominant_hue + params.iHighH, dominant_saturation + params.iHighS, params.iHighV), mask);
    cv::inRange(frame, cv::Scalar(params.iLowH, params.iLowS, params.iLowV), cv::Scalar(params.iHighH, params.iHighS, params.iHighV), mask);
    cv::inRange(frame_bgr, cv::Scalar(110, 130, 110), cv::Scalar(140, 185, 215), hand_mask);
    cv::inRange(frame_bgr, cv::Scalar(5, 5, 5), cv::Scalar(50, 50, 50), glove_mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::erode(hand_mask, hand_mask, kernel);
    cv::dilate(hand_mask, hand_mask, kernel);
    cv::erode(glove_mask, glove_mask, kernel);
    cv::dilate(glove_mask, glove_mask, kernel);
    

    // Dilate and erosion set operations on mask 
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    //cv::erode(mask, mask, kernel);
    //cv::dilate(mask, mask, kernel);
    //cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);
    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::MORPH_ELLIPSE, cv::Point(-1, -1), 5);

    // Perform canny on mask
    cv::Mat edge_map;
    double upper_th = 100.0;
    double lower_th = 10.0;
    cv::Canny(mask, edge_map, lower_th, upper_th);
    //cv::dilate(edge_map, edge_map, kernel);
    //cv::erode(edge_map, edge_map, kernel);

    std::vector<cv::Vec3f> circles, circles_big, circles_small;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
		6, // distance between circles
		100, 9, // canny edge detector parameters and circles center detection 
		3, 20); // min_radius & max_radius of circles to detect
    
    // Suppress circles
    lrds::suppress_billiard_holes(circles, params.corners, params.is_distorted);
    lrds::suppress_close_circles(circles, circles_big);
    lrds::suppress_small_circles(circles, circles_small);
    lrds::normalize_circles_radius(circles);
    
    // Get circles
    circles.insert(circles.end(), circles_big.begin(), circles_big.end());
    
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
    cv::imshow("Mask of change", glove_mask);
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
    /*FULL int iLowH = 60, iHighH = 255, maxH = 255;
    int iLowS = 150, iHighS = 255, maxS = 255;
    int iLowV = 115, iHighV = 255, maxV = 255;*/
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