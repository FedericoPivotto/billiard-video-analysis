#include <object_detection.h>

/* Librarires required in this source file and not already included in object_detection.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>

/* Static id definition */
int od::Ball::current_id = 0;

/* Ball empty constructor */
od::Ball::Ball() : id(++current_id), x(0), y(0), width(0), height(0), ball_class(0), confidence(0) {
}

/* Ball full constructor */
od::Ball::Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class, double confidence) : id(++current_id), x(x), y(y), width(width), height(height), ball_class(ball_class), confidence(confidence) {
}

/* Get ball center */
std::pair<unsigned int, unsigned int> od::Ball::center() const {
    // Compute ball center coordinates
    return {x + width / 2, y + height / 2};
}

/* Get ball radius */
unsigned int od::Ball::radius() const {
    // Compute ball radius
    return width < height ? width / 2 : height / 2;
}

cv::Rect od::Ball::get_rect_bbox() {
    // Create bounding box cv::Rect
    return cv::Rect(x, y, width, height);
}

void od::Ball::set_rect_bbox(cv::Rect bbox) {
    x = bbox.x;
    y = bbox.y;
    width = bbox.width;
    height = bbox.height;
}

/* Ball operator << overload */
std::ostream& od::operator<<(std::ostream& os, const Ball& ball) {
    // Ball information string
    return os << ball.x << " " << ball.y << " " << ball.width << " " << ball.height << " " << ball.ball_class;
}

/* Ball operator == overload */
bool od::operator==(const Ball& ball1, const Ball& ball2) {
    // Ball comparison
    return  ball1.x == ball2.x && 
            ball1.y == ball2.y && 
            ball1.width == ball2.width &&
            ball1.height == ball2.height && 
            ball1.ball_class == ball2.ball_class &&
            ball1.confidence == ball2.confidence;
}

/* Convert points from float to int */
void od::points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& points) {
    // Convert points from float to int
    for (cv::Point2f point : float_points)
        points.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
}

/* Preprocessing of frame in bgr */
void od::preprocess_bgr_frame(const cv::Mat& frame, cv::Mat& preprocessed_video_frame) {
    // Apply median to slightly remove noise
    cv::medianBlur(frame, preprocessed_video_frame, 3);

    cv::Mat gaussian_frame;
    cv::GaussianBlur(preprocessed_video_frame, gaussian_frame, cv::Size(3,3), 1.0);
    cv::addWeighted(preprocessed_video_frame, 1.5, gaussian_frame, -0.4, 0, preprocessed_video_frame);
    
    // Keep color information 
    cv::bilateralFilter(preprocessed_video_frame.clone(), preprocessed_video_frame, 9, 125.0, 50.0);
}

/* Morphological operations on mask */
void od::morpho_pre_process(cv::Mat& mask) {
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(mask, mask, kernel1);

    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(mask, mask, kernel2);

    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(mask, mask, kernel3);

    cv::Mat kernel4 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(mask, mask, kernel4);

    cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(mask, mask, kernel5);
}

/* Suppress circles too close to billiard holes */
void od::suppress_billiard_holes(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f> corners, const bool is_distorted) {
    // The ratio between billiard hole and a short border the table 
    const double ratio_hole_border = 0.08;
    std::vector<cv::Vec3f> circles_filtered;

    // Check if billiard table is distorted
    if(!is_distorted) {
        // Compute short border length 
        const int border_length = cv::norm(corners[0] - corners[3]);
        const int radius = border_length * ratio_hole_border;

        // Compute holes positions along long borders
        const float cx_one = (corners[0].x + corners[1].x) / 2, cx_two = (corners[2].x + corners[3].x) / 2;
        const float cy_one = (corners[0].y + corners[1].y) / 2, cy_two = (corners[2].y + corners[3].y) / 2;
        std::vector<cv::Point2f> mid_holes = {cv::Point2f(cx_one, cy_one), cv::Point2f(cx_two, cy_two)};
        
        // Check if hough circle is not too close to a billiard hole
        for(size_t i = 0; i < circles.size(); i++) {
            // Checking circle closeness to corner holes
            bool is_close = false;
            for(size_t j = 0; j < corners.size(); j++) {
                if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius)
                    is_close = true;
            }

            // Check holes in the long borders
            for(size_t j = 0; j < 2; j++) {
                if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius) {
                    is_close = true;
                }
            }

            // Check if circle not too close to a billiard hole
            if(!is_close) {
                circles_filtered.push_back(circles[i]);
            }
        }

    } else {
        // Compute short borders lengths 
        const int border_length_one = cv::norm(corners[0] - corners[1]), border_length_two = cv::norm(corners[2] - corners[3]);
        const int radius_one = border_length_one * ratio_hole_border, radius_two = border_length_two * ratio_hole_border;

        // Check holes in the long borders
        const double border_length_ratio = static_cast<double>(border_length_one) / border_length_two;
        double interpolation_weight = (border_length_ratio >= 0.70) ? 0.40 : (border_length_ratio >= 0.55) ? 0.38 : 0.3;

        std::vector<cv::Point2f> mid_holes = {corners[1] + interpolation_weight * (corners[2] - corners[1]), corners[0] + interpolation_weight * (corners[3] - corners[0])};

        // Check if hough circle is not too close to a billiard hole
        for(size_t i = 0; i < circles.size(); i++) {
            // Checking circle closeness to holes
            bool is_close = false;
            for(size_t j = 0; j < corners.size(); j++) {
                if(j <= 1) {
                    if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius_one)
                        is_close = true;
                } else {
                    if(cv::norm(corners[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius_two)
                        is_close = true;
                }
            }

            // Check holes in the long borders
            for(size_t j = 0; j < 2; j++) {
                if(j <= 1){
                    if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius_one)
                        is_close = true;
                } else {
                    if(cv::norm(mid_holes[j] - cv::Point2f(circles[i][0], circles[i][1])) <= radius_two)
                        is_close = true;
                }
            }

            // Check if circle not too close to a billiard hole
            if(!is_close)
                circles_filtered.push_back(circles[i]);
        }
    }

    // Update circles
    circles = circles_filtered;
}

/* Suppress too much close circles */
void od::suppress_close_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big) {
    const float min_distance = 15.0;
    std::vector<cv::Vec3f> circles_filtered;
    std::vector<bool> visited(circles.size(), false);

    // Suppress close circles and keep big circles
    for(size_t i = 0; i < circles.size(); i++) {
        int count = 1;
        float sum_radius = circles[i][2];
        cv::Point2f center_i(circles[i][0], circles[i][1]);

        if(!visited[i] && circles[i][2] <= 14.0) {
            visited[i] = true;

            for(size_t j = i + 1; j < circles.size(); j++) {
                if(!visited[j]) {    
                    // Compute distance between centers of circle i and circle j
                    cv::Point2f center_j(circles[j][0], circles[j][1]);
                    float distance = cv::norm(center_i - center_j);

                    if(distance <= min_distance) {
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

        } else if(!visited[i] && circles[i][2] >= 15.0) {
            visited[i] = true;
            circles_big.push_back(circles[i]);
        }
    }

    // Update circles
    circles = circles_filtered;
}

/* Suppress too much small circles */
void od::suppress_small_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_small) {
    const float radius_min = 5.0;
    std::vector<cv::Vec3f> circles_filtered;
    
    for(size_t i = 0; i < circles.size(); i++){
        if(circles[i][2] >= radius_min){
            circles_filtered.push_back(circles[i]);
        } else {
            circles_small.push_back(circles[i]);
        }
    }

    // Update circles
    circles = circles_filtered;
}

/* Suppress circles with black center */
void od::suppress_black_circles(std::vector<cv::Vec3f>& circles, cv::Mat mask) {
    // Suppress circles with black center
    std::vector<cv::Vec3f> circles_filtered;

    for(size_t i = 0; i < circles.size(); i++) {
        unsigned char pixel = mask.at<unsigned char>(circles[i][1], circles[i][0]);
        if(pixel != 0)
            circles_filtered.push_back(circles[i]);
    }

    // Update circles
    circles = circles_filtered;
}

/* Normalize too much small or large circles */
void od::normalize_circles_radius(std::vector<cv::Vec3f>& circles) {
    float radius_sum = 0.0, radius_avg = 0.0;
    std::vector<cv::Vec3f> circles_filtered;
    
    // Compute average radius only on not large circles
    for(size_t i = 0; i < circles.size(); i++)
        radius_sum += circles[i][2];

    radius_avg = radius_sum / circles.size();

    // Resize small circles
    for(size_t i = 0; i < circles.size(); i++) {
        if(circles[i][2] <= 15.0)
            circles[i][2] = radius_avg;
    }
}

/* Balls detection in given a video frame */
void od::object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, const bool is_distorted, cv::Mat& video_frame) {

    // Mask image to consider only the billiard table
    cv::Mat mask = cv::Mat::zeros(video_frames[n_frame].size(), CV_8UC3);
    std::vector<cv::Point> table_corners;
    od::points_float_to_int(corners, table_corners);
    cv::fillConvexPoly(mask, table_corners, cv::Scalar(255, 255, 255));

    // Filter out the background of the billiard table
    cv::Mat frame_masked;
    cv::bitwise_and(video_frames[n_frame], mask, frame_masked);

    // Masked frame preprocessing 
    cv::Mat preprocessed_video_frame;
    od::preprocess_bgr_frame(frame_masked, preprocessed_video_frame);

    // Convert to HSV color space
    cv::Mat frame_hsv;
    cv::cvtColor(preprocessed_video_frame, frame_hsv, cv::COLOR_BGR2HSV);

    // Mask on billiard balls
    cv::Mat mask_balls;
    cv::inRange(frame_hsv, cv::Scalar(60, 150, 115), cv::Scalar(120, 255, 255), mask_balls);
    cv::bitwise_not(mask_balls, mask_balls);
    cv::Mat mask_balls_roi = cv::Mat::zeros(frame_hsv.size(), CV_8UC1);
    cv::fillConvexPoly(mask_balls_roi, table_corners, cv::Scalar(255, 255, 255));
    cv::bitwise_and(mask_balls, mask_balls_roi, mask_balls);

    // Morphological operations on mask
    od::morpho_pre_process(mask_balls);

    // Circle Hough transform
    std::vector<cv::Vec3f> circles, circles_big, circles_small;
	cv::HoughCircles(mask_balls, circles, cv::HOUGH_GRADIENT, 1,
		7,      // Distance between circles
		100, 9, // Canny edge detector parameters and circles center detection 
		3, 22); // Min-radius and max-radius of circles to detect

    // Suppress circles
    od::suppress_billiard_holes(circles, corners, is_distorted);
    od::suppress_close_circles(circles, circles_big);
    od::suppress_small_circles(circles, circles_small);
    od::suppress_black_circles(circles, mask_balls);
    od::normalize_circles_radius(circles);

    // Get circles
    // TODO: check threholds radius big
    circles.insert(circles.end(), circles_big.begin(), circles_big.end());
    circles.insert(circles.end(), circles_small.begin(), circles_small.end());
    
    // Show detected circles
    // TODO: to be removed
    /*cv::Mat frame = video_frames[n_frame].clone();
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
    // TODO: to be removed
	cv::imshow("Frame with detection", frame);
    cv::imshow("Mask of change", mask_balls);
    cv::waitKey(0);*/

    // Ball bounding boxes from circles
    // TODO: review which circles
    std::vector<od::Ball> ball_bboxes;
    for(cv::Vec3f circle : circles) {
        // Circle data
        cv::Point center(circle[0], circle[1]);
        unsigned int radius = circle[2];
        // Ball bounding box
        od::Ball ball_bbox(center.x - radius, center.y - radius, 2*radius, 2*radius);
        ball_bboxes.push_back(ball_bbox);
    }
    
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::create_bboxes_frame_file(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // Compute magnitude on grayscale frame
    // TODO: to review
    // cv::Mat frame_gray, magnitude;
    // cv::cvtColor(preprocessed_video_frame, frame_gray, cv::COLOR_BGR2GRAY);
    // od::compute_gradient_magnitude(frame_gray, magnitude);

    // Scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // TODO: Ball class detection
        od::detect_ball_class(ball_bbox, video_frames[n_frame]);

        // TODO: Compute confidence value
        od::set_ball_bbox_confidence(ball_bbox);

        // Write Ball bounding box in frame bboxes text file
        fsu::write_ball_bbox(bboxes_frame_file, ball_bbox);
    }

    // Close frame bboxes text file
    bboxes_frame_file.close();
}

/* Ball class detection */
void od::detect_ball_class(Ball& ball_bbox, cv::Mat frame) {
    // TODO: remove background

    // TODO: detect ball class
    // - 1:white ball - white is the predominant color
    // - 2:black ball - black is the predominant color
    // - 3:solid ball - color, except white and black, is the predominant color
    // - 4:stripe ball - both white and color are predominant colors

    // Get bounding box center and radius
    cv::Point box_center = cv::Point(ball_bbox.x, ball_bbox.y);
    unsigned int radius = ball_bbox.radius();

    // Get ball mask
    cv::Mat mask_ball = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::circle(mask_ball, box_center, radius, cv::Scalar(255, 255, 255), cv::FILLED);

    // Get ball region
    // TODO: issue to fix for having circular ball mask
    cv::Mat frame_roi = frame(mask_ball);

    // Get grayscale frame
    cv::Mat frame_gray;
    cv::cvtColor(frame_roi, frame_gray, cv::COLOR_BGR2GRAY);

    // Compute ball gradient magnitude    
    cv::Mat magnitude;
    od::compute_gradient_magnitude(frame_gray, magnitude);
    double grad_score = cv::countNonZero(magnitude);

    // Compute ball color ratio w.r.t. white
    double ratio;
    od::compute_color_white_ratio(frame_roi, ratio);

    // Classify according to ratio and gradient magnitude
    double ball_score = ratio + 0.5 * grad_score;

    // TODO: to modify
    // Set ball class
    ball_bbox.ball_class = -2;
}

// TODO: define ball bbox confidence
void od::set_ball_bbox_confidence(od::Ball& ball) {
    // TODO: compute a confidence value

    // TODO: to modify
    ball.confidence = -3;
}

/* Compute gradient of grayscale image */
void od::compute_gradient_magnitude(const cv::Mat& frame, cv::Mat& magnitude){
    // Apply Sobel to get gradient magnitude
    cv::Mat grad_x, grad_y;
    cv::Sobel(frame, grad_x, 1, 0, 3);
    cv::Sobel(frame, grad_y, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, magnitude);

    // Normalize magnitude
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
}

void od::compute_color_white_ratio(const cv::Mat& ball, double& ratio){
    double color_ratio, white_ratio;
}