#include <object_detection.h>

/* Librarires required in this source file and not already included in object_detection.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>
// edge_detection: ed::sort_corners()
#include <edge_detection.h>
// segmentation: ball color constans
#include <segmentation.h>

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
void od::suppress_close_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_close, const double ratio) {
    // Collect not suppressed circles
    std::vector<cv::Vec3f> circles_filtered;

    // Create a vector of pairs of circles without repetitions
    std::vector<std::pair<cv::Vec3f, cv::Vec3f>> circles_pairs;
    for(size_t i = 0; i < circles.size(); i++) {
        for(size_t j = i + 1; j < circles.size(); j++)
            circles_pairs.push_back(std::make_pair(circles[i], circles[j]));
    }

    // For each pair of circles compute the distance between their centers
    for(std::pair<cv::Vec3f, cv::Vec3f> pair : circles_pairs) {
        // Compute distance between centers of circles i and j
        cv::Point2f center_i(pair.first[0], pair.first[1]);
        cv::Point2f center_j(pair.second[0], pair.second[1]);

        // double distance = cv::norm(center_i - center_j);
        double base = std::abs(center_i.x - center_j.x);
        double height = std::abs(center_i.y - center_j.y);
        double distance = std::sqrt(std::pow(base, 2) + std::pow(height, 2));

        // Index of circle with max radius between in pair
        cv::Vec3f max_circle = pair.first[2] > pair.second[2] ? pair.first : pair.second;
        cv::Vec3f min_circle = pair.first[2] <= pair.second[2] ? pair.first : pair.second;
        double max_circle_radius = max_circle[2];
        double min_circle_radius = min_circle[2];

        // Suppress close min circle
        double max_distance = max_circle_radius * ratio;
        if(distance <= max_distance) {
            max_circle[2] = (max_circle_radius + min_circle_radius) / 2;
            circles_filtered.push_back(max_circle);
        } else {
            circles_filtered.push_back(min_circle);
            circles_filtered.push_back(max_circle);
        }
        
        // std::cout << "Distance: " << distance <<  " Max distance: " << max_distance;
        // Print radius and position of max and min circles
        // std::cout << "\t| Max center(" << max_circle[0] << ", " << max_circle[1] << ") Max radius: " << max_circle[2];
        // std::cout << "\t| Min center(" << min_circle[0] << ", " << min_circle[1] << ") Min radius: " << min_circle[2] << std::endl;
    }

    // Update circles
    circles = circles_filtered;
}

/* Suppress too much big close circles */
void od::suppress_big_close_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big, const double min_distance) {
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
void od::suppress_small_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_small, const double radius_min) {
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

/* Suppress too much big circles */
void od::suppress_big_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big, const double radius_max) {
    std::vector<cv::Vec3f> circles_filtered;
    
    for(size_t i = 0; i < circles.size(); i++){
        if(circles[i][2] <= radius_max){
            circles_filtered.push_back(circles[i]);
        } else {
            circles_big.push_back(circles[i]);
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

/* Compute the mean circles of circles with center inside a bigger circle */
void od::compute_mean_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_mean, const double offset) {
    // Collect circles and new circles
    std::vector<cv::Vec3f> circles_filtered;
    
    // Compute the mean circles of circles with center inside a bigger circle
    for(size_t i = 0; i < circles.size(); i++) {
        // Circle i data
        cv::Vec3f circle_i = circles[i];
        cv::Point2f center_i(circle_i[0], circle_i[1]);
        double radius_i = circle_i[2];

        // Check if circle i is not already considered
        bool is_considered = false;
        for(size_t j = 0; j < circles_filtered.size(); j++) {
            // Circle j data
            cv::Vec3f circle_j = circles_filtered[j];
            cv::Point2f center_j(circle_j[0], circle_j[1]);
            double radius_j = circle_j[2];

            // Compute distance between centers of circle i and circle j
            double base = std::abs(center_i.x - center_j.x);
            double height = std::abs(center_i.y - center_j.y);
            double distance = std::sqrt(std::pow(base, 2) + std::pow(height, 2));

            // Set max distance
            double max_distance = radius_j + offset;
            
            // Check if circle i is inside circle j
            if(distance <= max_distance) {
                // Compute mean circle
                double radius_mean = (radius_i + radius_j) / 2;
                double x_mean = (center_i.x + center_j.x) / 2;
                double y_mean = (center_i.y + center_j.y) / 2;
                cv::Vec3f circle_mean(x_mean, y_mean, radius_mean);
                // Update circles mean
                circles_mean.push_back(circle_mean);
                // Replace pair of circles with mean circle
                circles_filtered[j] = circle_mean;
                // Update visiting
                is_considered = true;
            }
        }

        // Add circle i if not considered
        if(! is_considered)
            circles_filtered.push_back(circle_i);
    }

    // Update circles
    circles = circles_filtered;
}

// Function to find the median of a sorted vector
double get_median(std::vector<double>& vec) {
    // Vector size
    int size = vec.size();

    // If the size is even
    if (size % 2 == 0)
        return (vec[size / 2 - 1] + vec[size / 2]) / 2;
    else
        return vec[size / 2];
}

/* Normalize too much small or large circles */
void od::normalize_circles_radius(std::vector<cv::Vec3f>& circles) {
    float radius_sum = 0.0, radius_avg = 0.0;
    
    // Compute average radius only on not large circles
    for(size_t i = 0; i < circles.size(); i++)
        radius_sum += circles[i][2];
    radius_avg = radius_sum / circles.size();

    // Compute median of radius
    std::vector<double> radius_values;
    for(size_t i = 0; i < circles.size(); i++) {
        radius_values.push_back(circles[i][2]);
        std::cout << "Radius: " << circles[i][2] << std::endl;
    }
    double radius_median = get_median(radius_values);

    // Resize small circles
    for(size_t i = 0; i < circles.size(); i++) {
        if(circles[i][2] <= 14.0)
            circles[i][2] = radius_median;
    }

    /*// Compute median of radius
    std::vector<double> radius_values;
    for(size_t i = 0; i < circles.size(); i++)
        radius_values.push_back(circles[i][2]);
    double radius_median = get_median(radius_values);

    // Resize circles
    /*for(size_t i = 0; i < circles.size(); i++)
        circles[i][2] = radius_median;

    std::vector<double> radius_values_norm = radius_values;
    od::normalize_vector(radius_values_norm);
    for(size_t i = 0; i < circles.size(); i++)
        circles[i][2] = radius_median * (1 + radius_values_norm[i]);*/
}

/* Balls detection in given a video frame */
void od::object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners_float, const bool is_distorted, cv::Mat& video_frame, const std::string test_bboxes_video_path, const bool test_flag) {
    // Sorted float corners
    std::vector<cv::Point2f> corners(corners_float);
    ed::sort_corners(corners);

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
    std::vector<cv::Vec3f> circles, circles_big, circles_small, circles_mean, circles_close, circles_close_final;
	cv::HoughCircles(mask_balls, circles, cv::HOUGH_GRADIENT, 1,
		7,      // Distance between circles
		100, 9, // Canny edge detector parameters and circles center detection 
		3, 22); // Min-radius and max-radius of circles to detect

    // Suppress holes circles
    od::suppress_billiard_holes(circles, corners, is_distorted);
    // Merge neighboring circles
    od::compute_mean_circles(circles, circles_mean);
    // Suppress big circles
    double radius_max = 17.0;
    od::suppress_big_circles(circles, circles_big, radius_max);
    // Suppress small circles
    double radius_min = 3.0;
    od::suppress_small_circles(circles, circles_small, radius_min);

    // Additional operations
    od::suppress_black_circles(circles, mask_balls);
    // Normalize circles to median
    od::normalize_circles_radius(circles);
    // Show detected circles
    // TODO: to be removed
    cv::Mat frame = video_frames[n_frame].clone();
    for(size_t i = 0; i < circles.size(); i++) {
        // Circle data
        cv::Vec3i c = circles[i];
        cv::Point2f center(c[0], c[1]);
        unsigned int radius = c[2];

        cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    // Ball bounding boxes from circles
    std::vector<od::Ball> ball_bboxes;
    for(cv::Vec3f circle : circles) {
        // Circle data
        cv::Point center(circle[0], circle[1]);
        unsigned int radius = circle[2];

        // Ball bounding box
        od::Ball ball_bbox(center.x - radius, center.y - radius, 2*radius, 2*radius);
        ball_bboxes.push_back(ball_bbox);
    }

    // In case of test, read ball bboxes from dataset
    if(test_flag) {
        // Read true frame bboxes text file
        std::vector<od::Ball> test_ball_bboxes;
        std::string test_bboxes_frame_file_path;
        fsu::get_bboxes_frame_file_path(video_frames, n_frame, test_bboxes_video_path, test_bboxes_frame_file_path);

        // Read ball bounding box from frame bboxes text file
        bool confidence_flag = ! test_flag;
        fsu::read_ball_bboxes(test_bboxes_frame_file_path, test_ball_bboxes, confidence_flag);

        for(od::Ball& ball : test_ball_bboxes){
            ball.ball_class = 6;
        }
        // Replace detected ball bboxes with dataset ones
        ball_bboxes = test_ball_bboxes;
    }
    
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::create_bboxes_frame_file(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // TODO: print to remove
    if(bboxes_frame_file.is_open())
        std::cout << bboxes_frame_file_path << std::endl;

    // Compute magnitude on grayscale frame
    std::vector<double> gradient_counts, white_ratios, black_ratios;
    od::compute_gradient_balls(video_frames[n_frame], ball_bboxes, gradient_counts);

    // Compute ball colors informations
    od::compute_color_ratios(ball_bboxes, video_frames[n_frame], white_ratios, black_ratios);

    // Detect black and white balls
    int white_index = 0, black_index = 0;
    od::detect_white_black_balls(ball_bboxes, white_index, black_index, white_ratios, black_ratios, gradient_counts);

    // Scan each ball bounding box
    for(int i = 0; i < ball_bboxes.size(); ++i) {
        // Get ball bounding box
        od::Ball ball_bbox = ball_bboxes[i];

        // Ball class detection
        if(i != white_index && i != black_index){
            od::detect_ball_class(ball_bbox, i, white_ratios, black_ratios, gradient_counts);
        }

        // Compute confidence value
        od::set_ball_bbox_confidence(ball_bbox);

        // Write ball bounding box in frame bboxes text file
        fsu::write_ball_bbox(bboxes_frame_file, ball_bbox);

        // Apply ball classification to video frame
        od::overlay_ball_bounding_bbox(video_frame, ball_bbox);
    }

    // Close frame bboxes text file
    bboxes_frame_file.close();

    // Show video fram with classification
    cv::imshow("Frame with ball classification", video_frame);
    cv::waitKey(0);
}

/* Ball class detection */
void od::detect_ball_class(Ball& ball_bbox, const int ball_index, std::vector<double>& white_ratios, std::vector<double>& black_ratios, std::vector<double>& magnitude_counts) {
    // Extract magnitude count
    double magnitude_count = magnitude_counts[ball_index];

    // Extract ball color ratios
    double white_ratio = white_ratios[ball_index];
    double white_th = 0.15, grad_count_th = 0.1;

    // Ball classification
    if(white_ratio >= 1.75 * white_th) {
        // Stripe
        ball_bbox.ball_class = 4;
    } else if((white_ratio >= white_th) && (magnitude_count >= grad_count_th)){
        // Stripe
        ball_bbox.ball_class = 4;
    } else {
        // Solid
        ball_bbox.ball_class = 3;
    }
}

/* Set ball bounding box confidence value */
void od::set_ball_bbox_confidence(od::Ball& ball) {
    // TODO: Compute a confidence value
    ball.confidence = 1;
}

/* Compute gradient magnitude of the ball and the number of pixels with non-zero gradient */
void od::compute_gradient_balls(const cv::Mat& frame, const std::vector<od::Ball>& ball_bboxes, std::vector<double>& magnitude_counts) {
    // Scan all bounding boxes
    for(od::Ball ball_bbox : ball_bboxes) {
        // Get bounding box center and radius
        cv::Point box_center = cv::Point(ball_bbox.x + (ball_bbox.width / 2), ball_bbox.y + (ball_bbox.height / 2));
        unsigned int radius = ball_bbox.radius();

        // Get ball masks
        cv::Mat mask_ball = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat mask_grad_ball = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::circle(mask_ball, box_center, radius, cv::Scalar(255), cv::FILLED);
        cv::circle(mask_grad_ball, box_center, radius - 2, cv::Scalar(255), cv::FILLED);

        // Get gray ball region
        cv::Mat frame_roi;
        frame.copyTo(frame_roi, mask_ball);

        // Get grayscale frame
        cv::Mat frame_gray;
        cv::cvtColor(frame_roi, frame_gray, cv::COLOR_BGR2GRAY);

        // Compute ball gradient magnitude    
        cv::Mat magnitude;
        od::compute_gradient_magnitude(frame_gray, magnitude);
        magnitude.setTo(0, ~mask_grad_ball);

        // Compute gradient count
        double magnitude_count = static_cast<double> (cv::countNonZero(magnitude)) / cv::countNonZero(mask_grad_ball);
   
        // Store magnitude information
        magnitude_counts.push_back(magnitude_count);
    }
}

/* Compute gradient of grayscale image */
void od::compute_gradient_magnitude(const cv::Mat& frame, cv::Mat& magnitude) {
    // Apply Sobel to get gradient magnitude
    cv::Mat grad_x, grad_y;
    cv::Sobel(frame, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(frame, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, magnitude);

    // Normalize magnitude
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Keep only high valued gradients
    cv::threshold(magnitude, magnitude, 50, 0, cv::THRESH_TOZERO);
}

/* Compute ratio color-white ratio */
void od::compute_black_white_ratio(const cv::Mat& ball_region, double& white_ratio, double& black_ratio) {
    double color_count, white_count, black_count;
    
    // Define color thresholds
    cv::Scalar lower_th = cv::Scalar(100, 100, 100);
    cv::Scalar upper_th = cv::Scalar(255, 255, 255);

    // Count black and white pixels
    cv::Mat mask_color, mask_white, mask_black;
    cv::inRange(ball_region, lower_th, upper_th, mask_white);
    cv::inRange(ball_region, cv::Scalar(1, 1, 1), upper_th, mask_color);
    cv::inRange(ball_region, cv::Scalar(1, 1, 1), cv::Scalar(50, 50, 50), mask_black);
    
    white_count = cv::countNonZero(mask_white);
    black_count = cv::countNonZero(mask_black);
    color_count = cv::countNonZero(mask_color);

    white_ratio = white_count / (white_count + color_count);
    black_ratio = black_count / (white_count + color_count);
}

/* Compute white and black ratio for each ball */
void od::compute_color_ratios(std::vector<od::Ball> ball_bboxes, const cv::Mat& frame, std::vector<double>& white_ratios, std::vector<double>& black_ratios){
    for(od::Ball ball_bbox : ball_bboxes) {
        // Get bounding box center and radius
        cv::Point box_center = cv::Point(ball_bbox.x + (ball_bbox.width / 2), ball_bbox.y + (ball_bbox.height / 2));
        unsigned int radius = ball_bbox.radius();

        // Get ball masks
        cv::Mat mask_ball = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat mask_grad_ball = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::circle(mask_ball, box_center, radius, cv::Scalar(255), cv::FILLED);

        // Get ball region
        cv::Mat frame_roi;
        frame.copyTo(frame_roi, mask_ball);

        // Compute ball color ratio w.r.t. white
        double white_ratio, black_ratio;
        od::compute_black_white_ratio(frame_roi, white_ratio, black_ratio);

        white_ratios.push_back(white_ratio);
        black_ratios.push_back(black_ratio);
    }
}

/* Detect white and black balls */
void od::detect_white_black_balls(std::vector<od::Ball>& ball_bboxes, int& best_white_index, int& best_black_index, const std::vector<double>& white_ratio, const std::vector<double>& black_ratio, std::vector<double>& magnitude_counts){
    int sec_white_index = 0, sec_black_index = 0;
    
    best_white_index = 0;
    best_black_index = 0;

    // Detect best white ball candidates
    od::get_best_two_indexes(white_ratio, best_white_index, sec_white_index);

    // Detect best black ball candidates
    od::get_best_two_indexes(black_ratio, best_black_index, sec_black_index);

    // Check white consistency
    if((white_ratio[best_white_index] - white_ratio[sec_white_index]) <= 0.015 && magnitude_counts[sec_white_index] < magnitude_counts[best_white_index]){
        if(std::fabs(magnitude_counts[best_white_index] - magnitude_counts[sec_white_index]) > 0.2){
            best_white_index = sec_white_index;
        }
    }          

    // Check black consistency
    if(((black_ratio[best_black_index] - black_ratio[sec_black_index]) <= 0.03) && (magnitude_counts[sec_black_index] < magnitude_counts[best_black_index])){
        if(std::fabs(magnitude_counts[best_black_index] - magnitude_counts[sec_black_index]) > 0.05){
            best_black_index = sec_black_index;
        }
    }   

    ball_bboxes[best_white_index].ball_class = 1;
    ball_bboxes[best_black_index].ball_class = 2;
}

void od::get_best_two_indexes(const std::vector<double>& vec, int& best_index, int& sec_index){
    // Check index consistency
    if(best_index < 0 || best_index >= vec.size()){
        return;
    }
    
    // Make best and second best indexes different
    if(best_index == sec_index && vec.size() > 1) {
        if(best_index <= (vec.size() - 2)) {
            sec_index++;
        } else {
            sec_index = 0;
        }
    }

    // Choose best index
    for(size_t i = 0; i < vec.size(); i++) {
        if(vec[i] >= vec[best_index]){
            best_index = i;
        }
    }

    // Find second best index
    for(size_t i = 0; i < vec.size(); i++) {
        if(i != best_index) {
            if(vec[i] >= vec[sec_index]) {
                sec_index = i;
            }
        }
    }
}

/* Normalize given vector */
void od::normalize_vector(std::vector<double>& vec) {
    // Check for empty vector
    if(vec.empty()) {
        return;
    }
    
    // Compute norm
    double norm = 0, sum = 0;
    for(const double& elem : vec) {
        sum += (elem * elem);
    }
    norm = std::sqrt(sum);

    // Compute normalized vector
    std::vector<double> norm_vector;
    for(const double& elem : vec){
        norm_vector.push_back(elem / norm);
    }

    vec = norm_vector;
}

/* Show ball bounding boxes according to class color */
void od::overlay_ball_bounding_bbox(cv::Mat& video_frame, od::Ball ball_bbox) {
    // Ball bounding box corners offset
    unsigned int offset = 1;

    // Ball bounding box corners with offset
    cv::Point tl_corner(ball_bbox.x - offset, ball_bbox.y - offset);
    cv::Point br_corner(ball_bbox.x + ball_bbox.width + offset, ball_bbox.y + ball_bbox.height + offset);
    
    // Ball class colors
    std::vector<cv::Scalar> ball_colors = {sg::WHITE_BALL_BGR.second, sg::BLACK_BALL_BGR.second, sg::SOLID_BALL_BGR.second, sg::STRIPE_BALL_BGR.second};

    // Create a transparent rectangle
    cv::Mat ball_roi = video_frame(cv::Rect(tl_corner.x, tl_corner.y, br_corner.x - tl_corner.x , br_corner.y - tl_corner.y));
    cv::Mat color(ball_roi.size(), CV_8UC3, ball_colors[ball_bbox.ball_class-1]); 
    double alpha = 0.3;
    cv::addWeighted(color, alpha, ball_roi, 1.0 - alpha , 0, ball_roi); 

    // Draw ball bounding box
    int line_thickness = 2;
    cv::rectangle(video_frame, tl_corner, br_corner, ball_colors[ball_bbox.ball_class-1], line_thickness);
}