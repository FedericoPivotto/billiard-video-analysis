#include <object_detection.h>

/* Librarires required in this source file and not already included in object_detection.h */

// imgproc: cv::cvtColor(), color space conversion codes
#include <opencv2/imgproc.hpp>
// filesystem_utils: fsu::create_bboxes_frame_file()
#include <filesystem_utils.h>

/* Ball class */
od::Ball::Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class, double confidence) : x(x), y(y), width(width), height(height), ball_class(ball_class), confidence(confidence) {
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

/* Ball operator << overload */
std::ostream& od::operator<<(std::ostream& os, const Ball& ball) {
    // Ball information string
    return os << ball.x << " " << ball.y << " " << ball.width << " " << ball.height << " " << ball.ball_class;
}

/* Ball class detection */
void od::detect_ball_class(Ball& ball_bbox, cv::Mat frame) {
    // TODO: remove background

    // TODO: detect ball class
    // - 1:white ball - white is the predominant color
    // - 2:black ball - black is the predominant color
    // - 3:solid ball - color, except white and black, is the predominant color
    // - 4:stripe ball - both white and color are predominant colors

    // TODO: set ball class
    // ATTENTION: remove this unsensed code
    unsigned int min = 1, max = 4;
    ball_bbox.ball_class = std::rand() % (max - min + 1) + min;
}

/* hsv_callback function parameters */
typedef struct {
	// Window name
	const char* window_name;

	// Pointers to images
	cv::Mat* src;
	cv::Mat* dst;

    // HSV parameters
    int iLowH;
    int iHighH;
    int iLowS;
    int iHighS;
    int iLowV;
    int iHighV;

    // Max threshold
    int maxH;
    int maxS;
    int maxV;
} ParameterHSV;

/* hough_circle_callback function parameters */
typedef struct {
	// Window name
	const char* window_name;

	// Pointers to images
	cv::Mat* src;
	cv::Mat* dst;

    // Hough circle parameters
    int min_dist;
    int param1;
    int param2;
    int min_radius;
    int max_radius;

    // Max threshold
    int maxD;
    int maxP1;
    int maxP2;
    int maxMinR;
    int maxMaxR;
} ParameterHoughCircle;

/* Callback function */
static void hsv_callback(int pos, void* userdata) {
    // Get Canny parameters from userdata
	ParameterHSV params = *((ParameterHSV*) userdata);

    // Dereference images
    cv::Mat frame_hsv = *params.src;
	cv::Mat mask = *params.dst;

    // Threshold the image
    cv::inRange(frame_hsv, cv::Scalar(params.iLowH, params.iLowS, params.iLowV), cv::Scalar(params.iHighH, params.iHighS, params.iHighV), mask);

    // Display our result
	cv::imshow(params.window_name, mask);
}

/* Callback function */
static void hough_circle_callback(int pos, void* userdata) {
    // Get Hough circle parameters from userdata
    ParameterHoughCircle params = *((ParameterHoughCircle*) userdata);

    // Dereference images
    cv::Mat mask = *params.src;
    cv::Mat detected_edges = (*params.dst).clone();

    // Hough circle transform
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1, params.min_dist, params.param1, params.param2, params.min_radius, params.max_radius);

    // Show detected circles
    for(size_t i = 0; i < circles.size(); i++) {
        // Circle data
        cv::Vec3i c = circles[i];
        cv::Point center(c[0], c[1]);
        unsigned int radius = c[2];

        // Show circle center
        cv::circle(detected_edges, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        // Show circle outline
        cv::circle(detected_edges, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    }

    // Show the detected circles
    cv::imshow(params.window_name, detected_edges);
}

/* Balls detection in given a video frame */
void od::object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame) {
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::create_bboxes_frame_file(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Open frame bboxes text file
    std::ofstream bboxes_frame_file(bboxes_frame_file_path);

    // Vector of bounding boxes
    std::vector<od::Ball> ball_bboxes;

    // Video frame clone
    cv::Mat frame(video_frames[n_frame].clone());

    // BGR to HSV
    /*cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);*/

    // HSV to grayscale
    /*cv::Mat hsv_channels[3];
    cv::split(frame_hsv, hsv_channels);
    cv::Mat frame_gs = hsv_channels[2];*/

    // Useful code for ball mask       
    cv::Mat frame_hsv, frame_bilateral = frame.clone();
    cv::bilateralFilter(frame, frame_bilateral, 9, 100.0, 75.0);
    cv::cvtColor(frame_bilateral, frame_hsv, cv::COLOR_BGR2HSV);

    /*// HSV parameters
    int iLowH = 0, iHighH = 179, maxH = 179;
    int iLowS = 0, iHighS = 255, maxS = 255;
    int iLowV = 0, iHighV = 255, maxV = 255;

    // HSV window trackbars
    cv::Mat mask;
    ParameterHSV hsvp = {"Control HSV", &frame_hsv, &mask, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, maxH, maxS, maxV};

    // Control window
    cv::namedWindow(hsvp.window_name);

    // Create trackbar for Hue (0 - 179)
    cv::createTrackbar("LowH", hsvp.window_name, &hsvp.iLowH, hsvp.maxH, hsv_callback, &hsvp);
    cv::createTrackbar("HighH", hsvp.window_name, &hsvp.iHighH, hsvp.maxH, hsv_callback, &hsvp);

    // Create trackbar for Saturation (0 - 255)
    cv::createTrackbar("LowS", hsvp.window_name, &hsvp.iLowS, hsvp.maxS, hsv_callback, &hsvp);
    cv::createTrackbar("HighS", hsvp.window_name, &hsvp.iHighS, hsvp.maxS, hsv_callback, &hsvp);
    
    // Create trackbar for Value (0 - 255)
    cv::createTrackbar("LowV", hsvp.window_name, &hsvp.iLowV, hsvp.maxV, hsv_callback, &hsvp);
    cv::createTrackbar("HighV", hsvp.window_name, &hsvp.iHighV, hsvp.maxV, hsv_callback, &hsvp);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();*/
    
    /*// Threshold the image
    cv::Mat frame_thresholded;
    cv::inRange(frame_hsv, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), frame_thresholded);

    // Show the thresholded frame
    cv::imshow("Thresholded", frame_thresholded);
    // Show the original frame
    cv::imshow("Original", frame);
    // Wait key
    cv::waitKey(0);*/

    // Mask generation by ranged HSV color segmentation
    cv::Mat mask;
    cv::Scalar lower_hsv(60, 150, 140);
    cv::Scalar upper_hsv(120, 255, 255);  
    cv::inRange(frame_hsv, lower_hsv, upper_hsv, mask);

    // Dilate and erode mask
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9))); 

    // Show frame and mask
    cv::imshow("BGR", frame);
    cv::imshow("HSV", frame_hsv);
	cv::imshow("Mask", mask);
	// cv::waitKey(0);

    // Hough circle parameters
	cv::Mat detected_edges = frame.clone();
    ParameterHoughCircle hcp = {"Hough circle", &mask, &detected_edges, 10, 90, 9, 1, 17, 100, 100, 100, 100, 100};

    // Control window
    cv::namedWindow(hcp.window_name);

    // Create trackbar for min distance
    cv::createTrackbar("Min distance", hcp.window_name, &hcp.min_dist, hcp.maxD, hough_circle_callback, &hcp);
    // Create trackbar for canny edge detector parameter 1
    cv::createTrackbar("Param 1", hcp.window_name, &hcp.param1, hcp.maxP1, hough_circle_callback, &hcp);
    // Create trackbar for canny edge detector parameter 2
    cv::createTrackbar("Param 2", hcp.window_name, &hcp.param2, hcp.maxP2, hough_circle_callback, &hcp);
    // Create trackbar for min radius
    cv::createTrackbar("Min radius", hcp.window_name, &hcp.min_radius, hcp.maxMinR, hough_circle_callback, &hcp);
    // Create trackbar for max radius
    cv::createTrackbar("Max radius", hcp.window_name, &hcp.max_radius, hcp.maxMaxR, hough_circle_callback, &hcp);

    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();

    /*// Hough circle transform
    // TODO: tune parameters using trackbars
    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
		5, // distance between circles
		90, 10, // canny edge detector parameters and circles center detection 
		1, 20); // min_radius & max_radius of circles to detect
    
    // Show detected circles
	for(size_t i = 0; i < circles.size(); i++) {
		// Circle data
        cv::Vec3i c = circles[i];
		cv::Point center(c[0], c[1]);
		unsigned int radius = c[2];

        // Check if ball is inside the field
        if(! od::is_ball_inside_field(corners, center, radius)) {
            std::cout << "Center: (" << center.x << ", " << center.y << ") - Radius: " << radius << std::endl;
            continue;
        }

        // Ball creation
        od::Ball detected_ball(center.x-radius, center.y-radius, radius*2, radius*2, 0);

        // Circle ball
        ball_bboxes.push_back(detected_ball);
        
		// Show circle center
		cv::circle(frame, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
		// Show circle outline
		cv::circle(frame, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
	}

	// Show the detected circles
	cv::imshow("Detected circles", frame);
	cv::waitKey(0);

    // TODO: detect ball bounding boxes using Viola and Jones approach
    // TODO: update ball vector with bounding box x, y, width, height
    // SEE: notes p.122 for extract ball image

    // scan each ball bounding box
    for(od::Ball ball_bbox : ball_bboxes) {
        // TODO: ball class detection
        od::detect_ball_class(ball_bbox, video_frames[n_frame]);

        // TODO: compute confidence value
        od::set_ball_bbox_confidence(ball_bbox);

        // Write ball bounding box in frame bboxes text file
        fsu::write_ball_bbox(bboxes_frame_file, ball_bbox);
    }

    // Close frame bboxes text file
    bboxes_frame_file.close();*/
}

// TODO: define ball bbox confidence
void od::set_ball_bbox_confidence(od::Ball& ball) {
    ball.confidence = 1;
}

bool od::is_ball_inside_field(const std::vector<cv::Point2f> corners, cv::Point center, unsigned int radius) {
    return true;
}