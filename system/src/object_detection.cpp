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

/* hsv_hough_callback function parameters */
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
    
    cv::Mat edge_map;
    cv::Canny(mask, edge_map, 10, 100);

    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(edge_map, circles, cv::HOUGH_GRADIENT, 1,
		10, // distance between circles
		90, 9, // canny edge detector parameters and circles center detection 
		1, 20); // min_radius & max_radius of circles to detect
    
    // Show detected circles
    for(size_t i = 0; i < circles.size(); i++) {
        // Circle data
        cv::Vec3i c = circles[i];
        cv::Point center(c[0], c[1]);
        unsigned int radius = c[2];

        // Show circle center
        cv::circle(frame_hsv, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        // Show circle outline
        cv::circle(frame_hsv, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    }

    // Display our result
	cv::imshow(params.window_name, frame_hsv);
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
    cv::inRange(frame_hsv, cv::Scalar(params.iLowH, params.iLowS, params.iLowV), cv::Scalar(params.iHighH, params.iHighS, params.iHighV), mask);

    // Dilate and erosion set operations on mask 
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(mask, mask, kernel);
    cv::erode(mask, mask, kernel);

    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
		10, // distance between circles
		100, 9, // canny edge detector parameters and circles center detection 
		5, 20); // min_radius & max_radius of circles to detect
    
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


/* Balls detection in given a video frame */
void od::object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame) {
    // Create frame bboxes text file
    std::string bboxes_frame_file_path;
    fsu::create_bboxes_frame_file(video_frames, n_frame, bboxes_video_path, bboxes_frame_file_path);

    // Vector of bounding boxes
    std::vector<od::Ball> ball_bboxes;
    fsu::read_ball_bboxes(bboxes_frame_file_path, ball_bboxes);

    // Video frame clone
    cv::Mat frame(video_frames[n_frame].clone());
    cv::Mat frame_gray;

    //// CLAHE
    //cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    //clahe->setClipLimit(4);
    //cv::Mat frame_clahe;
    //clahe->apply(frame_gray, frame_clahe);
    //cv::imshow("CLAHE", frame_clahe);

    // Image frame pre-processing       
    //cv::Mat frame_hsv, frame_bilateral;
    //cv::bilateralFilter(frame, frame_bilateral, 9, 100.0, 75.0);
    //cv::cvtColor(frame_bilateral, frame_hsv, cv::COLOR_BGR2HSV);
    
    // HSV channels
    //cv::Mat hsv_channels[3];
    //cv::split(frame_hsv, hsv_channels);
    // Apply histogram equalization to each channel
    // cv::equalizeHist(hsv_channels[0], hsv_channels[0]);
    // cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
    // cv::equalizeHist(hsv_channels[2], hsv_channels[2]);

    // Merge the equalized channels back
    //cv::Mat frame_hsv_equalized;
    //cv::merge(hsv_channels, 3, frame_hsv_equalized);

    // Show the frame HSV channels
    // cv::imshow("Hue", hsv_channels[0]);
    // cv::imshow("Saturation", hsv_channels[1]);
    // cv::imshow("Value", hsv_channels[2]);

    // Show the frame BRG
    // cv::imshow("Frame BGR bilateral", frame_bilateral);

    // Show the frame HSV
    // cv::imshow("Frame HSV", frame_hsv);
    // Show the frame HSV equalized
    // cv::imshow("Frame HSV equalized", frame_hsv_equalized);

    // HSV parameters
    int iLowH = 60, iHighH = 120, maxH = 179;
    int iLowS = 150, iHighS = 255, maxS = 255;
    int iLowV = 110, iHighV = 255, maxV = 255;

    // Bilateral filter parameters
    int max_th = 300;
    int color_std = 100;
    int space_std = 75;

    // HSV window trackbars
    //cv::Mat mask;
    // ParameterHSV hsvp = {"Control HSV", &frame_hsv, &mask, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, maxH, maxS, maxV};
    ParameterHoughHSV hsvp = {"Control HSV", &frame, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV, color_std, space_std, max_th, maxH, maxS, maxV};
    
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

    //// Mask generation by ranged HSV
    //cv::Mat mask;
    //cv::Scalar lower_hsv(60, 150, 110), upper_hsv(120, 255, 255);
    //cv::inRange(frame_hsv_equalized, lower_hsv, upper_hsv, mask);
    //
    //// Dilate and erode mask
    //cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    //cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    // Show frame and mask
    // cv::imshow("Mask", mask);

    /*// Hough circle parameters
	cv::Mat detected_edges = frame.clone();
    ParameterHoughCircle hcp = {"Hough circle", &mask, &detected_edges, 10, 90, 9, 1, 20, 100, 100, 100, 100, 100};

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
    cv::destroyAllWindows();*/
/*
    // Hough circle transform
    // TODO: tune parameters using trackbars
    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
		10, // distance between circles
		90, 9, // canny edge detector parameters and circles center detection 
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
	
    // Wait key
    cv::waitKey(0);
    cv::destroyAllWindows();
*/
    /*// TODO: detect ball bounding boxes using Viola and Jones approach
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