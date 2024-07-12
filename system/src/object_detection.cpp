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
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    // HSV to grayscale
    cv::Mat hsv_channels[3];
    cv::split(frame_hsv, hsv_channels);
    cv::Mat frame_gs = hsv_channels[2];

    // Frame preprocess
    cv::GaussianBlur(frame_gs, frame_gs, cv::Size(5, 5), 2, 2);
	// cv::medianBlur(frame_gs, frame_gs, 5);

    // Hough circle transform
    std::vector<cv::Vec3f> circles;
	cv::HoughCircles(frame_gs, circles, cv::HOUGH_GRADIENT, 1,
		5, // distance between circles
		100, 10, // canny edge detector parameters and circles center detection 
		7, 15); // min_radius & max_radius of circles to detect
    
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
	// Wait key before going ahead
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
    bboxes_frame_file.close();
}

// TODO: define ball bbox confidence
void od::set_ball_bbox_confidence(od::Ball& ball) {
    ball.confidence = 1;
}

bool od::is_ball_inside_field(const std::vector<cv::Point2f> corners, cv::Point center, unsigned int radius) {
    return true;
}