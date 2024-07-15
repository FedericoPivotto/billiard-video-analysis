#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

/* Libraries required in this source file */

// utility: std::pair
#include <utility>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

/* Object detection namespace */
namespace od {
    // Classes declaration
    class Ball {
        public:
            // Ball bounding box
            unsigned int x, y, width, height, ball_class;
            double confidence;

            // Constructor
            Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class, double confidence = 0);

            // Function declarations
            std::pair<unsigned int, unsigned int> center() const;
            unsigned int radius() const;
            cv::Rect get_rect_bbox();
            void set_rect_bbox(cv::Rect bbox);
    
            // Operator overload declarations
            friend std::ostream& operator<<(std::ostream& os, const Ball& ball);
    };

    // Definition of the operator<< function outside the class
    std::ostream& operator<<(std::ostream& os, const Ball& ball);

    // Detect function declarations
    void detect_ball_class(Ball& ball_bbox, cv::Mat frame);

    // Ball bounding box confidence function declaration
    void set_ball_bbox_confidence(od::Ball& ball);

    // Object detection main declaration
    void object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame);

    // Check function declaration
    bool is_ball_inside_field(const std::vector<cv::Point2f> corners, cv::Point center, unsigned int radius);
}

#endif // OBJECT_DETECTION_H