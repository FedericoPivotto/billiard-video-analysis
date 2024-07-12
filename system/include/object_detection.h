#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

// libraries required in this source file

// utility: std::pair
#include <utility>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

namespace od {
    // classes declaration
    class Ball {
        public:
            // ball bounding box
            unsigned int x, y, width, height, ball_class;
            double confidence;

            // constructor
            Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class, double confidence = 0);

            // function declarations
            std::pair<unsigned int, unsigned int> center() const;
            unsigned int radius() const;
    
            // operator overload declarations
            friend std::ostream& operator<<(std::ostream& os, const Ball& ball);
    };

    // definition of the operator<< function outside the class
    std::ostream& operator<<(std::ostream& os, const Ball& ball);

    // detect function declarations
    void detect_ball_class(Ball& ball_bbox, cv::Mat frame);

    // object detection main
    void object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners, cv::Mat& video_frame);
}

#endif // OBJECT_DETECTION_H