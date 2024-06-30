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

            // constructor
            Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class);

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
}

#endif // OBJECT_DETECTION_H