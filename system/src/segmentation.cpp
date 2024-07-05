#include <segmentation.h>

// librarires required in this source file and not already included in segmentation.h
// object detection library
#include <object_detection.h>

int function_name() {
    int variable_name = 69;
    
    return variable_name;
}

/*void ball_segmentation(od::Ball ball_bbox, cv::Mat &frame) {
    if (ball_bbox.ball_class == 1)
    {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(255, 255, 255);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 2)
    {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(0, 0, 0);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 3)
    {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(255, 0, 0);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if(ball_bbox.ball_class == 4)
    {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(0, 0, 255);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
}*/