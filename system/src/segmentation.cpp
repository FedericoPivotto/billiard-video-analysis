#include <segmentation.h>

// librarires required in this source file and not already included in segmentation.h
#include <iostream>
// imgproc: cv::circle
#include <opencv2/imgproc.hpp>

void sg::ball_segmentation(od::Ball ball_bbox, const cv::Mat& frame) {

    //imshow("frame", frame);

    if (ball_bbox.ball_class == 1) {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(255, 255, 255);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 2) {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(0, 0, 0);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 3) {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(255, 0, 0);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if(ball_bbox.ball_class == 4) {
        std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        cv::Point center(center_point.first, center_point.second);
        cv::Scalar color(0, 0, 255);
        cv::circle( frame, center, ball_bbox.radius(), color, -1 );
    }
}

std::vector<cv::Point> sg::convertToIntegerPoints(const std::vector<cv::Point2f>& floatPoints) {
    std::vector<cv::Point> intPoints;
    intPoints.reserve(floatPoints.size()); 

    for (const auto& point : floatPoints) {
        intPoints.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }

    return intPoints;
}
