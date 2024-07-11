#include <segmentation.h>

// librarires required in this source file and not already included in segmentation.h
// imgproc: cv::circle
#include <opencv2/imgproc.hpp>

void sg::ball_segmentation(od::Ball ball_bbox, cv::Mat& frame) {
    
    std::pair<unsigned int, unsigned int> center_point = ball_bbox.center();
    cv::Point center(center_point.first, center_point.second);
    
    /*std::vector<std::vector<unsigned int>> triplets = {{255, 255, 255}, {0, 0, 0}, {255, 0, 0}, {0, 0, 255}};
    cv::Scalar color(triplets[ball_bbox.ball_class][0], triplets[ball_bbox.ball_class][1], 
        triplets[ball_bbox.ball_class][2], triplets[ball_bbox.ball_class][3]);*/
    
    cv::Scalar color;
    if (ball_bbox.ball_class == 1) {
        color = cv::Scalar(255, 255, 255);
    }
    else if (ball_bbox.ball_class == 2) {
        color = cv::Scalar(0, 0, 0);
    }
    else if (ball_bbox.ball_class == 3) {
        color = cv::Scalar(255, 0, 0);
    }
    else if(ball_bbox.ball_class == 4) {
        color = cv::Scalar(0, 0, 255);
    }

    cv::circle( frame, center, ball_bbox.radius(), color, -1 );
}

void sg::field_segmentation(const std::vector<cv::Point2f> corners, cv::Mat& frame){
    std::vector<cv::Point> intCorners = sg::convertToIntegerPoints(corners);
    std::vector<std::vector<cv::Point>> fillContAll;
    
    fillContAll.push_back(intCorners);

    cv::fillPoly(frame, fillContAll, cv::Scalar(0, 255, 0));
}

std::vector<cv::Point> sg::convertToIntegerPoints(const std::vector<cv::Point2f>& floatPoints) {
    std::vector<cv::Point> intPoints;
    intPoints.reserve(floatPoints.size()); 

    for (const auto& point : floatPoints) {
        intPoints.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }

    return intPoints;
}
