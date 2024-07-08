#include <segmentation.h>

// librarires required in this source file and not already included in segmentation.h

// imgproc: cv::circle
#include <opencv2/imgproc.hpp>

// WARNING: avoid using!
using namespace cv;
using namespace std;

void sg::ball_segmentation(od::Ball ball_bbox, cv::Mat& frame) {
    if (ball_bbox.ball_class == 1) {
        pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        Point center(center_point.first, center_point.second);
        Scalar color(255, 255, 255);
        circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 2) {
        pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        Point center(center_point.first, center_point.second);
        Scalar color(0, 0, 0);
        circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if (ball_bbox.ball_class == 3) {
        pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        Point center(center_point.first, center_point.second);
        Scalar color(255, 0, 0);
        circle( frame, center, ball_bbox.radius(), color, -1 );
    }
    else if(ball_bbox.ball_class == 4) {
        pair<unsigned int, unsigned int> center_point = ball_bbox.center();
        Point center(center_point.first, center_point.second);
        Scalar color(0, 0, 255);
        circle( frame, center, ball_bbox.radius(), color, -1 );
    }
}

vector<Point> sg::convertToIntegerPoints(const vector<Point2f>& floatPoints) {
    vector<Point> intPoints;
    intPoints.reserve(floatPoints.size()); 

    for (const auto& point : floatPoints) {
        intPoints.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }

    return intPoints;
}
