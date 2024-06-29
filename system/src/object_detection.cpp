#include <object_detection.h>

// librarires required in this source file and not already included in object_detection.h

od::Ball::Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class) : x(x), y(y), width(width), height(height), ball_class(ball_class) {
}

std::pair<unsigned int, unsigned int> od::Ball::center() const {
    return {x + width / 2, y + height / 2};
}

unsigned int od::Ball::radius() const {
    return width < height ? width / 2 : height / 2;
}

void od::detect_ball_class(Ball& ball_bbox, cv::Mat frame) {
    // TODO: remove background

    // TODO: detect ball class
    // - 1:white ball - white is the predominant color
    // - 2:black ball - black is the predominant color
    // - 3:solid ball - color, except white and black, is the predominant color
    // - 4:stripe ball - both white and color are predominant colors

    // TODO: set ball class
}

std::ostream& od::operator<<(std::ostream& os, const Ball& ball) {
    return os << ball.x << " " << ball.y << " " << ball.width << " " << ball.height << " " << ball.ball_class;
}