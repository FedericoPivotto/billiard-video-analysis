// user-defined libraries
#include <billiard_metric.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

// input parameter: video filename, true frame bbox filename, predicted frame bbox filename
// ASSUMPTION: each bounding box row has a 6th element which is the associated confidence value
int main(int argc, char** argv) {
    // TODO: extract true and predicted bounding boxes data
    // NOTE: fix this only after all metrics are fine, substituting this part extracting correctly data from the provide video filename as input parameter

    // read true bounding boxes
    // NOTE: it will come from result directory
    // ATTENTION: Federico will take care of this part
    std::vector<od::Ball> true_balls;
    std::string true_bboxes_frame_file_path = "../metrics/data/true_frame_first_bbox.txt";
    fsu::read_ball_bboxes(true_bboxes_frame_file_path, true_balls);

    // print true balls
    for(od::Ball ball : true_balls)
        std::cout << "True ball: " << ball << std::endl;

    // dummy print
    std::cout << std::endl;

    // read predicted bounding boxes
    // NOTE: it will come from result directory
    // ATTENTION: Federico will take care of this part
    std::vector<od::Ball> predicted_balls;
    std::string predicted_bboxes_frame_file_path = "../metrics/data/predicted_frame_first_bbox.txt";
    fsu::read_ball_bboxes_with_confidence(predicted_bboxes_frame_file_path, predicted_balls);

    // print predicted balls
    for(od::Ball ball : predicted_balls)
        std::cout << "Predicted ball: " << ball << " " << ball.confidence << std::endl;

    // TODO: matches search
    bm::matches_search();

    // TODO: localization metric
    bm::localization_metric();

    // TODO: segmentation metric
    bm::segmentation_metric();

    return 0;
}