/* User-defined libraries */
#include <billiard_metric.h>

// filesystem_utils: filesystem utilities
#include <filesystem_utils.h>

// object detection library
#include <object_detection.h>

/* Computer vision metrics main */
// Input parameter: video filename, true frame bbox filename, predicted frame bbox filename
// ASSUMPTION: each bounding box row has a 6th element which is the associated confidence value
int main(int argc, char** argv) {
    // TODO: extract true and predicted bounding boxes data
    // NOTE: fix this only after all metrics are fine, substituting this part extracting correctly data from the provide video filename as input parameter

    // Read true bounding boxes
    // NOTE: it will come from dataset directory
    // ATTENTION: Federico will take care of this part
    std::vector<od::Ball> true_balls;
    std::string true_bboxes_frame_file_path = "../metrics/data/true_frame_first_bbox.txt";
    fsu::read_ball_bboxes(true_bboxes_frame_file_path, true_balls);

    // Print true balls
    // TODO: to be removed
    for(od::Ball ball : true_balls)
        std::cout << "True ball: " << ball << std::endl;
    std::cout << "Number of true balls: " << true_balls.size() << std::endl;

    // Dummy print
    std::cout << std::endl;

    // Read predicted bounding boxes
    // NOTE: it will come from result directory
    // ATTENTION: Federico will take care of this part
    std::vector<od::Ball> predicted_balls;
    std::string predicted_bboxes_frame_file_path = "../metrics/data/predicted_frame_first_bbox.txt";
    bool confidence_flag = true;
    fsu::read_ball_bboxes(predicted_bboxes_frame_file_path, predicted_balls, confidence_flag);

    // Print predicted balls
    // TODO: to be removed
    for(od::Ball ball : predicted_balls)
        std::cout << "Predicted ball: " << ball << " " << ball.confidence << std::endl;
    std::cout << "Number of predicted balls: " << predicted_balls.size() << std::endl;
    
    // Dummy print
    std::cout << std::endl;

    // TODO: matches search
    bm::matches_search(true_balls, predicted_balls);

    // TODO: localization metric
    bm::localization_metric();

    // TODO: segmentation metric
    bm::segmentation_metric();

    return 0;
}