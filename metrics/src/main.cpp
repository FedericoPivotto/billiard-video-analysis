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
    // for(od::Ball ball : true_balls)
    //     std::cout << "True ball: " << ball << std::endl;
    // std::cout << "Number of true balls: " << true_balls.size() << std::endl;

    // Dummy print
    // std::cout << std::endl;

    // Read predicted bounding boxes
    // NOTE: it will come from result directory
    // ATTENTION: Federico will take care of this part
    std::vector<od::Ball> predicted_balls;
    std::string predicted_bboxes_frame_file_path = "../metrics/data/predicted_frame_first_bbox.txt";
    bool confidence_flag = true;
    fsu::read_ball_bboxes(predicted_bboxes_frame_file_path, predicted_balls, confidence_flag);

    // Print predicted balls
    // TODO: to be removed
    // for(od::Ball ball : predicted_balls)
    //     std::cout << "Predicted ball: " << ball << " " << ball.confidence << std::endl;
    // std::cout << "Number of predicted balls: " << predicted_balls.size() << std::endl;
    
    // Dummy print
    // std::cout << std::endl;

    // Average precision vector
    std::vector<double> aps;

    // For each ball class
    int num_classes = 4;
    for(size_t i = 1; i <= num_classes; ++i) {
        // Extract true balls of the current class
        std::vector<od::Ball> true_balls_class;
        for(od::Ball ball : true_balls)
            if(ball.ball_class == i)
                true_balls_class.push_back(ball);

        // Extract predicted balls of the current class
        std::vector<od::Ball> predicted_balls_class;
        for(od::Ball ball : predicted_balls)
            if(ball.ball_class == i)
                predicted_balls_class.push_back(ball);

        // Best matches search
        std::vector<bm::BallMatch> best_ball_matches;
        bm::matches_search(true_balls_class, predicted_balls_class, best_ball_matches);
        
        // Print best ball matches
        // TODO: to remove
        /*std::cout << "Best ball matches: " << std::endl;
        for(bm::BallMatch ball_match : best_ball_matches)
            std::cout << ball_match << std::endl;
        std::cout << "Average_precision:" << std::endl;*/

        // Class average precision
        aps.push_back(bm::average_precision(true_balls_class.size(), best_ball_matches));

        // Dummy print
        // TODO: to remove
        /*std::cout << std::endl;*/
    }

    // TODO: Localization metric
    // BUG: to fix calculation
    double map = bm::localization_metric(aps, num_classes);

    // Print mean average precision
    std::cout << "Mean Average Precision (mAP): " << map << std::endl;

    // TODO: Segmentation metric
    bm::segmentation_metric();

    return 0;
}