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
    // TODO: it will come from dataset directory
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
    // TODO: it will come from result directory
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
        // CHECK: is ground truth per class
        aps.push_back(bm::average_precision(best_ball_matches, predicted_balls_class, true_balls_class.size()));

        // Print average precision
        // TODO: to remove
        std::cout << "Average Precision (AP) for class " << i << ": " << aps[i - 1] << std::endl;
    }

    // TODO: Localization metric
    // BUG: to fix calculation
    double map = bm::localization_metric(aps, num_classes);

    // Print mean average precision
    // TODO: to remove and replace with writing in a text file
    std::cout << "Mean Average Precision (mAP): " << map << std::endl;

    // Dummy print
    std::cout << std::endl;

    // Read true mask
    // TODO: it will come from result directory
    // std::string true_mask_path = "../metrics/data/true_mask_frame_first.png";
    std::string true_mask_path = "../metrics/data/wrong_mask.png";
    cv::Mat true_mask = cv::imread(true_mask_path, cv::IMREAD_GRAYSCALE);
    // safety check on true mask
	if(true_mask.data == NULL) {
		std::cout << "Error: The image cannot be read." << std::endl;
		exit(bm::IMAGE_READ_ERROR);
	}

    // Read predicted mask
    // TODO: it will come from result directory
    std::string predicted_mask_path = "../metrics/data/predicted_mask_frame_first.png";
    cv::Mat predicted_mask = cv::imread(predicted_mask_path, cv::IMREAD_GRAYSCALE);
    // safety check on predicted mask
	if(predicted_mask.data == NULL) {
		std::cout << "Error: The image cannot be read." << std::endl;
		exit(bm::IMAGE_READ_ERROR);
	}

    // Show true and predicted mask
    /*cv::imshow("True mask", true_mask);
    cv::imshow("Predicted mask", predicted_mask);
    cv::waitKey(0);*/

    // Vectore of IoU values
    std::vector<double> ious;

    // TODO: Segmentation metric
    // TODO: to fix
    // For each ball class
    num_classes = 5;
    for(size_t i = 1; i <= num_classes; ++i) {
        // Class IoU
        double class_iou = bm::iou_class(true_mask, predicted_mask, i);

        // Add class IoU
        ious.push_back(class_iou);

        // Print result
        // TODO: to remove
        std::cout << "IoU for class " << i << ": " << class_iou << std::endl;
    }

    // Compute mIoU
    double miou = bm::segmentation_metric(ious);

    // Print mIoU
    // TODO: to remove and replace with writing in a text file
    std::cout << "Mean Intersection over Union (mIoU): " << miou << std::endl;

    return 0;
}