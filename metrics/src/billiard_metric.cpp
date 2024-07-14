#include <billiard_metric.h>

/* Librarires required in this source file and not already included in library.h */
#include <iostream>
#include <algorithm>

/* Distance function to compute the correspondences*/
void bm::distance_function(const od::Ball &true_ball, std::vector<od::Ball> predicted_balls){

    //for(od::Ball predicted_ball : predicted_balls) {
    //    std::cout << predicted_ball << " " << predicted_ball.confidence << std::endl;        
    //}

    float min_distance = std::numeric_limits<float>::max();
    od::Ball* closest_ball = nullptr;
    std::vector<std::pair<od::Ball, od::Ball>> ball_pairs;

    for(od::Ball predicted_ball : predicted_balls) {
        if (predicted_ball.ball_class == true_ball.ball_class) {
            float distance = std::sqrt(std::pow(true_ball.x - predicted_ball.x, 2) + std::pow(true_ball.y - predicted_ball.y, 2));
            if (distance < min_distance) {
                min_distance = distance;
                closest_ball = &predicted_ball;
            }
            ball_pairs.push_back(std::make_pair(true_ball, *closest_ball));
        }       
    }

    for (const auto& pair : ball_pairs) {
        std::cout << std::endl;
        std::cout << "Pair: " << std::endl;
        std::cout << pair.first << std::endl;
        std::cout << pair.second << std::endl;
        std::cout << " " << std::endl;
    }
}

/* Matches search */
void bm::matches_search(const std::vector<od::Ball>& predicted_balls, const std::vector<od::Ball>& true_balls) {
    // Input: true bounding boxes file, predicted bounding boxes file

    // Define the distance function between two bounding boxes rows (x, y, width, height, ball class)
    // ATTENTION: pay attention to the fact the matches are just between same ball classes

    // For each true bounding box:
    // 1. Associate true bounding box with the corresponding predicted bounding box according to the distance function
    for (std::size_t i = 0; i < true_balls.size(); ++i) {
        //std::cout << true_balls[i] << " ";
        //std::cout << std::endl;
        distance_function(true_balls[i], predicted_balls);
    }
    // 2. Compute corresponding IoU
    //    * IoU = intersected area / union area
    // 3. Determine if TP (above IoU threshold 0.5) or TN (below IoU threshold 0.5) with IoU threshold 0.5
    
    // For each true bounding box not matched, assign IoU=0 (below IoU threshold 0.5) since FP (does not exist a corresponding predicted bounding box)

    // For each predicted bounding box not matched, assign IoU=0 since FP (does not exist a corresponding true bounding box)

    // NOTE: for each predicted bounding box store label it if TP, TN, FP, FN
    
    // Sort the matches found in descending order of the confidence value

    // Output: sorted matches
}

/* Localization metric */
void bm::localization_metric() {
    // Input: sorted matches
    
    // Initialize to 0 cumulative TP and cumulative FP

    // For each bounding box matched in the sorted vector:
    // 1. Update cumulative TP and cumulative FP according to the current match
    // 2. Compute and store cumulative precision and cumulative recall:
    //    * Cumulative precision = cumulative TP / (cumulative TP + cumulative FP)
    //    * Cumulative recall = cumulative TP / (cumulative TP + cumulative FN)

    // Sort cumulative precision values

    // Sort cumulative recall values

    // For each class:
    // 1. Compute Average Precision (AP) according to PASCAL VOC 11 point interpolation
    //    * Create a vector of 11 double number (in order, each cell is associated to the recall values 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    //    * Assign to each element of the vector the highest cumulative precision value between those with the a higher or equal value to the sorted cumulative recall values
    //    * Compute interpolated_precision as the sum of the values of this resulting vector (11 values in total)
    //    * AP = 1 / (11 * interpolated_precision)

    // Compute the mean Average Precision (mAP):
    //    * mAP = (sum of the computed AP's) / number_of_classes

    // Output: computed mAP
}

/* Segmentation metric */
void bm::segmentation_metric() {
    // Input: sorted matches

    // For each class:
    // 1. For each match:
    //    * Compute the IoU considering the areas of the inscribed circle in the true bounding box and in the predicted bounding box
    // 2. Compute the average of the computed IoU's

    // Consider the corresponding segmentation mask frame from the directory ../system/result generate by our system by assigning to each class the greyscale level of the id od the class itself (reason why we were not able to se nothing... i understood this while writing, lol)
    
    // Gather the corresponding ground truth mask from the dataset
    
    // Compute the intersection area between the two table areas identified by the grayscale level 5
    // Compute the union area between the two table areas identified by the grayscale level 5

    // Compute the table IoU considered these two area values

    // Output: ball mIoU, table IoU
}