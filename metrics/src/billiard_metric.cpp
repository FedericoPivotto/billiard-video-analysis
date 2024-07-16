#include <billiard_metric.h>

/* Librarires required in this source file and not already included in billiard_metric.h */

// iostream: std::cout
#include <iostream>

/* Ball match class */
bm::BallMatch::BallMatch(const od::Ball true_ball, const od::Ball predicted_ball) : true_ball(true_ball), predicted_ball(predicted_ball) {
    // Set values
    set_distance();
    set_iou();
    set_state();
}

/* Set distance of true and predicted bounding boxes */
// TODO: alter to get it independent from the order of true balls and predicted balls vectors
// NOTE: define the distance function between two bounding boxes rows (x, y, width, height, ball class)
void bm::BallMatch::set_distance() {
    // Distance value
    distance = std::numeric_limits<double>::infinity();
    
    // Check ball class
    if(true_ball.ball_class != predicted_ball.ball_class)
        return;
    
    // Compute ball distance contribute
    double geometric_distance = std::sqrt(std::pow(true_ball.center().first - predicted_ball.center().first, 2) + std::pow(true_ball.center().second - predicted_ball.center().second, 2));

    // Compute bounding box difference contribute
    int width_diff = true_ball.width - predicted_ball.width;
    int height_diff = true_ball.height - predicted_ball.height;

    // Compute distance as sum of all contributes
    distance = geometric_distance + std::abs(width_diff) + std::abs(height_diff);
}

/* Set IoU of true and predicted bounding boxes */
void bm::BallMatch::set_iou() {
    // True ball top-left corner
    int true_tl_x = true_ball.center().first - std::round(true_ball.width / 2.0);
    int true_tl_y = true_ball.center().second - (true_ball.height / 2.0);
    // True ball bottom-right corner
    int true_br_x = true_ball.center().first + (true_ball.width / 2.0);
    int true_br_y = true_ball.center().second + (true_ball.height / 2.0);

    // Predicted ball top-left corner
    int predicted_tl_x = predicted_ball.center().first - (predicted_ball.width / 2.0);
    int predicted_tl_y = predicted_ball.center().second - (predicted_ball.height / 2.0);
    // Predicted ball bottom-right corner
    int predicted_br_x = predicted_ball.center().first + (predicted_ball.width / 2.0);
    int predicted_br_y = predicted_ball.center().second + (predicted_ball.height / 2.0);

    // Intersection top-left corner
    int intersection_tl_x = std::max(true_tl_x, predicted_tl_x);
    int intersection_tl_y = std::max(true_tl_y, predicted_tl_y);
    // Intersection bottom-right corner
    int intersection_br_x = std::min(true_br_x, predicted_br_x);
    int intersection_br_y = std::min(true_br_y, predicted_br_y);

    // Bounding boxes area
    int true_area = true_ball.width * true_ball.height;
    int predicted_area = predicted_ball.width * predicted_ball.height;

    // Intersection width and height
    unsigned int intersection_width = std::max(0, intersection_br_x - intersection_tl_x);
    unsigned int intersection_height = std::max(0, intersection_br_y - intersection_tl_y);

    // Intersection area
    double intersection_area = intersection_width * intersection_height;
    // Union area
    double union_area = true_area + predicted_area - intersection_area;

    // Compute IoU
    iou = intersection_area / union_area;
}

/* Set state of true and predicted bounding boxes */
void bm::BallMatch::set_state() {
    // Compute state value
    state = iou >= bm::IOU_THRESHOLD ? bm::TP : bm::FP;
}

/* Ball match operator << overload */
std::ostream& bm::operator<<(std::ostream& os, const bm::BallMatch& ball_match) {
    // Ball match information string
    return os << "(" << ball_match.true_ball << " | " << ball_match.predicted_ball << "), " << ball_match.state << ", IoU(" << ball_match.iou << "), Distance(" << ball_match.distance << ")";
}

/* Ball matches search */
void bm::matches_search(const std::vector<od::Ball>& true_balls, const std::vector<od::Ball>& predicted_balls, std::vector<bm::BallMatch>& best_ball_matches) {
    // For each true bounding box associate predicted bounding box according to distance function
    for(size_t i = 0; i < true_balls.size(); ++i) {
        // Possible true ball matches
        std::vector<bm::BallMatch> ball_matches;
        for(size_t j = 0; j < predicted_balls.size(); ++j) {
            // Create and store ball match
            bm::BallMatch ball_match(true_balls[i], predicted_balls[j]);
            ball_matches.push_back(ball_match);
        }

        // Sort ball matches in ascending order of the distance value
        std::sort(ball_matches.begin(), ball_matches.end(), [](const bm::BallMatch& bm1, const bm::BallMatch& bm2) { return bm1.distance < bm2.distance; });

        // Select the match with minimum distance if predicted ball is not in a best match
        for(bm::BallMatch ball_match : ball_matches) {
            // Check if predicted ball is in a best match
            bool is_predicted_ball_present = false;
            for(bm::BallMatch best_ball_match : best_ball_matches) {
                is_predicted_ball_present = ball_match.predicted_ball == best_ball_match.predicted_ball;
                if(is_predicted_ball_present)
                    break;
            }

            // Add ball if predicted ball is not in a best match
            if(! is_predicted_ball_present) {
                best_ball_matches.push_back(ball_match);
                break;
            }
        }
    }
}

/* Localization metric */
void bm::localization_metric(std::vector<bm::BallMatch>& ball_matches) {
    // Sort ball matches in descending order of the confidence value
    std::sort(ball_matches.begin(), ball_matches.end(), [](const bm::BallMatch& bm1, const bm::BallMatch& bm2) { return bm1.predicted_ball.confidence > bm2.predicted_ball.confidence; });

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