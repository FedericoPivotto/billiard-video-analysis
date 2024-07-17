#include <billiard_metric.h>

/* Librarires required in this source file and not already included in billiard_metric.h */

// iostream: std::cout
#include <iostream>
// numeric: std::accumulate
#include <numeric>
// algorithm: std::max_element
#include <algorithm>

/* Ball match class */
bm::BallMatch::BallMatch(const od::Ball true_ball, const od::Ball predicted_ball) : true_ball(true_ball), predicted_ball(predicted_ball) {
    // Set values
    set_iou();
    set_state();
}

/* Set IoU of true and predicted bounding boxes */
void bm::BallMatch::set_iou() {
    // True ball top-left corner and bottom-right corner
    std::pair<double, double> true_tl(true_ball.center().first - true_ball.width / 2.0, true_ball.center().second - true_ball.height / 2.0);
    std::pair<double, double> true_br(true_ball.center().first + true_ball.width / 2.0, true_ball.center().second + true_ball.height / 2.0);

    // Predicted ball top-left corner and bottom-right corner
    std::pair<double, double> predicted_tl(predicted_ball.center().first - predicted_ball.width / 2.0, predicted_ball.center().second - predicted_ball.height / 2.0);
    std::pair<double, double> predicted_br(predicted_ball.center().first + predicted_ball.width / 2.0, predicted_ball.center().second + predicted_ball.height / 2.0);

    // Intersection top-left corner and bottom-right corner
    std::pair<double, double> intersection_tl(std::max(true_tl.first, predicted_tl.first), std::max(true_tl.second, predicted_tl.second));
    std::pair<double, double> intersection_br(std::min(true_br.first, predicted_br.first), std::min(true_br.second, predicted_br.second));

    // Intersection area
    double intersection_width = std::max(0.0, intersection_br.first - intersection_tl.first);
    double intersection_height = std::max(0.0, intersection_br.second - intersection_tl.second);
    double intersection_area = intersection_width * intersection_height;
    
    // Union area
    double true_area = true_ball.width * true_ball.height;
    double predicted_area = predicted_ball.width * predicted_ball.height;
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
    return os << "(" << ball_match.true_ball << " | " << ball_match.predicted_ball << "), " << ball_match.state << ", IoU(" << ball_match.iou << ")";
}

/* Ball matches search */
void bm::matches_search(const std::vector<od::Ball>& true_balls, const std::vector<od::Ball>& predicted_balls, std::vector<bm::BallMatch>& best_ball_matches) {
    // For each true bounding box associate predicted bounding box according to IoU
    for(size_t i = 0; i < true_balls.size(); ++i) {
        // Possible true ball matches
        std::vector<bm::BallMatch> ball_matches;
        for(size_t j = 0; j < predicted_balls.size(); ++j) {
            // Create and store ball match
            bm::BallMatch ball_match(true_balls[i], predicted_balls[j]);
            ball_matches.push_back(ball_match);
        }

        // Sort ball matches in descending order of IoU
        std::sort(ball_matches.begin(), ball_matches.end(), [](const bm::BallMatch& bm1, const bm::BallMatch& bm2) { return bm1.iou > bm2.iou; });

        // Select the match with maximum IoU if predicted ball is not in a best match
        for(bm::BallMatch ball_match : ball_matches) {
            // Check if predicted ball is in a best match
            bool is_predicted_ball_present = false;
            for(bm::BallMatch best_ball_match : best_ball_matches) {
                is_predicted_ball_present = ball_match.predicted_ball == best_ball_match.predicted_ball;
                if(is_predicted_ball_present)
                    break;
            }

            // Add ball if predicted ball is not in a best match
            if(! is_predicted_ball_present && ball_match.state == bm::TP) {
                best_ball_matches.push_back(ball_match);
                break;
            }
        }
    }
}

/* Compute class mean Average Precision */
double bm::average_precision(std::vector<bm::BallMatch>& ball_matches, std::vector<od::Ball>& predicted_balls, int total_ground_truths) {
    // Collect predicted balls not matched which are FP
    std::vector<od::Ball> predicted_balls_not_matched;
    for(od::Ball predicted_ball : predicted_balls) {
        bool is_predicted_ball_matched = false;
        for(bm::BallMatch ball_match : ball_matches) {
            is_predicted_ball_matched = predicted_ball.id == ball_match.predicted_ball.id;
            if(is_predicted_ball_matched)
                break;
        }

        if(! is_predicted_ball_matched)
            predicted_balls_not_matched.push_back(predicted_ball);
    }

    // Print predicted balls not matched
    for(od::Ball predicted_ball : predicted_balls_not_matched)
        std::cout << "Predicted ball not matched: " << predicted_ball << std::endl;

    // For each predicted bounding box not matched add a FP ball match
    std::vector<bm::BallMatch> ball_matches_copy(ball_matches);
    for(od::Ball predicted_ball : predicted_balls_not_matched) {
        // Create ball not matched forced to FP
        bm::BallMatch ball_match(od::Ball(), predicted_ball);
        ball_match.state = bm::FP;
        // Add ball not matched
        ball_matches_copy.push_back(ball_match);
    }

    // Sort ball matches in descending order of the confidence value
    std::sort(ball_matches_copy.begin(), ball_matches_copy.end(), [](const bm::BallMatch& bm1, const bm::BallMatch& bm2) { return bm1.predicted_ball.confidence > bm2.predicted_ball.confidence; });

    // Cumulative precision and recall vectors
    int cumulative_tp = 0, cumulative_fp = 0;
    std::vector<double> cumulative_precision, cumulative_recall;

    // For each predicted bounding box in ball matches
    for(bm::BallMatch ball_match_copy : ball_matches_copy) {
        // Update cumulative TP and cumulative FP according to ball match
        ball_match_copy.state == bm::TP ? ++cumulative_tp : ++cumulative_fp;

        // Compute and store cumulative precision and cumulative recall:
        cumulative_precision.push_back(static_cast<double>(cumulative_tp) / (cumulative_tp + cumulative_fp));
        cumulative_recall.push_back(static_cast<double>(cumulative_tp) / total_ground_truths);

        // Print cumulative_tp and total_ground_truths
        /* std::cout << cumulative_tp << " " << total_ground_truths << std::endl;*/
    }

    // Vector of precision and recall values
    std::vector<std::pair<double, double>> recall_sorted_precision;
    for(size_t i = 0; i < cumulative_recall.size(); ++i)
        recall_sorted_precision.push_back(std::pair(cumulative_recall[i], cumulative_precision[i]));

    // Sort recall sorted_precision according to recall values
    std::sort(recall_sorted_precision.begin(), recall_sorted_precision.end(), [](std::pair<double, double> &p1, std::pair<double, double> &p2) { return p1.first < p2.first; });

    // Print confidence, precision and recall of each predicted ball
    // TODO: to remove
    /*for(size_t i = 0; i < cumulative_precision.size(); ++i)
        std::cout << ball_matches_copy[i].predicted_ball.confidence << " P: " << cumulative_precision[i] << " R: " << cumulative_recall[i] << std::endl;*/

    // Print recall sorted_precision
    // TODO: to remove
    /*for(std::pair<double, double> p : recall_sorted_precision)
        std::cout << p.first << " " << p.second << std::endl;*/

    // Vector of 11 double number associated to recalla values
    std::vector<double> interpolated_precisions(11, 0.0);
    for(size_t i = 0; i < interpolated_precisions.size(); ++i) {
        // Recall value to consider
        double recall = i / 10.0;

        // Assign the highest cumulative precision next to recall
        // TODO: understand if PASCAL VOC 11 wants < or <=
        size_t j = 0; for(; j < cumulative_recall.size() && cumulative_recall[j] < recall; ++j);
        interpolated_precisions[i] = j < cumulative_recall.size() ? *std::max_element(cumulative_precision.begin() + j, cumulative_precision.end()) : 0;
    }

    // Print interpolated precisions
    // TODO: to remove
    /*std::cout << std::endl;
    double rec = 0.0;
    for(double inter : interpolated_precisions) {
        std::cout << "P: " << inter << " R:" << rec << std::endl;
        rec += 0.1;
    }*/

    // Compute Average Precision (AP)
    double interpolated_precision = std::accumulate(interpolated_precisions.begin(), interpolated_precisions.end(), 0.0);
    double ap = interpolated_precision / 11;

    return ap;
}

/* Localization metric */
double bm::localization_metric(std::vector<double>& aps, int num_classes) {
    // Compute the mean Average Precision (mAP)
    return std::accumulate(aps.begin(), aps.end(), 0.0) / num_classes;
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