#include <billiard_metric.h>

/* Librarires required in this source file and not already included in billiard_metric.h */

// iostream: std::cout
#include <iostream>
// numeric: std::accumulate
#include <numeric>
// algorithm: std::max_element
#include <algorithm>
// imgproc: cv::threshold
#include <opencv2/imgproc.hpp>

/* Ball match class */
bm::BallMatch::BallMatch(const od::Ball true_ball, const od::Ball predicted_ball) : true_ball(true_ball), predicted_ball(predicted_ball) {
    set_iou(); set_state();
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
    std::vector<bm::BallMatch> ball_matches_copy(ball_matches);
    for(od::Ball predicted_ball : predicted_balls) {
        bool is_predicted_ball_matched = false;
        for(bm::BallMatch ball_match : ball_matches) {
            is_predicted_ball_matched = predicted_ball.id == ball_match.predicted_ball.id;
            if(is_predicted_ball_matched)
                break;
        }

        if(! is_predicted_ball_matched) {
            // Create ball not matched forced to FP
            bm::BallMatch ball_match(od::Ball(), predicted_ball);
            ball_match.state = bm::FP;
            // Add ball not matched
            ball_matches_copy.push_back(ball_match);
        }
    }

    // Sort ball matches in descending order of the confidence value
    std::sort(ball_matches_copy.begin(), ball_matches_copy.end(), [](const bm::BallMatch& bm1, const bm::BallMatch& bm2) { return bm1.predicted_ball.confidence > bm2.predicted_ball.confidence; });

    // Collect cumulative precision and recall vectors
    int cumulative_tp = 0, cumulative_fp = 0;
    std::vector<double> cumulative_precision, cumulative_recall;
    for(bm::BallMatch ball_match_copy : ball_matches_copy) {
        // Update cumulative TP and cumulative FP according to ball match
        ball_match_copy.state == bm::TP ? ++cumulative_tp : ++cumulative_fp;

        // Compute and store cumulative precision and cumulative recall:
        cumulative_precision.push_back(static_cast<double>(cumulative_tp) / (cumulative_tp + cumulative_fp));
        cumulative_recall.push_back(static_cast<double>(cumulative_tp) / total_ground_truths);
    }

    // Vector of precision and recall values
    std::vector<std::pair<double, double>> recall_sorted_precision;
    for(size_t i = 0; i < cumulative_recall.size(); ++i)
        recall_sorted_precision.push_back(std::pair(cumulative_recall[i], cumulative_precision[i]));

    // Sort recall sorted_precision according to recall values
    std::sort(recall_sorted_precision.begin(), recall_sorted_precision.end(), [](std::pair<double, double> &p1, std::pair<double, double> &p2) { return p1.first < p2.first; });

    // Vector of 11 double number associated to recalla values
    std::vector<double> interpolated_precisions(11, 0.0);
    for(size_t i = 0; i < interpolated_precisions.size(); ++i) {
        // Recall value to consider
        double recall = i / 10.0;

        // Assign the highest cumulative precision next to recall
        size_t j = 0; for(; j < cumulative_recall.size() && cumulative_recall[j] < recall; ++j);
        interpolated_precisions[i] = j < cumulative_recall.size() ? *std::max_element(cumulative_precision.begin() + j, cumulative_precision.end()) : 0;
    }

    // Compute Average Precision (AP)
    double interpolated_precision = std::accumulate(interpolated_precisions.begin(), interpolated_precisions.end(), 0.0);
    double ap = interpolated_precision / 11;

    return ap;
}

/* Localization metric */
double bm::localization_metric(const std::vector<double>& aps, const int num_classes) {
    // Compute the mean Average Precision (mAP)
    return std::accumulate(aps.begin(), aps.end(), 0.0) / num_classes;
}

/* Compute IoU of given class */
double bm::iou_class(cv::Mat& true_mask, cv::Mat& predicted_mask, int class_id) {
    // Binary true and predicted masks for class id
    cv::Mat mask1, mask2;
    cv::inRange(true_mask, class_id, class_id, mask1);
    cv::inRange(predicted_mask, class_id, class_id, mask2);

    // Intersection mask between true and predicted masks
    cv::Mat intersection_mask, union_mask;
    cv::bitwise_and(mask1, mask2, intersection_mask);
    cv::bitwise_or(mask1, mask2, union_mask);
    
    // Show masks
    /*cv::imshow("True mask", mask1);
    cv::imshow("Predicted mask", mask2);
    cv::imshow("Intersection mask", intersection_mask);
    cv::imshow("Union mask", union_mask);
    cv::waitKey(0);*/

    // Compute IoU as ratio of intersection area to union area
    double intersection_area = cv::countNonZero(intersection_mask);
    double union_area = cv::countNonZero(union_mask);
    double iou = union_area != 0 ? intersection_area / union_area : 0;

    return iou;
}

/* Segmentation metric */
double bm::segmentation_metric(std::vector<double> ious, const int num_classes) {
    // Compute average of given IoU
    return std::accumulate(ious.begin(), ious.end(), 0.0) / num_classes;
}