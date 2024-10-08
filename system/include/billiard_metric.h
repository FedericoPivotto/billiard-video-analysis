/* Leonardo Egidati */

#ifndef BILLIARD_METRIC_H
#define BILLIARD_METRIC_H

/* User-defined libraries required */
#include <object_detection.h>

/* Billiard metric namespace */
namespace bm {
    // Constant declarations
    const double IOU_THRESHOLD = 0.5;
    enum State {TP, TN, FP, FN};

    // Class declaration
    class BallMatch {
        public:
            // Ball match attributes
            od::Ball true_ball, predicted_ball;
            double iou;

            // Ball match state
            State state;

            // Constructor declaration
            BallMatch(const od::Ball true_ball, const od::Ball predicted_ball);

            // Operator overload declaration
            friend std::ostream& operator<<(std::ostream& os, const bm::BallMatch& ball_match);
        
        private:
            // Function declarations
            void set_iou();
            void set_state();
    };

    // Operator overload declaration outside the class
    std::ostream& operator<<(std::ostream& os, const BallMatch& ball_match);

    // Evaluate function declarations
    void evaluate_localization_metric(const std::string true_bboxes_frame_file_path, const std::string predicted_bboxes_frame_file_path, std::string& metrics_result);
    void evaluate_segmentation_metric(const std::string true_mask_path, const std::string predicted_mask_path, std::string& metrics_result);

    // Auxiliary function declarations
    void matches_search(const std::vector<od::Ball>& true_balls, const std::vector<od::Ball>& predicted_balls, std::vector<bm::BallMatch>& best_ball_matches);
    double average_precision(std::vector<bm::BallMatch>& ball_matches, std::vector<od::Ball>& predicted_balls, int total_ground_truths);
    double localization_metric(const std::vector<double>& aps, const int num_classes);
    double iou_class(cv::Mat& true_mask, cv::Mat& predicted_mask, int class_id);
    double segmentation_metric(std::vector<double> ious, const int num_classes);
}

#endif // BILLIARD_METRIC_H