#ifndef BILLIARD_METRIC_H
#define BILLIARD_METRIC_H

/* Libraries required in this source file */

// object_detection: od::Ball
#include <object_detection.h>

/* Billiard metric namespace */
namespace bm {
    // Constant declaration
    const double IOU_THRESHOLD = 0.5;
    enum State {TP, TN, FP, FN};

    // Class declaration
    class BallMatch {
        public:
            // Ball match attributes
            od::Ball true_ball, predicted_ball;
            double distance, iou;

            // Ball match state
            State state;

            // Constructor
            BallMatch(const od::Ball true_ball, const od::Ball predicted_ball);

            // Operator overload declarations
            friend std::ostream& operator<<(std::ostream& os, const bm::BallMatch& ball_match);
        
        private:
            // Function declarations
            void set_distance();
            void set_iou();
            void set_state();
    };

    // Definition of the operator<< function outside the class
    std::ostream& operator<<(std::ostream& os, const BallMatch& ball_match);

    // Auxiliary function declarations
    void matches_search(const std::vector<od::Ball>& true_balls, const std::vector<od::Ball>& predicted_balls, std::vector<bm::BallMatch>& best_ball_matches);
    double average_precision(int total_ground_truths, std::vector<bm::BallMatch>& ball_matches);

    // Metric function declarations
    double localization_metric(std::vector<double>& aps, int num_classes);
    void segmentation_metric();
}

#endif // BILLIARD_METRIC_H