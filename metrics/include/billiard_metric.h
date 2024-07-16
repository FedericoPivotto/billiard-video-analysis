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
            BallMatch(od::Ball true_ball, od::Ball predicted_ball);

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
    void matches_search(std::vector<od::Ball>& true_balls, std::vector<od::Ball>& predicted_balls);

    // Metric function declarations
    void localization_metric();
    void segmentation_metric();
}

#endif // BILLIARD_METRIC_H