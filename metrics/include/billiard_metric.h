#ifndef BILLIARD_METRIC_H
#define BILLIARD_METRIC_H

/* Libraries required in this source file */
// object_detection: od::Ball
#include <object_detection.h>

/* Billiard metric namespace */
namespace bm {
    // Class declarations
    class BallMatch {
        public:
            // Ball match information
            od::Ball true_ball, predicted_ball;
            std::string state;
            double distance;

            // Constructor
            BallMatch(od::Ball true_ball, od::Ball predicted_ball, distance);

            // Function declarations
            void set_state();
    
            // Operator overload declarations
            friend std::ostream& operator<<(std::ostream& os, const Ball& ball);
    };

    // Auxiliary function declaration
    double distance_function(const od::Ball true_ball, const od::Ball predicted_ball);
    void matches_search(std::vector<od::Ball>& true_balls, std::vector<od::Ball>& predicted_balls);

    // Metric function declarations
    void localization_metric();
    void segmentation_metric();
}

#endif // BILLIARD_METRIC_H