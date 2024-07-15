#ifndef BILLIARD_METRIC_H
#define BILLIARD_METRIC_H

/* Libraries required in this source file */
// object_detection: od::Ball
#include <object_detection.h>

/* Billiard metric namespace */
namespace bm {
    // Auxiliary function declaration
    double distance_function(const od::Ball true_ball, const od::Ball predicted_ball);
    void matches_search(std::vector<od::Ball>& true_balls, std::vector<od::Ball>& predicted_balls);

    // Metric function declarations
    void localization_metric();
    void segmentation_metric();
}

#endif // BILLIARD_METRIC_H