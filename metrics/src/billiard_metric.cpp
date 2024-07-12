#include <billiard_metric.h>

/* Librarires required in this source file and not already included in library.h */

/* Matches search */
void bm::matches_search() {
    // Input: true bounding boxes file, predicted bounding boxes file

    // Define the distance function between two bounding boxes rows (x, y, width, height, ball class)
    // ATTENTION: pay attention to the fact the matches are just between same ball classes

    // For each true bounding box:
    // 1. Associate true bounding box with the corresponding predicted bounding box according to the distance function
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