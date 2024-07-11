#include <billiard_metric.h>

// librarires required in this source file and not already included in library.h

/* Matches search */
void bm::matches_search() {
    // define the distance function between two bounding boxes rows (x, y, width, height, ball class)
    // ATTENTION: pay attention to the fact the matches are just between same ball classes

    // for each true bounding box:
    // 1. associate true bounding box with the corresponding predicted bounding box according to the distance function
    // 2. compute corresponding IoU
    //    * IoU = intersected area / union area
    // 3. determine if TP (above IoU threshold 0.5) or TN (below IoU threshold 0.5) with IoU threshold 0.5
    
    // for each true bounding box not matched, assign IoU=0 (below IoU threshold 0.5) since FP (does not exist a corresponding predicted bounding box)

    // for each predicted bounding box not matched, assign IoU=0 since FP (does not exist a corresponding true bounding box)

    // NOTE: for each predicted bounding box store label it if TP, TN, FP, FN
    
    // sort the matches found in descending order of the confidence value
}

/* Localization metric */
void bm::localization_metric() {
    // input: sorted matches
    
    // initialize to 0 cumulative TP and cumulative FP

    // for each bounding box matched in the sorted vector:
    // 1. update cumulative TP and cumulative FP according to the current match
    // 2. compute and store cumulative precision and cumulative recall:
    //    * cumulative precision = cumulative TP / (cumulative TP + cumulative FP)
    //    * cumulative recall = cumulative TP / (cumulative TP + cumulative FN)

    // sort cumulative precision values

    // sort cumulative recall values

    // for each class:
    // 1. compute Average Precision (AP) according to PASCAL VOC 11 point interpolation
    //    * create a vector of 11 double number (in order, each cell is associated to the recall values 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    //    * assign to each element of the vector the highest cumulative precision value between those with the a higher or equal value to the sorted cumulative recall values
    //    * compute interpolated_precision as the sum of the values of this resulting vector (11 values in total)
    //    * AP = 1 / (11 * interpolated_precision)

    // compute the mean Average Precision (mAP):
    //    * mAP = (sum of the computed AP's) / number_of_classes

    // output: computed mAP
}

/* Segmentation metric */
void bm::segmentation_metric() {
    // input: sorted matches

    // for each class:
    // 1. for each match:
    //    * compute the IoU considering the areas of the inscribed circle in the true bounding box and in the predicted bounding box
    // 2. compute the average of the computed IoU's

    // consider the corresponding segmentation mask frame from the directory ../system/result generate by our system by assigning to each class the greyscale level of the id od the class itself (reason why we were not able to se nothing... i understood this while writing, lol)
    
    // gather the corresponding ground truth mask from the dataset
    
    // compute the intersection area between the two table areas identified by the grayscale level 5
    // compute the union area between the two table areas identified by the grayscale level 5

    // compute the table IoU considered these two area values

    // output: ball mIoU, table IoU
}