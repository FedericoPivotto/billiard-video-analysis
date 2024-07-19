#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

/* Libraries required in this source file */

// utility: std::pair
#include <utility>
// highgui: cv::Mat
#include <opencv2/highgui.hpp>

/* Object detection namespace */
namespace od {
    // Classes declaration
    class Ball {
        private:
            // Static id declaration
            static int current_id;

        public:
            // Ball unique id
            int id;

            // Ball bounding box
            unsigned int x, y, width, height, ball_class;
            double confidence;

            // Constructors
            Ball();
            Ball(unsigned int x, unsigned int y, unsigned int width, unsigned int height, unsigned int ball_class = -1, double confidence = 0);

            // Function declarations
            std::pair<unsigned int, unsigned int> center() const;
            unsigned int radius() const;
            cv::Rect get_rect_bbox();
            void set_rect_bbox(cv::Rect bbox);
    
            // Operator overload declarations
            friend std::ostream& operator<<(std::ostream& os, const Ball& ball);
            friend bool operator==(const Ball& ball1, const Ball& ball2);
    };

    // Definition of the operator<< function outside the class
    std::ostream& operator<<(std::ostream& os, const Ball& ball);
    // Definition of the operator== function outside the class
    bool operator==(const Ball& ball1, const Ball& ball2);

    // Detect function declarations
    void detect_ball_class(Ball& ball_bbox, const int ball_index, std::vector<double>& white_ratios, std::vector<double>& black_ratios, std::vector<double>& magnitude_counts);

    // Ball bounding box confidence function declaration
    void set_ball_bbox_confidence(od::Ball& ball);

    // Object detection main declaration
    void object_detection(const std::vector<cv::Mat>& video_frames, const int n_frame, const std::string bboxes_video_path, const std::vector<cv::Point2f> corners_float, const bool is_distorted, cv::Mat& video_frame, const std::string test_bboxes_video_path = "", const bool test_flag = false);

    // Circle filters declarations
    void suppress_billiard_holes(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f> corners, const bool is_distorted);
    void suppress_small_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_small, const double radius_min);
    void suppress_big_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_big, const double radius_max);
    void suppress_black_circles(std::vector<cv::Vec3f>& circles, cv::Mat mask);
    void compute_mean_circles(std::vector<cv::Vec3f>& circles, std::vector<cv::Vec3f>& circles_mean, const double offset = 0);
    void normalize_circles_radius(std::vector<cv::Vec3f>& circles);

    // Object detection auxiliary functions
    void points_float_to_int(const std::vector<cv::Point2f> float_points, std::vector<cv::Point>& points);
    void preprocess_bgr_frame(const cv::Mat& frame, cv::Mat& preprocessed_video_frame);
    void morpho_pre_process(cv::Mat& mask);

    // Object classification auxiliary functions
    void compute_gradient_balls(const cv::Mat& frame, const std::vector<od::Ball>& ball_bboxes, std::vector<double>& magnitude_counts);
    void compute_gradient_magnitude(const cv::Mat& frame, cv::Mat& gradient);
    void compute_black_white_ratio(const cv::Mat& ball, double& white_ratio, double& black_ratio);
    void compute_color_ratios(std::vector<od::Ball> ball_bboxes, const cv::Mat& frame, std::vector<double>& white_ratios, std::vector<double>& black_ratios);
    void get_best_two_indexes(const std::vector<double>& vec, int& best_index, int& sec_index);
    void normalize_vector(std::vector<double>& vec);
    void overlay_ball_bounding_bbox(cv::Mat& video_frame, od::Ball ball_bbox);
    void detect_white_black_balls(std::vector<od::Ball>& ball_bboxes, int& white_index, int& black_index, const std::vector<double>& white_ratio, const std::vector<double>& black_ratio, std::vector<double>& magnitude_counts);
}

#endif // OBJECT_DETECTION_H