// Setup CMake: mkdir build && cd build && cmake ..
// Compile with CMake: cd build && make

// Compile: g++ federico.cpp -o federico -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_core -lopencv_imgcodecs
// Execute: ./federico

#include <iostream>
#include <opencv2/highgui.hpp>

// user-defined libraries
#include <object_detection.h>

// error constants
#define INVALID_ARGUMENTS_ERROR -1
#define IMAGE_READ_ERROR -2

int main(int argc, char** argv) {
    // safety check on argc
	if(argc < 2) {
		std::cout << "Warning: An image filename shall be provided." << std::endl;
		exit(INVALID_ARGUMENTS_ERROR);
	}

    // load image
    std::string img_filename(argv[1]);
    cv::Mat img = cv::imread(img_filename);

    // safety check on image
	if(img.data == NULL) {
		std::cout << "Error: The image cannot be read." << std::endl;
		exit(IMAGE_READ_ERROR);
	}

	// show image
    cv::namedWindow("Image");
    cv::imshow("Image", img);
    cv::waitKey(0);

    // library function call
    int out = function_name();
    std::cout << "function_name(): " << out << std::endl;

    return 0;
}