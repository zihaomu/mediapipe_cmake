//
// Created by mzh on 2023/10/11.
//
#include <opencv2/opencv.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "interactive_segmenter.h"

using namespace mpp;
using namespace cv;

cv::Mat image;
bool drawing = false;
cv::Point prevPoint;

bool runSeg = false;

InterativeSegmenter segmenter("/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/interactive_segmenter/models/magic_touch.mnn");

std::vector<cv::Point> mouseClick;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (runSeg == false)
    {
        if (event == cv::EVENT_LBUTTONDOWN) {
            drawing = true;
            prevPoint = cv::Point(x, y);
            cv::circle(image, prevPoint, 2, cv::Scalar(0, 0, 255));
            cv::imshow("Image", image);
            mouseClick.push_back(prevPoint);
        }
        else if (event == cv::EVENT_MOUSEMOVE && drawing) {
            cv::Point currentPoint(x, y);
            cv::line(image, prevPoint, currentPoint, cv::Scalar(0, 0, 255), 2);
            prevPoint = currentPoint;
            cv::imshow("Image", image);
            mouseClick.push_back(prevPoint);
        }
        else if (event == cv::EVENT_LBUTTONUP) {
            drawing = false;
            runSeg = true;
//            for (int i = 0; i < mouseClick.size(); i++)
//            {
//                std::cout<<mouseClick[i]<<std::endl;
//            }
        }
    }
}

//[423, 271]
void test()
{
    image = cv::imread("/Users/mzh/work/my_project/mediapipe_cmake/data/body_image/test3.jpeg");
//    dnn::printMatShape(image);
//    resize(image, image, {512, 512});
//    image.resize((512, 512));
    cv::Mat mask;
    cv::Mat out;

//    std::vector<cv::Point> mouseClick2 = {cv::Point(337, 337)};
    std::vector<cv::Point> mouseClick2 = {cv::Point(423, 271)};
//    std::vector<cv::Point> mouseClick2 = {cv::Point(214, 216)};
//    [214, 216]

    segmenter.pointsToAlpha(image, mouseClick2, mask);
    cv::imshow("mask", mask);
    cv::waitKey(0);
    segmenter.run(image, mouseClick2, out);

    cv::imshow("maskOut", out);

    cv::waitKey(0);
    runSeg = false;
}

int main()
{
//    test();

    // Load an image
    image = cv::imread("/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/body_image/test3.jpeg");
//    image = cv::imread("/Users/mzh/work/github/mediapipe_commont/mediapipe/tasks/testdata/vision/cats_and_dogs.jpg");


//    resize(image, image, {512, 512});
    if (image.empty())
    {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    cv::imshow("Image", image);
    cv::setMouseCallback("Image", onMouse, reinterpret_cast<void*>(&image));

    while (true)
    {
        cv::Mat out;
        if (runSeg)
        {
            cv::Mat mask;
            segmenter.pointsToAlpha(image, mouseClick, mask);
            segmenter.run(image, mouseClick, out);

            cv::imshow("maskOut", out);
            runSeg = false;
        }
        // TODO drawing the out.

        int key = cv::waitKey(1);
        if (key == 27)  // 27 is the ASCII code for the 'ESC' key
            break;
    }

//    cv::imwrite("output.jpg", image); // Save the edited image
    cv::destroyAllWindows();

    return 0;
}
