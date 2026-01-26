//
// Created by mzh on 2023/8/21.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "hand_detector.h"

using namespace cv;
using namespace mpp;
using namespace std;

void draw(const Mat& img, const std::vector<BoxKp2>& box)
{
    for (int i = 0; i < box.size(); i++)
    {
        rectangle(img, box[i].rect, Scalar(255, 0, 0), 2);
        std::string scoreText = std::to_string(box[i].score);
        putText(img, scoreText, cv::Point(box[i].rect.x, box[i].rect.y), 2, 1, cv::Scalar(0, 255, 0));

        for (int k = 0; k < box[i].points.size(); k++)
        {
            std::string scoreText = std::to_string(k);
            putText(img, scoreText, box[i].points[k], 2, 1, cv::Scalar(0, 255, 0));
            circle(img, box[i].points[k], 2, Scalar(255, 0, 255), 3);
        }
    }
}

// ROOT PATH and test data
std::string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string test_data_path = root_path + "data/";


void test_image()
{
    string imgPath = test_data_path + "hand_image/palm_hands.jpeg";
    Mat img = imread(imgPath);

    string modelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    HandDetector detector(modelPath);

    std::vector<BoxKp2> boxOut = {};
    detector.run(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string modelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    HandDetector detector(modelPath);

    VideoCapture cap(0);
    TickMeter t;
    Mat frame;
    while (1)
    {
        t.reset();
        cap >> frame;
        std::vector<BoxKp2> boxOut = {};
        t.start();
        detector.run(frame, boxOut);
        t.stop();

        std::string scoreText = "time = " + std::to_string(t.getTimeMilli());
        putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 255, 0));

        draw(frame, boxOut);
        imshow("img", frame);
        if (waitKey(1) == 27)
            break;
    }
}

int main()
{
//    test_image();
    test_camera();
}
