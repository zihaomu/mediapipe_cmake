//
// Created by mzh on 2023/8/21.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "pose_detector.h"

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

void test_image()
{
    string imgPath = "/Users/mzh/work/opencv_dev/mediapipe_reproduce/data/body_image/test3.jpeg";
    Mat img = imread(imgPath);

    string modelPath = "/Users/mzh/work/opencv_dev/mediapipe_reproduce/pose_detector/models/pose_detection.mnn";
    PoseDetector detector(modelPath);

    std::vector<BoxKp2> boxOut = {};
    detector.run(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{

    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.mnn";
//    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.tflite";
    PoseDetector detector(modelPath);

    VideoCapture cap("/Users/mzh/work/data/video/v0.mp4");
//    VideoCapture cap(0);
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
