//
// Created by mzh on 2023/10/11.
//
#include <opencv2/opencv.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "selfie_segmenter.h"
#include "iostream"

using namespace std;
using namespace mpp;
using namespace cv;



void test_image()
{
    string imgPath = "../data/face_image/face0.jpg";
    Mat img = imread(imgPath);
    string modelPath = "../selfie_segmenter/models/selfie_segmentation.tflite";
    SelfieSegmenter detector(modelPath);

    Mat outMask;
    detector.run(img, outMask);

    imshow("img", img);
    imshow("mask 0", outMask);
    waitKey(0);
}

void test_camera()
{
    string modelPath = "../selfie_segmenter/models/selfie_segmentation.tflite";
    SelfieSegmenter detector(modelPath);

    Mat frame;
    VideoCapture cap(0);
    Mat outMask;

    TickMeter t;
    while (1)
    {
        t.reset();
        cap >> frame;

        t.start();
        detector.run(frame, outMask);
        t.stop();

        std::string scoreText = "time = " + std::to_string(t.getTimeMilli());
        putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 255, 0));

        imshow("img", frame);
        imshow("mask 0", outMask);
        if (waitKey(1) == 27)
            break;
    }
}

int main()
{
//    test_image();
    test_camera();

    return 0;
}
