//
// Created by mzh on 2023/10/11.
//
#include <opencv2/opencv.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "hair_segmenter.h"
#include "iostream"

using namespace std;
using namespace mpp;
using namespace cv;

void test_image()
{
    string imgPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/face_image/face0.jpg";
    Mat img = imread(imgPath);

    //    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string modelPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/hair_segmenter/models/hair_segmentation.tflite";
    HairSegmenter detector(modelPath);

    Mat out;
    detector.run(img, out);

    std::vector<Mat> outMasks;
    split(out, outMasks);

    imshow("img", img);
    imshow("mask 0", outMasks[0]);
    imshow("mask 1", outMasks[1]);
    waitKey(0);
}


void test_camera()
{
    //    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string modelPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/hair_segmenter/models/hair_segmentation.tflite";
    HairSegmenter detector(modelPath);

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

        std::vector<Mat> outMasks;
        split(outMask, outMasks);

        imshow("frame", frame);
        imshow("mask 0", outMasks[0]);
        imshow("mask 1", outMasks[1]);
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
