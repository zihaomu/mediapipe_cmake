//
// Created by mzh on 2023/8/29.
//
#include "opencv2/opencv.hpp"
#include "face_landmark.h"

using namespace cv;
using namespace mpp;
using namespace std;

void draw(const Mat& img, const std::vector<BoxKp3>& box)
{
    int w = img.cols;
    int h = img.rows;

    for (int i = 0; i < box.size(); i++)
    {
        rectangle(img, box[i].rect, Scalar(255, 0, 0), 2);
//        std::string scoreText = std::to_string(box[i].score);
//        putText(img, scoreText, cv::Point(box[i].rect.x * w, box[i].rect.y * h), 2, 1, cv::Scalar(0, 255, 0));

        for (int k = 0; k < box[i].points.size(); k++)
        {
            std::string scoreText = std::to_string(k);
            auto p = Point2f(box[i].points[k].x * w, box[i].points[k].y * h);
//            putText(img, scoreText, p, 2, 1, cv::Scalar(0, 255, 0));
            circle(img, p, 0.5, Scalar(255, 0, 255), 2);
        }
    }
}

void test_image()
{
    string imgPath = "/Users/mzh/work/my_project/mediapipe_cmake/data/face_image/face0.jpg";
    Mat img = imread(imgPath);

    string detectModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_full_range.mnn";
    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_landmarker/models/face_landmark468.mnn";
    FaceLandmarker landmarker(detectModelPath, landmarkModelPath, 2);
    std::vector<BoxKp3> boxOut = {};
    landmarker.runImage(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string detectModelPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string landmarkModelPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_landmarker/models/face_landmark468.mnn";
    FaceLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::vector<BoxKp3> boxOut = {};
    VideoCapture cap(0);
    TickMeter t;
    Mat frame;
    while (1)
    {
        t.reset();
        cap >> frame;

        t.start();
        landmarker.runVideo(frame, boxOut); // video mode, use hand tracking for speeding up.
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
    return 0;
}