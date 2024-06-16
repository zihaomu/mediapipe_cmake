//
// Created by mzh on 2023/10/11.
//
#include <opencv2/opencv.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "iris_tracker.h"
#include "iostream"

using namespace std;
using namespace mpp;
using namespace cv;


void draw(const Mat& img, const PointList3f& points)
{
    int w = img.cols;
    int h = img.rows;

    for (int k = 0; k < points.size(); k++)
    {
        std::string scoreText = std::to_string(k);
        auto p = Point2f(points[k].x, points[k].y);
//            putText(img, scoreText, p, 2, 1, cv::Scalar(0, 255, 0));
        circle(img, p, 0.5, Scalar(255, 0, 255), 2);
    }
}

void test_image()
{
    string imgPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/face_image/face0.jpg";
    Mat img = imread(imgPath);

    //    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string modelPathIris = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/iris_tracker/models/iris_landmark.mnn";
    string modelPathFaceDetect = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_detector/models/face_detection_full_range.mnn";
    string modelPathFaceLandmark = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_landmarker/models/face_landmark468.mnn";
    IrisTracker tracker(modelPathIris, modelPathFaceDetect, modelPathFaceLandmark);

    std::vector<IrisOutput> outs;
    tracker.runImage(img, outs);

    if (!outs.empty())
    {
        draw(img, outs[0].landmarkEye);
        draw(img, outs[0].landmarkIris);

        draw(img, outs[1].landmarkEye);
        draw(img, outs[1].landmarkIris);
    }

    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string imgPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/face_image/face0.jpg";
//    Mat img = imread(imgPath);

    //    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string modelPathIris = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/iris_tracker/models/iris_landmark.mnn";
    string modelPathFaceDetect = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_detector/models/face_detection_full_range.mnn";
    string modelPathFaceLandmark = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_landmarker/models/face_landmark468.mnn";
    IrisTracker tracker(modelPathIris, modelPathFaceDetect, modelPathFaceLandmark);

    Mat img;
    VideoCapture cap(0);
    Mat outMask;
    std::vector<IrisOutput> outs;
    TickMeter t;
    while (1)
    {
        t.reset();
        cap >> img;

        t.start();

        tracker.runVideo(img, outs);

        t.stop();
        if (!outs.empty())
        {
//            draw(img, outs[0].landmarkEye);
            draw(img, outs[0].landmarkIris);

//            draw(img, outs[1].landmarkEye);
            draw(img, outs[1].landmarkIris);
        }

        imshow("img", img);
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
