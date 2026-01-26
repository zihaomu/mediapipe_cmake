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


void draw(const Mat& img, const PointList3f& points, const Scalar color = Scalar(0, 255, 0))
{
    int w = img.cols;
    int h = img.rows;

    for (int k = 0; k < points.size(); k++)
    {
        std::string scoreText = std::to_string(k);
        auto p = Point2f(points[k].x, points[k].y);
        circle(img, p, 0.5, color, 2);
        // circle(img, p, 0.5, Scalar(255, 0, 255), 2);
    }
}

std::string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string test_data_path = root_path + "data/";

void test_image()
{
    string imgPath = test_data_path + "face_image/face0.jpg";
    Mat img = imread(imgPath);

    //    string modelPath = "/Users/mzh/work/my_project/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
    string modelPathIris = root_path + "iris_tracker/models/iris_landmark.mnn";
    string modelPathFaceDetect = root_path + "face_detector/models/face_detection_full_range.mnn";
    string modelPathFaceLandmark = root_path + "face_landmarker/models/face_landmark468.mnn";
    IrisTracker tracker(modelPathIris, modelPathFaceDetect, modelPathFaceLandmark);

    std::vector<IrisOutput> outs;
    tracker.runImage(img, outs);

    if (!outs.empty())
    {
        if (outs[0].eyeType != INVALID_EYE)
        {
            draw(img, outs[0].landmarkEye, Scalar(0, 255, 0));
            draw(img, outs[0].landmarkIris, Scalar(0, 0, 255));
        }
        
        if (outs[1].eyeType != INVALID_EYE)
        {
            draw(img, outs[1].landmarkEye, Scalar(0, 255, 0));
            draw(img, outs[1].landmarkIris, Scalar(0, 0, 255));
        }
    }

    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string modelPathIris = root_path + "iris_tracker/models/iris_landmark.mnn";
    string modelPathFaceDetect = root_path + "face_detector/models/face_detection_full_range.mnn";
    string modelPathFaceLandmark = root_path + "face_landmarker/models/face_landmark468.mnn";
    IrisTracker tracker(modelPathIris, modelPathFaceDetect, modelPathFaceLandmark);

    Mat img_original;
    std::string video_path = test_data_path + "video/head_move.mp4";
    VideoCapture cap(video_path);

    std::vector<IrisOutput> outs;
    TickMeter t;

    while (1)
    {
        t.reset();
        cap >> img_original;

        if (img_original.empty())
            break;

        Mat img;
        // resize img to 640x480 while keeping h/w ratio
        resizeUnscale(img_original, img, 640, 480);

        t.start();

        tracker.runVideo(img, outs);

        t.stop();
               
        // Display FPS
        string fps_text = "Time: " + to_string(t.getTimeMilli()) + " ms";
        putText(img, fps_text, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        if (!outs.empty())
        {
            if (outs[0].eyeType != INVALID_EYE)
            {
                draw(img, outs[0].landmarkEye, Scalar(0, 255, 0));
                draw(img, outs[0].landmarkIris, Scalar(0, 0, 255));
            }
            
            if (outs[1].eyeType != INVALID_EYE)
            {
                draw(img, outs[1].landmarkEye, Scalar(0, 255, 0));
                draw(img, outs[1].landmarkIris, Scalar(0, 0, 255));
            }
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
