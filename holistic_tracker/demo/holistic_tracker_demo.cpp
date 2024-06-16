//
// Created by mzh on 2023/10/11.
//
#include <opencv2/opencv.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "holistic_tracker.h"
#include "hand_landmark.h"
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
        auto p = Point2f(points[k].x * w, points[k].y * h);
//            putText(img, scoreText, p, 2, 1, cv::Scalar(0, 255, 0));
        circle(img, p, 1.5, Scalar(255, 0, 255), 2);
    }
}

/// \param pose_detector
/// \param pose_landmarker
/// \param face_detector
/// \param face_landmarker
/// \param hand_recrop
/// \param hand_landmark
/// \param device

const string pose_detector = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/pose_detector/models/pose_detection.mnn";
const string pose_landmarker = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/pose_landmarker/models/pose_landmark_full_sim.mnn";
const string face_detector = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_detector/models/face_detection_short_range.mnn";
const string face_landmarker = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/face_landmarker/models/face_landmark468.mnn";
const string hand_recrop = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/holistic_tracker/models/hand_recrop.mnn";
const string hand_landmak = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/hand_landmarker/models/hand_landmark_full.mnn";

void test_image()
{
    string imgPath = "/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/body_image/stand.jpg";
    Mat img = imread(imgPath);

    HolisticTracker tracker(pose_detector, pose_landmarker, face_detector, face_landmarker, hand_recrop, hand_landmak);

    HolisticOutput output;
    tracker.runImage(img, output);

    if (!output.poseLandmark.empty())
    {
        draw(img, output.poseLandmark[0].points);
        if (!output.rightHandLandmark.empty())
            draw(img, output.rightHandLandmark[0].points);
        if (!output.leftHandLandmark.empty())
            draw(img, output.leftHandLandmark[0].points);

        if (!output.faceLandmark.empty())
            draw(img, output.faceLandmark[0].points);
    }

    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    HolisticTracker tracker(pose_detector, pose_landmarker, face_detector, face_landmarker, hand_recrop, hand_landmak);

    Mat img;
    VideoCapture cap(0);
//    VideoCapture cap("/Users/moo/work/my_project/mediapipe_cmake2/mediapipe_cmake/data/video/dance.mp4");

    TickMeter t;

    HolisticOutput output;
    while (1) {
        t.reset();
        cap >> img;

        t.start();

        tracker.runVideo(img, output);

        t.stop();


        if (!output.poseLandmark.empty())
        {
            draw(img, output.poseLandmark[0].points);
            if (!output.rightHandLandmark.empty())
                drawConnection(img, output.rightHandLandmark[0].points, kHandConnections, img.cols, img.rows);
            if (!output.leftHandLandmark.empty())
                drawConnection(img, output.leftHandLandmark[0].points, kHandConnections, img.cols, img.rows);

            if (!output.faceLandmark.empty())
                draw(img, output.faceLandmark[0].points);
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
