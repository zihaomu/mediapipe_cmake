//
// Created by mzh on 2023/8/29.
//
#include "opencv2/opencv.hpp"
#include "pose_landmark.h"
#include "common.h"
#include "fstream"
#include "pose_landmark_c_api.h"

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
        for (int k = 0; k < box[i].points.size(); k++)
        {
            std::string scoreText = std::to_string(k);
            auto p = Point2f(box[i].points[k].x * w, box[i].points[k].y * h);
            circle(img, p, 2, Scalar(255, 0, 255), 3);
        }
    }
}

// ROOT PATH
std::string ROOT_PATH = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string DATA_PATH = ROOT_PATH + "data/";

void test_image()
{
    string imgPath = DATA_PATH + "/body_image/test.jpeg";
    Mat img = imread(imgPath);

    string detectModelPath = ROOT_PATH + "pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = ROOT_PATH + "pose_landmarker/models/pose_landmark_full_sim.mnn";
    PoseLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::vector<BoxKp3> boxOut = {};
    landmarker.runImage(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string imgPath2 = ROOT_PATH + "data/hand_image/palm_hands.jpeg";
    Mat img2 = imread(imgPath2);

    string detectModelPath = ROOT_PATH + "pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = ROOT_PATH + "pose_landmarker/models/pose_landmark_full_sim.mnn";
    PoseLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::vector<BoxKp3> boxOut = {};
    VideoCapture cap(ROOT_PATH + "data/video/dance_2.mp4");

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
        if (!boxOut.empty())
        {
            scoreText += ", score = " + std::to_string(boxOut[0].score);
            drawConnection(frame, boxOut[0].points, kPoseLandmarksConnections, frame.cols, frame.rows);
        }

        putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 0, 255));

        draw(frame, boxOut);
        imshow("Pose Landmark by mpp_cmake", frame);
        if (waitKey(1) == 27)
            break;
    }
}

void loadModel2Buffer(string modelPath, char*& buffer, long& fileSize)
{
    std::ifstream file_detect(modelPath, std::ios::binary);

    // Find the length of the file
    file_detect.seekg(0, std::ios::end);
    fileSize = file_detect.tellg();
    file_detect.seekg(0, std::ios::beg);

    // Allocate memory for the buffer
    buffer = new char[fileSize];

    // Read file content into the buffer
    file_detect.read(buffer, fileSize);

    // Close the file
    file_detect.close();
}

void test_camera_c_api()
{
    string imgPath2 = ROOT_PATH + "data/hand_image/palm_hands.jpeg";
    Mat img2 = imread(imgPath2);

    string detectModelPath = ROOT_PATH + "pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = ROOT_PATH + "pose_landmarker/models/pose_landmark_full_sim.mnn";

    initPoseLandmarker();

    // New: load models directly from file path
    loadModelPoseDetectFromFile(detectModelPath.c_str(), 0);
    loadModelPoseLandmarkFromFile(landmarkModelPath.c_str(), 0);

    VideoCapture cap(ROOT_PATH + "data/video/dance_2.mp4");
    TickMeter t;
    Mat frame;

    PoseLandmarkResult* resut = new PoseLandmarkResult();
    while (1)
    {
        t.reset();
        cap >> frame;

        t.start();

        runPoseLandmark((const char *)frame.data, frame.cols, frame.rows, frame.step, 0, -1, 1, resut);
        t.stop();

        int w = frame.cols;
        int h = frame.rows;

        std::string scoreText = "time = " + std::to_string(t.getTimeMilli());
        if (resut->poseNum == 1)
        {
            rectangle(frame, {resut->rect[0], resut->rect[1],resut->rect[2],resut->rect[3]}, Scalar(255, 0, 0), 2);

            for (int k = 0; k < 33; k++)
            {
                std::string scoreText = std::to_string(k);
                auto p = Point2f(resut->points[k * 2] * w, resut->points[k * 2+1] * h);
                circle(frame, p, 2, Scalar(255, 0, 255), 3);
            }
        }

        putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 0, 255));

        imshow("Pose Landmark by mpp_cmake", frame);
        if (waitKey(1) == 27)
            break;
    }

    releasePoseLandmark();
}

void test_img_c_api()
{
    string detectModelPath = ROOT_PATH + "pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = ROOT_PATH + "pose_landmarker/models/pose_landmark_full_sim.mnn";

    initPoseLandmarker();

    char* bufferDetect;
    char* bufferLandmark;
    long detectSize, landmarkSize;
    loadModel2Buffer(detectModelPath, bufferDetect, detectSize);
    loadModel2Buffer(landmarkModelPath, bufferLandmark, landmarkSize);

    char* model_suffix = "mnn";
    loadModelPoseDetect(bufferDetect, detectSize, model_suffix, false);
    loadModelPoseLandmark(bufferLandmark, landmarkSize, model_suffix, false);

    delete bufferDetect;
    delete bufferLandmark;

    Mat frame = imread("/Users/mzh/work/my_project/mediapipe_cmake/data/body_image/test.jpeg");

    PoseLandmarkResult* resut = new PoseLandmarkResult();
    runPoseLandmark((const char *)frame.data, frame.cols, frame.rows, frame.step, 0, -1, 1, resut);


    int w = frame.cols;
    int h = frame.rows;

    if (resut->poseNum == 1)
    {
        rectangle(frame, {resut->rect[0], resut->rect[1],resut->rect[2],resut->rect[3]}, Scalar(255, 0, 0), 2);

        for (int k = 0; k < 33; k++)
        {
            auto p = Point2f(resut->points[k * 2] * w, resut->points[k * 2+1] * h);
            circle(frame, p, 2, Scalar(255, 0, 255), 3);
        }
    }

    imshow("Pose Landmark by mpp_cmake", frame);
    waitKey(0);
    releasePoseLandmark();
}

int main()
{
//    test_image();
    // test_camera();
   test_camera_c_api();
//    test_img_c_api();
    return 0;

}