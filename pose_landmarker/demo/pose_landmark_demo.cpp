//
// Created by mzh on 2023/8/29.
//
#include "opencv2/opencv.hpp"
#include "pose_landmark.h"
#include "common.h"
#include "fstream"
#include "pose_landmark_c.h"

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
            circle(img, p, 2, Scalar(255, 0, 255), 3);
        }
    }
}

void test_image()
{
    string imgPath = "/Users/mzh/work/opencv_dev/mediapipe_cmake/data/body_image/test.jpeg";
    Mat img = imread(imgPath);

    string detectModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_landmarker/models/pose_landmark_full_sim.mnn";
    PoseLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::vector<BoxKp3> boxOut = {};
    landmarker.runImage(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string imgPath2 = "/Users/mzh/work/my_project/mediapipe_cmake/data/hand_image/palm_hands.jpeg";
    Mat img2 = imread(imgPath2);

    string detectModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.tflite";
//    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_landmarker/models/pose_landmark_full_sim.mnn";
    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_landmarker/models/pose_landmark_full.tflite";
    PoseLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::vector<BoxKp3> boxOut = {};
    VideoCapture cap("/Users/mzh/work/data/video/v0.mp4");
//    VideoCapture cap(0);
    TickMeter t;
    Mat frame;
    while (1)
    {
        t.reset();
        cap >> frame;

        t.start();
        landmarker.runVideo(frame, boxOut); // video mode, use hand tracking for speeding up.
//        landmarker.runImage(frame, boxOut); // image mode, will carry on the hand detector for each frame.
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
    string imgPath2 = "/Users/mzh/work/my_project/mediapipe_cmake/data/hand_image/palm_hands.jpeg";
    Mat img2 = imread(imgPath2);

    string detectModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_landmarker/models/pose_landmark_full_sim.mnn";

    initPoseLandmarker();

    char* bufferDetect;
    char* bufferLandmark;
    long detectSize, landmarkSize;
    loadModel2Buffer(detectModelPath, bufferDetect, detectSize);
    loadModel2Buffer(landmarkModelPath, bufferLandmark, landmarkSize);

    loadModelPoseDetect(bufferDetect, detectSize, false);
    loadModelPoseLandmark(bufferLandmark, landmarkSize, false);

    delete bufferDetect;
    delete bufferLandmark;

    VideoCapture cap("/Users/mzh/work/data/video/v0.mp4");
//    VideoCapture cap(0);
    TickMeter t;
    Mat frame;

    PoseLandmarkResult* resut = new PoseLandmarkResult();
    while (1)
    {
        t.reset();
        cap >> frame;

        t.start();

        runPoseLandmark((const char *)frame.data, frame.cols, frame.rows, frame.step, 0, -1, 1, resut);
//        landmarker.runVideo(frame, boxOut); // video mode, use hand tracking for speeding up.
//        landmarker.runImage(frame, boxOut); // image mode, will carry on the hand detector for each frame.
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
    string detectModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_detector/models/pose_detection.mnn";
    string landmarkModelPath = "/Users/mzh/work/my_project/mediapipe_cmake/pose_landmarker/models/pose_landmark_full_sim.mnn";

    initPoseLandmarker();

    char* bufferDetect;
    char* bufferLandmark;
    long detectSize, landmarkSize;
    loadModel2Buffer(detectModelPath, bufferDetect, detectSize);
    loadModel2Buffer(landmarkModelPath, bufferLandmark, landmarkSize);

    loadModelPoseDetect(bufferDetect, detectSize, false);
    loadModelPoseLandmark(bufferLandmark, landmarkSize, false);

    delete bufferDetect;
    delete bufferLandmark;

    Mat frame = imread("/Users/mzh/work/my_project/mediapipe_cmake/data/body_image/test.jpeg");

    PoseLandmarkResult* resut = new PoseLandmarkResult();
    runPoseLandmark((const char *)frame.data, frame.cols, frame.rows, frame.step, 0, -1, 1, resut);


    int w = frame.cols;
    int h = frame.rows;

//    std::string scoreText = "time = " + std::to_string(t.getTimeMilli());
    if (resut->poseNum == 1)
    {
        rectangle(frame, {resut->rect[0], resut->rect[1],resut->rect[2],resut->rect[3]}, Scalar(255, 0, 0), 2);

        for (int k = 0; k < 33; k++)
        {
//            std::string scoreText = std::to_string(k);
            auto p = Point2f(resut->points[k * 2] * w, resut->points[k * 2+1] * h);
            circle(frame, p, 2, Scalar(255, 0, 255), 3);
        }
    }

//    putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 0, 255));

    imshow("Pose Landmark by mpp_cmake", frame);
    waitKey(0);
    releasePoseLandmark();
}

void test_bin_2_yuv()
{
    String ROOT_PATH = "/Users/mzh/work/py_script/data/TestAndroidImg_YUV/";
    vector<string> file_paths = {"data0.bin", "data1.bin", "data2.bin"}; // 数据分别是Y,U,V

    // 由于图像是从Android传下来的数据，图片是旋转了90度，所以长比宽大
    // Image dimensions
    int height = 720; // Define the height of the image
    int width = 1280;  // Define the width of the image

    // 创建一个buffer
    vector<uint8_t > dataBuffer(height * width * 3, 0);

    int64_t index = 0;
    for (const auto& file_pathE : file_paths) {
        String file_path = ROOT_PATH + file_pathE;
        std::cout<<"load file "<< file_path<<std::endl;
        ifstream file(file_path, ios::binary);

        if (file.is_open()) {
            // Get the file size
            file.seekg(0, ios::end);
            streampos size = file.tellg();
            file.seekg(0, ios::beg);

            // Read the binary data into a buffer
            file.read(reinterpret_cast<char*>(dataBuffer.data() + index), size);
            index+=size;

            file.close();
        } else {
            cout << "Unable to open file: " << file_path << endl;
        }
    }

    Mat matYUV = Mat(height * 1.5, width, CV_8UC1, dataBuffer.data());

    Mat matRGB;
    cvtColor(matYUV, matRGB, COLOR_YUV2RGB_NV21);
    std::cout<<"W = "<<matRGB.cols<<", H = "<<matRGB.rows;
    rotate(matRGB, matRGB, 0); //

    imshow("img YUV ", matYUV);
    imshow("img BGR ", matRGB);
    waitKey(0);

}

int main()
{
//    test_bin_2_yuv();
//    test_image();
    test_camera();
//    test_camera_c_api();
//    test_img_c_api();
    return 0;

}