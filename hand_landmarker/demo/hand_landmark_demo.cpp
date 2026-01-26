//
// Created by mzh on 2023/8/29.
//
#include "opencv2/opencv.hpp"
#include "hand_landmark.h"

using namespace cv;
using namespace mpp;
using namespace std;

void draw(const Mat& img, const std::vector<BoxKp3>& box)
{
    int w = img.cols;
    int h = img.rows;

    for (int i = 0; i < box.size(); i++)
    {
        cv::Point2f center = Point2f(box[i].points[0].x * w, box[i].points[0].y * h);
        drawRotateRect(img, box[i].rect, box[i].radians, center);

        for (int k = 0; k < box[i].points.size(); k++)
        {
            std::string scoreText = std::to_string(k);
            auto p = Point2f(box[i].points[k].x * w, box[i].points[k].y * h);
            circle(img, p, 2, Scalar(255, 0, 255), 3);
        }
    }
}

std::string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string test_data_path = root_path + "data/";

void test_image()
{
    string imgPath = test_data_path + "hand_image/palm_hands.jpeg";
    Mat img = imread(imgPath);

    string detectModelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    string landmarkModelPath = root_path + "hand_landmarker/models/hand_landmark_full.mnn";
    HandLandmarker landmarker(detectModelPath, landmarkModelPath, 2);
    std::vector<BoxKp3> boxOut = {};
    landmarker.runImage(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    waitKey(0);
}

void test_camera()
{
    string detectModelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    string landmarkModelPath = root_path + "hand_landmarker/models/hand_landmark_full.mnn";
    HandLandmarker landmarker(detectModelPath, landmarkModelPath, 2);

    std::vector<BoxKp3> boxOut = {};
    // VideoCapture cap(0);
    
    std::string video_path = test_data_path + "video/hands03.mp4";
    VideoCapture cap(video_path);
    
    if (!cap.isOpened()) {
        std::cout << "Failed to open video!" << std::endl;
        return;
    }
    Mat iframe, frame;
    cap >> iframe;
    if (iframe.empty()) {
        std::cout << "Failed to open video!" << std::endl;
        return;
    }
    resize(iframe, frame, Size(1920, 1080));

    TickMeter t;
    // Mat frame_orig, frame;
    while (1)
    {
        t.reset();
        cap >> iframe;
        if (iframe.empty())
            break;
        // Keep aspect ratio and fit into 1920x1080
        float scale = std::min(1920.0f / iframe.cols, 1080.0f / iframe.rows);
        int new_width = static_cast<int>(iframe.cols * scale);
        int new_height = static_cast<int>(iframe.rows * scale);
        Mat temp(1080, 1920, iframe.type(), Scalar(0, 0, 0));
        resize(iframe, frame, Size(new_width, new_height));
        frame.copyTo(temp(Rect((1920 - new_width) / 2, (1080 - new_height) / 2, new_width, new_height)));
        frame = temp;

        t.start();
        landmarker.runVideo(frame, boxOut); // video mode, use hand tracking for speeding up.
        t.stop();

        std::string scoreText = "time = " + std::to_string(t.getTimeMilli()) + " ms";
        putText(frame, scoreText, cv::Point(10, 100), 2, 1, cv::Scalar(0, 255, 0));

        for (int i = 0; i < boxOut.size(); i++)
        {
            drawConnection(frame, boxOut[i].points, kHandConnections, frame.cols, frame.rows);
        }

        draw(frame, boxOut);
        imshow("img", frame);
        if (waitKey(1) == 27)
            break;
    }
}


int main()
{
//    test_image(); // Image mode
    test_camera();  // Video mode
    return 0;
}