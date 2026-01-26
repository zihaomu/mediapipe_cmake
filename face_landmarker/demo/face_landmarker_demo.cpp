//
// Created by mzh on 2023/8/29.
//
#include "opencv2/opencv.hpp"
#include "face_landmark.h"

using namespace cv;
using namespace mpp;
using namespace std;

// ROOT PATH and test data

std::string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string test_data_path = root_path + "data/";

void draw(const Mat& img, const std::vector<BoxKp3>& box)
{
    int w = img.cols;
    int h = img.rows;

    for (int i = 0; i < box.size(); i++)
    {
        rectangle(img, box[i].rect, Scalar(255, 0, 0), 2);

        std::cout<<"face box[i].points.size()" << box[i].points.size() << std::endl;
        for (int k = 0; k < box[i].points.size(); k++)
        {
            std::string scoreText = std::to_string(k);
            auto p = Point2f(box[i].points[k].x * w, box[i].points[k].y * h);
            circle(img, p, 0.5, Scalar(255, 0, 255), 2);
        }
    }
}

void test_image()
{
    string imgPath = test_data_path + "face_image/face0.jpg";
    string output_img_path = test_data_path + "face_image/face0_landmark.jpg";
    Mat img = imread(imgPath);

    string detectModelPath = root_path + "face_detector/models/face_detection_full_range.mnn";
    string landmarkModelPath = root_path + "face_landmarker/models/face_landmark478.mnn";
    FaceLandmarker landmarker(detectModelPath, landmarkModelPath, 2);
    std::vector<BoxKp3> boxOut = {};
    landmarker.runImage(img, boxOut);

    draw(img, boxOut);
    imshow("img", img);
    imwrite(output_img_path, img);  
    waitKey(0);
}

void test_camera()
{
    string detectModelPath = root_path + "face_detector/models/face_detection_short_range.mnn";
    string landmarkModelPath = root_path + "face_landmarker/models/face_landmark478.mnn";
    FaceLandmarker landmarker(detectModelPath, landmarkModelPath, 1);

    std::string video_path = test_data_path + "video/face_smile.mp4";
    std::vector<BoxKp3> boxOut = {};
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open video!" << std::endl;
        return;
    }
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cout << "Failed to open video!" << std::endl;
        return;
    }

    TickMeter t;

    // Now start the loop
    while (true)
    {
        t.reset();
        
        // Only read next frame inside loop
        cap >> frame;
        if (frame.empty())
            break;

        t.start();
        landmarker.runVideo(frame, boxOut);
        t.stop();

        std::string scoreText = "time = " + std::to_string(t.getTimeMilli()) + " ms";
        putText(frame, scoreText, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

        draw(frame, boxOut);
        imshow("img", frame);
        if (waitKey(1) == 27)
            break;
    }
}

#include "face_landmark_c_api.h"
void test_c_api()
{
    string imgPath = test_data_path + "face_image/face0.jpg";
    string output_img_path = test_data_path + "face_image/face0_landmark.jpg";
    Mat img = imread(imgPath);

    string detectModelPath = root_path + "face_detector/models/face_detection_full_range.mnn";
    string landmarkModelPath = root_path + "face_landmarker/models/face_landmark478.mnn";
    std::vector<BoxKp3> boxOut = {};

    FaceLandmarkResult faces[2];

    initFaceLandmarker(2, 0); // max 2 faces, CPU
    loadModelFaceDetectFromFile(detectModelPath.c_str(), 0);
    loadModelFaceLandmarkFromFile(landmarkModelPath.c_str(), 0);

    // BGR input buffer with row stride width*3, no flip, no rotation
    int found = runFaceLandmarkImage(reinterpret_cast<const char*>(img.data), img.cols, img.rows, img.cols * 3,
                                     MPP_FACE_NO_FLIP, MPP_FACE_ROTATION_0,
                                     MPP_FACE_IMAGE_TYPE_BGR, faces, 2);
    
    std::cout << "Found " << found << " faces via C API" << std::endl;
    
    // Convert C API results to BoxKp3 for visualization
    for (int i = 0; i < found; i++)
    {
        BoxKp3 box;
        box.rect = Rect2f(faces[i].rect[0], faces[i].rect[1], faces[i].rect[2], faces[i].rect[3]);
        box.score = faces[i].score;
        box.radians = faces[i].radians;
        for (int k = 0; k < faces[i].landmark_count; k++)
        {
            box.points.push_back(Point3f(faces[i].points[k * 3], faces[i].points[k * 3 + 1], faces[i].points[k * 3 + 2]));
        }
        boxOut.push_back(box);
    }
    
    releaseFaceLandmarker();
    draw(img, boxOut);
    imshow("img", img);
    imwrite(output_img_path, img);  
    waitKey(0);
}

int main()
{
   test_image();
//    test_camera();
    // test_c_api();
    return 0;
}