//
// Created for C API demo
//
#include "opencv2/opencv.hpp"
#include "hand_landmark_c_api.h"

using namespace cv;
using namespace std;

std::string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
std::string test_data_path = root_path + "data/";

void draw_hand_landmarks(Mat& img, const HandLandmarkResult* hands, int hand_count)
{
    int w = img.cols;
    int h = img.rows;

    // Hand connections (same as kHandConnections in hand_landmark.h)
    const vector<pair<int, int>> connections = {
        {0, 1}, {0, 5}, {9, 13}, {13, 17}, {5, 9}, {0, 17}, {1, 2},
        {2, 3}, {3, 4}, {5, 6}, {6, 7}, {7, 8}, {9, 10}, {10, 11},
        {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}
    };

    for (int i = 0; i < hand_count; i++)
    {
        const auto& hand = hands[i];
        
        // Draw bounding box
        rectangle(img, 
                  Rect(hand.rect[0], hand.rect[1], hand.rect[2], hand.rect[3]),
                  Scalar(255, 0, 0), 2);
        
        // Draw score
        string scoreText = "Score: " + to_string(hand.score);
        putText(img, scoreText, Point(hand.rect[0], hand.rect[1] - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        // Draw connections
        for (const auto& conn : connections)
        {
            int idx1 = conn.first;
            int idx2 = conn.second;
            if (idx1 < hand.landmark_count && idx2 < hand.landmark_count)
            {
                Point2f p1(hand.points[idx1 * 3] * w, hand.points[idx1 * 3 + 1] * h);
                Point2f p2(hand.points[idx2 * 3] * w, hand.points[idx2 * 3 + 1] * h);
                line(img, p1, p2, Scalar(0, 255, 255), 2);
            }
        }
        
        // Draw landmarks
        for (int k = 0; k < hand.landmark_count; k++)
        {
            Point2f p(hand.points[k * 3] * w, hand.points[k * 3 + 1] * h);
            circle(img, p, 3, Scalar(255, 0, 255), -1);
        }
    }
}

void test_c_api_image()
{
    cout << "=== Testing Hand Landmark C API - Image Mode ===" << endl;
    
    string imgPath = test_data_path + "hand_image/palm_hands.jpeg";
    Mat img = imread(imgPath);
    if (img.empty())
    {
        cout << "Failed to load image: " << imgPath << endl;
        return;
    }

    string detectModelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    string landmarkModelPath = root_path + "hand_landmarker/models/hand_landmark_full.mnn";

    // Initialize C API
    HandLandmarkResult hands[2];
    
    int ret = initHandLandmarker(2, 0); // max 2 hands, CPU
    if (ret != 0)
    {
        cout << "Failed to initialize hand landmarker: " << ret << endl;
        return;
    }
    
    ret = loadModelHandDetectFromFile(detectModelPath.c_str(), 0);
    if (ret != 0)
    {
        cout << "Failed to load detection model: " << ret << endl;
        releaseHandLandmarker();
        return;
    }
    
    ret = loadModelHandLandmarkFromFile(landmarkModelPath.c_str(), 0);
    if (ret != 0)
    {
        cout << "Failed to load landmark model: " << ret << endl;
        releaseHandLandmarker();
        return;
    }

    // Run detection
    int hand_count = runHandLandmarkImage(
        reinterpret_cast<const char*>(img.data),
        img.cols, img.rows, img.cols * 3,
        MPP_HAND_NO_FLIP, MPP_HAND_ROTATION_0,
        MPP_HAND_IMAGE_TYPE_BGR,
        hands, 2
    );
    
    cout << "Found " << hand_count << " hands" << endl;
    
    if (hand_count > 0)
    {
        for (int i = 0; i < hand_count; i++)
        {
            cout << "Hand " << i << ": " << hands[i].landmark_count << " landmarks, score=" 
                 << hands[i].score << endl;
        }
        
        // Draw and display
        draw_hand_landmarks(img, hands, hand_count);
        imshow("Hand Landmarks - C API", img);
        cout << "Press any key to continue..." << endl;
        waitKey(0);
    }
    
    releaseHandLandmarker();
}

void test_c_api_video()
{
    cout << "=== Testing Hand Landmark C API - Video Mode ===" << endl;
    
    string detectModelPath = root_path + "hand_detector/models/palm_detection_full.mnn";
    string landmarkModelPath = root_path + "hand_landmarker/models/hand_landmark_full.mnn";

    // Initialize C API
    int ret = initHandLandmarker(2, 0); // max 2 hands, CPU
    if (ret != 0)
    {
        cout << "Failed to initialize hand landmarker: " << ret << endl;
        return;
    }
    
    ret = loadModelHandDetectFromFile(detectModelPath.c_str(), 0);
    if (ret != 0)
    {
        cout << "Failed to load detection model: " << ret << endl;
        releaseHandLandmarker();
        return;
    }
    
    ret = loadModelHandLandmarkFromFile(landmarkModelPath.c_str(), 0);
    if (ret != 0)
    {
        cout << "Failed to load landmark model: " << ret << endl;
        releaseHandLandmarker();
        return;
    }

    // Open video
    string video_path = test_data_path + "video/hands03.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout << "Failed to open video: " << video_path << endl;
        releaseHandLandmarker();
        return;
    }
    
    cout << "Processing video. Press 'q' or ESC to quit." << endl;
    
    HandLandmarkResult hands[2];
    TickMeter timer;
    Mat frame, display_frame;
    
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        
        // Resize for consistent processing
        resize(frame, display_frame, Size(640, 480));
        
        timer.reset();
        timer.start();
        
        // Run detection with C API (video mode for tracking)
        int hand_count = runHandLandmarkVideo(
            reinterpret_cast<const char*>(display_frame.data),
            display_frame.cols, display_frame.rows, display_frame.cols * 3,
            MPP_HAND_NO_FLIP, MPP_HAND_ROTATION_0,
            MPP_HAND_IMAGE_TYPE_BGR,
            hands, 2
        );
        
        timer.stop();
        
        // Draw results
        if (hand_count > 0)
        {
            draw_hand_landmarks(display_frame, hands, hand_count);
        }
        
        // Display FPS
        string fps_text = "Time: " + to_string(timer.getTimeMilli()) + " ms | Hands: " + to_string(hand_count);
        putText(display_frame, fps_text, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        imshow("Hand Landmarks - C API Video", display_frame);
        
        int key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) // 'q' or ESC
            break;
    }
    
    cap.release();
    destroyAllWindows();
    releaseHandLandmarker();
    
    cout << "Video processing complete." << endl;
}

int main()
{
    // Test image mode
    test_c_api_image();
    
    // Test video mode
    test_c_api_video();
    
    return 0;
}
