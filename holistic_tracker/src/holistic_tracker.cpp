// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt

// The following is original lincense:
// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holistic_tracker.h"
#include "face_processing.h"
#include "hand_processing.h"
#include "pose_landmark.h"

using namespace cv;

namespace mpp
{

void draw_debug(const Mat& img, const PointList3f& points)
{
    int w = img.cols;
    int h = img.rows;

    for (int k = 0; k < points.size(); k++)
    {
        std::string scoreText = std::to_string(k);
        auto p = Point2f(points[k].x * w, points[k].y * h);
//            putText(img, scoreText, p, 2, 1, cv::Scalar(0, 255, 0));
        circle(img, p, 0.5, Scalar(255, 0, 255), 2);
    }
}

// TODO finish the constructor
HolisticTracker::HolisticTracker(std::string pose_detector, std::string pose_landmarker, std::string face_detector, std::string face_landmarker,
                         std::string hand_recrop, std::string hand_landmark, int _device)
: device(_device)
{
    const int maxHunameNum = 1;
    poseLandmarker = makePtr<PoseLandmarker>(pose_detector, pose_landmarker, maxHunameNum, _device);
    faceProcessor = makePtr<FaceProcessor>(face_detector, face_landmarker);
    handProcessor = makePtr<HandProcessor>(hand_recrop, hand_landmark, _device);
}

HolisticTracker::~HolisticTracker()
{
}
//
//#define INTERNAL_IRIS_MIN(x, y) x > y ? y : x
//#define INTERNAL_IRIS_MAX(x, y) x > y ? x : y
//
//inline void projectBackResizeUnscale(const ImgScaleParams& params, float imgW, float imgH, const Rect2f& roi,
//                                     PointList3f& landmarkIris, PointList3f& landmarkEye)
//{
//    float div_ratio = 1.0/params.ratio;
//    for(int j=0; j<landmarkIris.size(); j++)
//    {
//        landmarkIris[j].x = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkIris[j].x - params.dw) * div_ratio + roi.x, imgW), 0.0f);
//        landmarkIris[j].y = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkIris[j].y - params.dh) * div_ratio + roi.y, imgH), 0.0f);
//    }
//
//    for(int j=0; j<landmarkEye.size(); j++)
//    {
//        landmarkEye[j].x = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkEye[j].x - params.dw) * div_ratio + roi.x, imgW), 0.0f);
//        landmarkEye[j].y = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkEye[j].y - params.dh) * div_ratio + roi.y, imgH), 0.0f);
//    }
//}
//
//inline void flipLandmark(float imgW, PointList3f& landmarkIris, PointList3f& landmarkEye)
//{
//    for(int j=0; j<landmarkIris.size(); j++)
//    {
//        landmarkIris[j].x = imgW - landmarkIris[j].x;
//    }
//
//    for(int j=0; j<landmarkEye.size(); j++)
//    {
//        landmarkEye[j].x = imgW - landmarkEye[j].x;
//    }
//}

void HolisticTracker::runImage(const cv::Mat &input, HolisticOutput &output)
{
    CV_Assert(!input.empty());

    const int imgH = input.rows;
    const int imgW = input.cols;
    std::vector<BoxKp3> boxLandmark;

    // step1: get pose landmark
    poseLandmarker->runImage(input, output.poseLandmark);

#ifdef HOLISTIC_DEBUG
    if (output.poseLandmark.size() == 1)
        draw_debug(input, output.poseLandmark[0].points);
    imshow("reCrop imgCrop", input);
    waitKey(1);
#endif

    // step2: get face landmark
    faceProcessor->run(input, output);
#ifdef HOLISTIC_DEBUG
    if (output.faceLandmark.size() == 1)
        draw_debug(input, output.faceLandmark[0].points);
    imshow("reCrop imgCrop", input);
    waitKey(1);
#endif

    // step3: get hand landmark
    handProcessor->run(input, output);

#ifdef HOLISTIC_DEBUG
    if (output.leftHandLandmark.size() == 1)
        draw_debug(input, output.leftHandLandmark[0].points);
    if (output.rightHandLandmark.size() == 1)
        draw_debug(input, output.rightHandLandmark[0].points);
    imshow("reCrop imgCrop", input);
    waitKey(1);
#endif
}

void HolisticTracker::runVideo(const cv::Mat &input, HolisticOutput &output)
{
    CV_Assert(!input.empty());

    const int imgH = input.rows;
    const int imgW = input.cols;
    std::vector<BoxKp3> boxLandmark;

    // step1: get pose landmark
    poseLandmarker->runVideo(input, output.poseLandmark);

    // step2: get face landmark
    faceProcessor->run(input, output);

    // step3: get hand landmark
    handProcessor->run(input, output);
}

}
