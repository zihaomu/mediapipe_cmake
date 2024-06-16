// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.cc

// The following is original lincense:
/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pose_landmark.h"
#include "pose_landmark_impl.h"
#include "opencv2/dnn.hpp"

using namespace cv;
using namespace dnn;
namespace mpp
{

PoseLandmarker::PoseLandmarker(std::string detectorPath, std::string landmarkPath, int _maxHumanNum, int _device)
: maxHumanNum(_maxHumanNum), device(_device)
{
    poseDetector = makePtr<PoseDetector>(detectorPath, maxHumanNum, device);
    poseLanmark_impl = makePtr<PoseLandmarker_Impl>(landmarkPath, device);

    smoother = makePtr<OneEuroSmoother>(0.01f, 80.f);
}

PoseLandmarker::PoseLandmarker(int _maxHumanNum, int _device)
        : maxHumanNum(_maxHumanNum), device(_device)
{
    smoother = makePtr<OneEuroSmoother>(0.01f, 80.f);
}

void PoseLandmarker::loadDetectModel(std::string detector_path, int _device)
{
    device = _device;
    poseDetector = makePtr<PoseDetector>(detector_path, maxHumanNum, device);
}

void PoseLandmarker::loadLandmarkModel(std::string landmark_path, int _device)
{
    device = _device;
    poseLanmark_impl = makePtr<PoseLandmarker_Impl>(landmark_path, device);
}

void PoseLandmarker::loadDetectModel(const char *buffer, long buffer_size, bool _isTFlite, int _device)
{
    device = _device;
    isTFlite = _isTFlite;
    poseDetector = makePtr<PoseDetector>(buffer, buffer_size, isTFlite,  maxHumanNum, device);
}

void PoseLandmarker::loadLandmarkModel(const char *buffer, long buffer_size, bool _isTFlite, int _device)
{
    device = _device;
    isTFlite = _isTFlite;
    poseLanmark_impl = makePtr<PoseLandmarker_Impl>(buffer, buffer_size, isTFlite, device);
}

PoseLandmarker::~PoseLandmarker()
{
}

Point2f computeCenterFromLandmark(const BoxKp3& box, int imgW, int imgH)
{
    constexpr int leftHip = 23;
    constexpr int rightHip = 24;

    return Point2f((box.points[leftHip].x + box.points[rightHip].x) * 0.5f * imgW,
                   (box.points[leftHip].y + box.points[rightHip].y) * 0.5f * imgH);
}

void PoseLandmarker::runImage(const cv::Mat &img, std::vector<BoxKp3> &boxLandmark)
{
    CV_Assert(poseDetector && poseLanmark_impl && "The poseDetector or poseLanmark is null!");
    boxLandmark.clear();

    // step1: pose detect
    std::vector<BoxKp2> boxes;
    std::vector<float> angles;
    std::vector<Point2f> bodyCenters;
    runDetect(img, boxes, angles, bodyCenters);

    CV_Assert(boxes.size() == angles.size());

    int handNum = boxes.size();
    for (int i = 0; i < handNum; i++)
    {
        Mat imgCrop;
        Mat tranMatInv;

        bodyAlign(img, boxes[i], bodyCenters[i], angles[i], tranMatInv, imgCrop);

        PointList3f landmark_pixel, landmark_world;
        PointList2f visPre;
        float score_pixel;
        int H, W;
        poseLanmark_impl->getInputWH(W, H);
        poseLanmark_impl->run(imgCrop, landmark_pixel, landmark_world, visPre, score_pixel);

        if (score_pixel < threshold)
            continue;

        PointList3f landmark_pixel_out;
        float ratio_z = boxes[i].rect.width / (float) W/ float(img.cols);
        projectLandmarkBack(landmark_pixel, img.cols, img.rows, tranMatInv, landmark_pixel_out, ratio_z);

        BoxKp3 box = {};
        box.rect = boxes[i].rect;
        box.points = landmark_pixel_out;
        box.score = score_pixel;
        box.vis_pre = visPre;
        boxLandmark.push_back(box);
    }
}

void PoseLandmarker::runVideo(const cv::Mat &img, std::vector<BoxKp3> &boxLandmark)
{
    CV_Assert(poseDetector && poseLanmark_impl && "The poseDetector or poseLanmark is null!");
    boxLandmark.clear();

    // step1: pose detect
    std::vector<BoxKp2> boxes;
    std::vector<float> angles;
    std::vector<Point2f> bodyCenters;

    TickMeter m;
    m.reset();

    m.start();
    runTrack(img, boxes, angles, bodyCenters);
    m.stop();

    CV_Assert(boxes.size() == angles.size());

    m.reset();
    m.start();

    int bodyNum = boxes.size();
    for (int i = 0; i < bodyNum; i++)
    {
        Mat imgCrop;
        Mat tranMatInv;

        int H, W;
        poseLanmark_impl->getInputWH(W, H);
        bodyAlign(img, boxes[i], bodyCenters[i], angles[i], tranMatInv, imgCrop);

        PointList3f landmark_pixel, landmark_world;
        PointList2f visPre;
        float score_pixel;

        poseLanmark_impl->run(imgCrop, landmark_pixel, landmark_world, visPre, score_pixel);

        if (score_pixel < threshold)
        {
            continue;
        }

        float ratio_z = boxes[i].rect.width / (float)W / float(img.cols);
        PointList3f landmark_pixel_out;
        projectLandmarkBack(landmark_pixel, img.cols, img.rows, tranMatInv, landmark_pixel_out, ratio_z);

        BoxKp3 box = {};
        box.rect = boxes[i].rect;
        box.points = landmark_pixel_out;
        box.score = score_pixel;
        box.vis_pre = visPre;

        boxLandmark.push_back(box);
    }

    if (!boxLandmark.empty())
    {
        if (boxLandmark.size() == 1)
        {
            PointList3f landmarkSmoothed;
            int64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                    .count();

            smoother->apply(boxLandmark[0].points, timestamp, landmarkSmoothed);
            boxLandmark[0].points = landmarkSmoothed;
        }

        preBoxPoints = boxLandmark;
    }
    else
    {
        smoother->reset();
        preBoxPoints.clear();
    }

    m.stop();
}

// TODO fix the following function
float PoseLandmarker::computeRotateFromLandmark(const mpp::PointList3f &landmarks)
{
    constexpr int leftHip = 23;
    constexpr int rightHip = 24;
    constexpr int leftShoulder = 11;
    constexpr int rightShoulder = 12;

    CV_Assert(!landmarks.empty() && "The input points list is empty!");
    float x0 = (landmarks[leftHip].x + landmarks[rightHip].x)/2.0f;
    float y0 = (landmarks[leftHip].y + landmarks[rightHip].y)/2.0f;

    float x1 = (landmarks[leftShoulder].x + landmarks[rightShoulder].x)/2.0f;
    float y1 = (landmarks[leftShoulder].y + landmarks[rightShoulder].y)/2.0f;

    double r = M_PI * 0.5 - std::atan2(-(y1 - y0), x1 - x0);   // compute the radians
    return normalizeRadius(r);
}

float PoseLandmarker::computeRotateFromDetect(const BoxKp2& box)
{
    CV_Assert(box.points.size() == 4 && "The key points from hand detector miss match the shape.");
    float x0 = box.points[0].x;
    float y0 = box.points[0].y;
    float x1 = box.points[1].x;
    float y1 = box.points[1].y;

    double r = M_PI * 0.5 - std::atan2(-(y1 - y0), x1 - x0);   // compute the radians
    return normalizeRadius(r);
}

void PoseLandmarker::bodyAlign(const cv::Mat &src, BoxKp2 &box, cv::Point2f &center, float angle,
                               cv::Mat &transMatInv, cv::Mat &dst)
{
    float cos_r = std::cos(angle);
    float sin_r = std::sin(angle);

    std::vector<cv::Point2f> dstPts(4);
    std::vector<cv::Point2f> srcPts(4);

    int H, W;
    poseLanmark_impl->getInputWH(W, H);
    dstPts[0] = cv::Point2f(0, 0);
    dstPts[1] = cv::Point2f(W, 0);
    dstPts[2] = cv::Point2f(W, H);
    dstPts[3] = cv::Point2f(0, H);

    auto& rect = box.rect;

    // NOTE: Instead of using the center of the hand detection box, use the center of the palm part.
    // This is done because the hand and finger rotates around the palm of the hand.
    float center_x = center.x;
    float center_y = center.y;

    srcPts[0] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[1] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[2] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r +
            center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r +
            center_y
    );

    srcPts[3] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r + center_y
    );

    cv::Mat transMat = cv::getAffineTransform(srcPts.data(), dstPts.data());

    cv::warpAffine(src, dst, transMat, cv::Size(W, H), 1, 0);
    cv::invertAffineTransform(transMat, transMatInv);
}

void PoseLandmarker::runDetect(const cv::Mat &img, std::vector<BoxKp2> &boxes, std::vector<float> &angles,
                               std::vector<cv::Point2f> &bodyCenters)
{
    CV_Assert(poseDetector);
    poseDetector->run(img, boxes);

    // compute rotation angle and palm center. NOTE: the palm center will be used at rotated center.
    angles.resize(boxes.size());
    bodyCenters.resize(boxes.size());
    for (int i = 0; i < boxes.size(); i++)
    {
        angles[i] = computeRotateFromDetect(boxes[i]);
        bodyCenters[i] = boxes[i].points[0]; // the rotated center is the mid-hip point.
    }
}

void getRectFromBodyLandmark(const BoxKp3& points, int imgW, int imgH, Point2f& center, Rect2f& rect)
{
    // Find boundaries of landmarks.
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    auto& pointList = points.points;
    int pointsSize = pointList.size();

    for (int i = 0; i < pointsSize; ++i)
    {
        max_x = std::max(max_x, pointList[i].x);
        max_y = std::max(max_y, pointList[i].y);
        min_x = std::min(min_x, pointList[i].x);
        min_y = std::min(min_y, pointList[i].y);
    }

    // bounding box
    float width = (max_x - min_x) * imgW;
    float height = (max_y - min_y) * imgH;

    float expandRatio = 1.35f;
    float longEdge = std::max(width, height) * expandRatio;

    rect.width = longEdge;
    rect.height = longEdge;

    const float center_x = center.x; //(max_x + min_x) * imgW / 2.f;
    const float center_y = center.y; //(max_y + min_y) * imgH / 2.f;

    rect.x = center_x - 0.5f * longEdge;//std::max(0.f, center_x - 0.5f * longEdge);
    rect.y = center_y - 0.5f * longEdge; //std::max(0.f, center_y - 0.5f * longEdge - 0.05f * longEdge);
}

void PoseLandmarker::runTrack(const cv::Mat &img, std::vector<BoxKp2> &boxes, std::vector<float> &angles,
                              std::vector<cv::Point2f> &bodyCenters)
{
    LOGD("LOG of C++, runTrack preBox = %d, maxHuman = %d!", preBoxPoints.size(), maxHumanNum);
    if (preBoxPoints.size() >= maxHumanNum) // get bounding box from pre landmark.
    {
        int preBodyNum = preBoxPoints.size();
        boxes.resize(preBodyNum);
        angles.resize(preBodyNum);
        bodyCenters.resize(preBodyNum);

        for (int i = 0; i < preBodyNum; i++)
        {
            BoxKp2 box = {};
            Rect2f rect = {};

            float angle = computeRotateFromLandmark(preBoxPoints[i].points);
            bodyCenters[i] = computeCenterFromLandmark(preBoxPoints[i], img.cols, img.rows);
            getRectFromBodyLandmark(preBoxPoints[i], img.cols, img.rows, bodyCenters[i], rect);
            boxes[i].rect = rect;
            angles[i] = angle;
        }
    }
    else
    {
        runDetect(img, boxes, angles, bodyCenters);
    }
}

} // namespace mpp