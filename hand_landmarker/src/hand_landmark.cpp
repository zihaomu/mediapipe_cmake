// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.cc and
// hand_landmarker_graph.cc

// The following is original lincense:
/* Copyright 2022 The MediaPipe Authors.

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


#include "hand_landmark.h"
#include "hand_landmark_impl.h"

using namespace cv;
namespace mpp
{
HandLandmarker::HandLandmarker(std::string detector_path, std::string landmark_path, int _maxHandNum, int _device)
: maxHandNum(_maxHandNum), device(_device)
{
    // create handLandmarker
    handLanmark_impl = makePtr<HandLandmarker_Impl>(landmark_path);

    // create hand detector
    handDetector = makePtr<HandDetector>(detector_path, maxHandNum, device);
    smoother = makePtr<OneEuroSmoother>(0.05f, 50.f);
}

HandLandmarker::HandLandmarker(int _maxHandNum, int _device)
: maxHandNum(_maxHandNum), device(_device)
{
    smoother = makePtr<OneEuroSmoother>(0.05f, 50.f);
}

void HandLandmarker::loadDetectModel(std::string detector_path)
{
    handDetector = makePtr<HandDetector>(detector_path, maxHandNum, device);
}

void HandLandmarker::loadLandmarkModel(std::string landmark_path)
{
    handLanmark_impl = makePtr<HandLandmarker_Impl>(landmark_path, device);
}

void HandLandmarker::loadDetectModel(const char *buffer, long buffer_size, std::string model_suffix)
{
    CV_Assert(model_suffix == "mnn"); // 目前仅支持MNN
    handDetector = makePtr<HandDetector>(buffer, buffer_size, model_suffix, maxHandNum, device);
}

void HandLandmarker::loadLandmarkModel(const char *buffer, long buffer_size, std::string model_suffix)
{
    CV_Assert(model_suffix == "mnn"); // 目前仅支持MNN
    handLanmark_impl = makePtr<HandLandmarker_Impl>(buffer, buffer_size, model_suffix, device);
}

HandLandmarker::~HandLandmarker()
{
}

Point2f computeCenterFromDetect(const BoxKp2& box)
{
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    auto& pointList = box.points;
    int pointsSize = pointList.size();

    for (int i = 0; i < pointsSize; ++i)
    {
        max_x = std::max(max_x, pointList[i].x);
        max_y = std::max(max_y, pointList[i].y);
        min_x = std::min(min_x, pointList[i].x);
        min_y = std::min(min_y, pointList[i].y);
    }
    return Point2f((max_x + min_x) * 0.5f, (max_y + min_y) * 0.5f);
}

Point2f computeCenterFromLandmark(const BoxKp3& box, int imgW, int imgH)
{
    static const std::vector<int> palmIndex = {0, 1, 2, 5, 9, 13, 17};
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    auto& pointList = box.points;
    for (int i = 0; i < palmIndex.size(); ++i)
    {
        max_x = std::max(max_x, pointList[palmIndex[i]].x);
        max_y = std::max(max_y, pointList[palmIndex[i]].y);
        min_x = std::min(min_x, pointList[palmIndex[i]].x);
        min_y = std::min(min_y, pointList[palmIndex[i]].y);
    }
    return Point2f((max_x + min_x) * 0.5f * imgW, (max_y + min_y) * 0.5f * imgH);
}

void HandLandmarker::runDetect(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles,
                               std::vector<Point2f>& palmCenters)
{
    CV_Assert(handDetector);
    handDetector->run(img, boxes);

    // compute rotation angle and palm center. NOTE: the palm center will be used at rotated center.
    angles.resize(boxes.size());
    palmCenters.resize(boxes.size());
    for (int i = 0; i < boxes.size(); i++)
    {
        angles[i] = computeRotateFromDetect(boxes[i]);
        palmCenters[i] = computeCenterFromDetect(boxes[i]);
    }
}

void getRectFromHandLandmark(const BoxKp3& points, int imgW, int imgH, Point2f& center, Rect2f& rect)
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

    const float center_x = center.x; //(max_x + min_x) * imgW / 2.f;
    const float center_y = center.y; //(max_y + min_y) * imgH / 2.f;

    // bounding box
    float width = (max_x - min_x) * imgW;
    float height = (max_y - min_y) * imgH;

    // TODO fine-tuning the following params. This param is sensitive to the final hand landmark result.
    float expandRatio = 2.f;
    float shift_y = -0.25f;
    float longEdge = std::max(width, height) * expandRatio;

    rect.width = longEdge;
    rect.height = longEdge;

    rect.x = center_x - 0.5f * longEdge;
    rect.y = center_y - 0.5f * longEdge + shift_y * longEdge;
}

// This function will contain the strategy to reason the next frame hand bounding box by current frame hand landmark.
// When the detected hand number equal the maxHandNum, we would like use the above strategy instead of calling the
// hand detector for each frame.
void HandLandmarker::runTrack(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles, std::vector<cv::Point2f>& palmCenters)
{
    // TODO: try to optimized the hand tracking strategy. Right now the effect is not good enough.
    if (preBoxPoints.size() >= maxHandNum) // get bounding box from pre landmark.
    {
        int preHandNum = preBoxPoints.size();
        boxes.resize(preHandNum);
        angles.resize(preHandNum);
        palmCenters.resize(preHandNum);

        for (int i = 0; i < preHandNum; i++)
        {
            BoxKp2 box = {};
            Rect2f rect = {};

            float angle = computeRotateFromLandmark(preBoxPoints[i].points);
            palmCenters[i] = computeCenterFromLandmark(preBoxPoints[i], img.cols, img.rows);
            getRectFromHandLandmark(preBoxPoints[i], img.cols, img.rows, palmCenters[i], rect);
            boxes[i].rect = rect;
            angles[i] = angle;
        }

#if 0
        // only for box debug
        std::vector<BoxKp2> boxes_d;
        std::vector<float> angles_d;
        std::vector<cv::Point2f> palmCenters_d;
        runDetect(img, boxes_d, angles_d, palmCenters_d);

        // draw tracking and detect for a same img so that to refine the tracking box to feet the size of detection box.

        // draw tracking
        for (int i = 0; i < preHandNum; i++)
        {
            rectangle(img, boxes[i].rect, Scalar(255, 0, 0), 2);
        }

        // draw detection
        for (int i = 0; i < boxes_d.size(); i++)
        {
            rectangle(img, boxes_d[i].rect, Scalar(0, 255, 0), 2);
        }
#endif
    }
    else
    {
        runDetect(img, boxes, angles, palmCenters);
    }
}

float HandLandmarker::computeRotateFromDetect(const BoxKp2& box)
{
    CV_Assert(box.points.size() == 7 && "The key points from hand detector miss match the shape.");
    float x0 = box.points[0].x;
    float y0 = box.points[0].y;
    float x1 = box.points[1].x;
    float y1 = box.points[1].y;

    double r = M_PI * 0.5 - std::atan2(-(y1 - y0), x1 - x0);   // compute the radians
    return normalizeRadius(r);
}

float HandLandmarker::computeRotateFromLandmark(const PointList3f& landmarks)
{
    constexpr int kWristJoint = 0;
    constexpr int kMiddleFingerPIPJoint = 6;
    constexpr int kIndexFingerPIPJoint = 4;
    constexpr int kRingFingerPIPJoint = 8;
    constexpr int kNumLandmarks = 21;

    CV_Assert(!landmarks.empty() && "The input points list is empty!");
    float x0 = landmarks[kWristJoint].x;
    float y0 = landmarks[kWristJoint].y;

    float x1 = (landmarks[kIndexFingerPIPJoint].x + landmarks[kRingFingerPIPJoint].x)/2.0f;
    float y1 = (landmarks[kIndexFingerPIPJoint].y + landmarks[kRingFingerPIPJoint].y)/2.0f;

    x1 = (x1 + landmarks[kMiddleFingerPIPJoint].x) / 2.0f;
    y1 = (y1 + landmarks[kMiddleFingerPIPJoint].y) / 2.0f;

    // The finger direction is required to be horizontal.
    const float r = M_PI * 0.5 - std::atan2(-(y1 - y0), x1 - x0);
    return normalizeRadius(r);
}

void HandLandmarker::runImage(const cv::Mat &img, std::vector<BoxKp3> &boxLandmark)
{
    CV_Assert(handDetector && handLanmark_impl && "The handDetector or handLanmark is null!");
    boxLandmark.clear();

    // step1: hand detect
    std::vector<BoxKp2> boxes;
    std::vector<float> angles;
    std::vector<Point2f> palmCenters;
    runDetect(img, boxes, angles, palmCenters);

    CV_Assert(boxes.size() == angles.size());

    int handNum = boxes.size();
    for (int i = 0; i < handNum; i++)
    {
        Mat imgCrop;
        Mat tranMatInv;

        int H, W;
        handLanmark_impl->getInputWH(W, H);
        imageAlignment(img, W, H, boxes[i], angles[i], tranMatInv, imgCrop, palmCenters[i]);

        PointList3f landmark_pixel, landmark_world;
        float score_pixel, score_world;

        handLanmark_impl->run(imgCrop, landmark_pixel, score_pixel, landmark_world, score_world);

        if (score_pixel < threshold)
            continue;

        PointList3f landmark_pixel_out;
        float ratio_z = boxes[i].rect.width / (float)W / float(img.cols);
        projectLandmarkBack(landmark_pixel, img.cols, img.rows, tranMatInv, landmark_pixel_out, ratio_z);

        BoxKp3 box = {};
        box.rect = boxes[i].rect;
        box.radians = angles[i];
        box.points = landmark_pixel_out;
        box.score = score_pixel;

        boxLandmark.push_back(box);
    }
}

void HandLandmarker::runVideo(const cv::Mat &img, std::vector<BoxKp3> &boxLandmark)
{
    CV_Assert(handDetector && handLanmark_impl && "The handDetector or handLanmark is null!");
    boxLandmark.clear();

    // step1: hand tracking, get the hand bounding box and rotated angle.
    std::vector<BoxKp2> boxes;
    std::vector<float> angles;
    std::vector<Point2f> palmCenters;
    runTrack(img, boxes, angles, palmCenters);

    CV_Assert(boxes.size() == angles.size());

    int handNum = boxes.size();
    for (int i = 0; i < handNum; i++)
    {
        Mat imgCrop;
        Mat tranMatInv;
        int H, W;
        handLanmark_impl->getInputWH(W, H);
        imageAlignment(img, W, H, boxes[i], angles[i], tranMatInv, imgCrop, palmCenters[i]);

        PointList3f landmark_pixel, landmark_world;
        float score_pixel, score_world;

        handLanmark_impl->run(imgCrop, landmark_pixel, score_pixel, landmark_world, score_world);

        if (score_pixel < threshold)
            continue;

        PointList3f landmark_pixel_out;
        projectLandmarkBack(landmark_pixel, img.cols, img.rows, tranMatInv, landmark_pixel_out);

        PointList3f landmark_world_out;
        projectLandmarkBack(landmark_world, img.cols, img.rows, tranMatInv, landmark_world_out);

        // replace landmark_pixel_out z to scaled landmark_world_out z
        for (int k = 0; k < landmark_world_out.size(); ++k)
        {
            landmark_pixel_out[k].z = landmark_world_out[k].z;
        }

        BoxKp3 box = {};
        box.rect = boxes[i].rect;
        box.radians = angles[i];
        box.points = landmark_pixel_out;
        // box.points_world = landmark_world_out;
        box.score = score_pixel;

        boxLandmark.push_back(box);
    }

    // store the current hand landmark
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
}

} // namespace mpp