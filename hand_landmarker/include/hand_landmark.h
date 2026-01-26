// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h

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

#ifndef MPP_REPRODUCE_HAND_LANDMARK_H
#define MPP_REPRODUCE_HAND_LANDMARK_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "hand_detector.h"
#include "smoothing.h"

namespace mpp
{

// Hand model 21 landmark names.
enum class HandLandmarkName {
  kWrist = 0,
  kThumb1,
  kThumb2,
  kThumb3,
  kThumb4,
  kIndex1,
  kIndex2,
  kIndex3,
  kIndex4,
  kMiddle1,
  kMiddle2,
  kMiddle3,
  kMiddle4,
  kRing1,
  kRing2,
  kRing3,
  kRing4,
  kPinky1,
  kPinky2,
  kPinky3,
  kPinky4,
};

class HandLandmarker_Impl;
class HandLandmarker
{
public:

    /// Construct the HandLandmarker instance.
    /// \param detector_path the hand detect model path
    /// \param landmark_path the hand landmark model path
    /// \param maxHandNum -1 means return all detected hand landmark.
    /// \param device 0 means cpu.
    HandLandmarker(std::string detector_path, std::string landmark_path, int maxHandNum, int device = 0);

    // create instance with late model loading.
    HandLandmarker(int maxHandNum, int device = 0);
    void loadDetectModel(std::string detector_path);
    void loadDetectModel(const char* buffer, long buffer_size, std::string model_suffix);

    void loadLandmarkModel(std::string landmark_path);
    void loadLandmarkModel(const char* buffer, long buffer_size, std::string model_suffix);

    ~HandLandmarker();

    /// Run hand landmark model on the given image. It will detect the hand first
    /// and then estimate the hand landmark from the given bonding box.
    /// \param img the input image.
    /// \param boxLandmark hand landmark and the hand bonding box.
    void runImage(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

    /// Run hand landmark on the given image sequence. It will detect the hand for first hand.
    /// And use the hand traker to generate the bounding box for the next frame.
    /// When the detected hand number hit the maxHandNum, we will use the hand tracker instead of hand detector.
    /// \param img the input image, frame by frame.
    /// \param boxLandmark hand landmark and the hand bonding box.
    void runVideo(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

private:
    int maxHandNum;
    int device;

    // compute the rotation angle from the hand detection keypoints.
    float computeRotateFromDetect(const BoxKp2& box);

    // compute the rotation angle from the hand landmarks.
    float computeRotateFromLandmark(const PointList3f& landmarks);

    // Run hand detector model.
    void runDetect(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles, std::vector<cv::Point2f>& palmCenters);

    /// Try to track hands with given image. And if tracker fail to track hands, we will run handDetector.
    /// \param[in] img input image.
    /// \param[out] boxes output
    /// \param[out] angles output boxes angle one by one.
    void runTrack(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles, std::vector<cv::Point2f>& palmCenters);

    // TODO check the threshold by mediapipe.
    float threshold = 0.5f;

    cv::Ptr<OneEuroSmoother> smoother; // only use in video mode.
    std::vector<BoxKp3> preBoxPoints;
    cv::Ptr<HandLandmarker_Impl> handLanmark_impl = nullptr;
    cv::Ptr<HandDetector> handDetector = nullptr;
};

const std::vector<std::vector<int > > kHandConnections {
    {0, 1},   {0, 5},   {9, 13},  {13, 17}, {5, 9},   {0, 17},  {1, 2},
    {2, 3},   {3, 4},   {5, 6},   {6, 7},   {7, 8},   {9, 10},  {10, 11},
    {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}
};


} // namespace mpp

#endif //MPP_REPRODUCE_HAND_LANDMARK_H
