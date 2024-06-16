// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h

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


#ifndef MPP_REPRODUCE_HAND_LANDMARK_H
#define MPP_REPRODUCE_HAND_LANDMARK_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "pose_detector.h"
#include "smoothing.h"

namespace mpp
{

//# 0 - nose
//# 1 - left eye (inner)
//# 2 - left eye
//# 3 - left eye (outer)
//# 4 - right eye (inner)
//# 5 - right eye
//# 6 - right eye (outer)
//# 7 - left ear
//# 8 - right ear
//# 9 - mouth (left)
//# 10 - mouth (right)
//# 11 - left shoulder
//# 12 - right shoulder
//# 13 - left elbow
//# 14 - right elbow
//# 15 - left wrist
//# 16 - right wrist
//# 17 - left pinky
//# 18 - right pinky
//# 19 - left index
//# 20 - right index
//# 21 - left thumb
//# 22 - right thumb
//# 23 - left hip
//# 24 - right hip
//# 25 - left knee
//# 26 - right knee
//# 27 - left ankle
//# 28 - right ankle
//# 29 - left heel
//# 30 - right heel
//# 31 - left foot index
//# 32 - right foot index

/// PoseLandmarker performs the pose estimate on the given image. The output is the following:
/// - 33 key points of pixel value [x, y, z]
/// - the visibility, presence of 33 key points  [visibility, presence]. The visibility, presence is [0, 1] probability.
/// - the 33 key point of world landmark, instead of value in pixel, it is value in meters.
/// - the score.
class PoseLandmarker
{
public:

    /// Construct the PoseLandmarker instance.
    /// \param detector_path the hand detect model path
    /// \param landmark_path the hand landmark model path
    /// \param maxHumanNum -1 means return all detected body landmark.
    /// \param device 0 CPU, 1 GPU.
    PoseLandmarker(std::string detector_path, std::string landmark_path, int maxHumanNum, int device = 0);

    /// Construct the PoseLandmark with late model loading.
    /// \param maxHumanNum -1 means return all detected body landmark.
    /// \param device 0 means cpu.
    PoseLandmarker(int maxHumanNum, int device = 0);

    void loadDetectModel(std::string detector_path, int device = 0);
    void loadDetectModel(const char* buffer, long buffer_size, bool isTFlite, int device = 0);

    void loadLandmarkModel(std::string landmark_path, int device = 0);
    void loadLandmarkModel(const char* buffer, long buffer_size, bool isTFlite, int device = 0);

    ~PoseLandmarker();

    /// Run pose landmark model on the given image. It will detect the hand first
    /// and then estimate the pose landmark from the given bonding box.
    /// \param img the input image.
    /// \param boxLandmark pose landmark and the hand bonding box.
    void runImage(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

    /// Run pose landmark on the given image sequence. It will detect the hand for first hand.
    /// And use the pose traker to generate the bounding box for the next frame. In addition, we will use the smoother to
    /// filter the jitter.
    /// When the detected pose number hit the maxHumanNum, we will use the pose tracker instead of pose detector.
    /// \param img the input image, frame by frame.
    /// \param boxLandmark hand landmark and the pose bonding box.
    void runVideo(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

    class PoseLandmarker_Impl;

private:
    int maxHumanNum;
    int device;
    bool isTFlite;

    // compute the rotation angle from the pose detection key points.
    float computeRotateFromDetect(const BoxKp2& box);

    // compute the rotation angle from the hand landmarks.
    float computeRotateFromLandmark(const PointList3f& landmarks);

    // Align the body base on the detection 4 key points.
    void bodyAlign(const cv::Mat& src, BoxKp2& box, cv::Point2f& center, float angle, cv::Mat& transMatInv, cv::Mat& dst);

    // Run pose detector model.
    void runDetect(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles, std::vector<cv::Point2f> &bodyCenters);

    /// Try to track hands with given image. And if tracker fail to track hands, we will run handDetector.
    /// \param[in] img input image.
    /// \param[out] boxes output
    /// \param[out] angles output boxes angle one by one.
    void runTrack(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles, std::vector<cv::Point2f>& palmCenters);

    // TODO check the threshold by mediapipe.
    float threshold = 0.5f;

    cv::Ptr<OneEuroSmoother> smoother; // only use in video mode.
    std::vector<BoxKp3> preBoxPoints;
    cv::Ptr<PoseLandmarker_Impl> poseLanmark_impl = nullptr;
    cv::Ptr<PoseDetector> poseDetector = nullptr;
};

// The following is the point connection.
const std::vector<std::vector<int > > kPoseLandmarksConnections{
  {0, 4},    // (nose, right_eye_inner)
  {4, 5},    // (right_eye_inner, right_eye)
  {5, 6},    // (right_eye, right_eye_outer)
  {6, 8},    // (right_eye_outer, right_ear)
  {0, 1},    // (nose, left_eye_inner)
  {1, 2},    // (left_eye_inner, left_eye)
  {2, 3},    // (left_eye, left_eye_outer)
  {3, 7},    // (left_eye_outer, left_ear)
  {10, 9},   // (mouth_right, mouth_left)
  {12, 11},  // (right_shoulder, left_shoulder)
  {12, 14},  // (right_shoulder, right_elbow)
  {14, 16},  // (right_elbow, right_wrist)
  {16, 18},  // (right_wrist, right_pinky_1)
  {16, 20},  // (right_wrist, right_index_1)
  {16, 22},  // (right_wrist, right_thumb_2)
  {18, 20},  // (right_pinky_1, right_index_1)
  {11, 13},  // (left_shoulder, left_elbow)
  {13, 15},  // (left_elbow, left_wrist)
  {15, 17},  // (left_wrist, left_pinky_1)
  {15, 19},  // (left_wrist, left_index_1)
  {15, 21},  // (left_wrist, left_thumb_2)
  {17, 19},  // (left_pinky_1, left_index_1)
  {12, 24},  // (right_shoulder, right_hip)
  {11, 23},  // (left_shoulder, left_hip)
  {24, 23},  // (right_hip, left_hip)
  {24, 26},  // (right_hip, right_knee)
  {23, 25},  // (left_hip, left_knee)
  {26, 28},  // (right_knee, right_ankle)
  {25, 27},  // (left_knee, left_ankle)
  {28, 30},  // (right_ankle, right_heel)
  {27, 29},  // (left_ankle, left_heel)
  {30, 32},  // (right_heel, right_foot_index)
  {29, 31},  // (left_heel, left_foot_index)
  {28, 32},  // (right_ankle, right_foot_index)
  {27, 31},  // (left_ankle, left_foot_index)
};


} // namespace mpp

#endif //MPP_REPRODUCE_HAND_LANDMARK_H
