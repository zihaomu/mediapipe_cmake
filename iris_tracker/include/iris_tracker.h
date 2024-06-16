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

#ifndef MPP_HAIR_SEGMENTER_H
#define MPP_HAIR_SEGMENTER_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "face_landmark.h"

namespace mpp
{
enum EYE_TYPE
{
    LEFT_EYE = 0,
    RIGHT_EYE = 1,
};

struct IrisOutput
{
    EYE_TYPE eyeType;
    PointList3f landmarkIris; // 5 2d points, the final value is the socre.
    PointList3f landmarkEye;  // 71 3d points.
};

// The IrisTracker class can give the iris landmark with given input image.
// This class needs the face detector and face landmark to locate the eye roi,
// and then crop the eye image as the iris_landmaker.tflite input.
// NOTE: IrisTracker can only handle the single face scene.
class IrisTracker
{
public:

    /// create a IrisTracker instance.
    /// \param iris_landmark_path
    /// \param face_detect_path
    /// \param face_landmark_path
    /// \param device default CPU device num is 0, GPU is 1.
    IrisTracker(std::string iris_landmark_path, std::string face_detect_path, std::string face_landmark_path, int device = 0);

    ~IrisTracker();

    /// run the interative segmenter with given input image and points over the input.
    /// \param input input image, it should be the RGB, int8_t data type.
    /// \param output output mask, 2 channels, has the same size of the input image.
    void runImage(const cv::Mat& input, std::vector<IrisOutput>& output);

    void runVideo(const cv::Mat& input, std::vector<IrisOutput>& output);

    class IrisLandmarker_Impl;

private:
    int device;
    cv::Ptr<IrisLandmarker_Impl> irisLandmarker_impl = nullptr;
    cv::Ptr<FaceLandmarker> faceLandmarker = nullptr;

//    const std::vector<int> right_eye_boundary_index = {362, 263};
//    const std::vector<int> left_eye_boundary_index = {33, 133};
    const std::vector<int> left_eye_boundary_index = {362, 263};
    const std::vector<int> right_eye_boundary_index = {33, 133};
    const float scaleFactorEyeRoi = 2.3;  // to scale the eye roi.
};

}

#endif //MPP_HAIR_SEGMENTER_H
