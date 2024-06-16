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

#ifndef MPP_CMAKE_IRIS_TRACKER_IMPL_H
#define MPP_CMAKE_IRIS_TRACKER_IMPL_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "iris_tracker.h"

namespace mpp
{

class IrisTracker::IrisLandmarker_Impl
{
public:
    IrisLandmarker_Impl(std::string modelPath, int device = 0);
    ~IrisLandmarker_Impl();

    /// Given the eye cropped image.
    /// \param img cropped image
    /// \param landmarkIris 5 iris 2d landmark, the final value is point socre.
    /// \param landmarkEye 71 eye 3d and brow landmarks.
    void run(const cv::Mat& img, PointList3f& landmarkIris, PointList3f& landmarkEye);

    void getInputWH(int& W, int& H);
    const int IRIS_LANDMARK_NUM = 5;
    const int EYE_LANDMARK_NUM = 71;
private:
    void init();

    int device;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::string> outputName;

    std::vector<std::vector<int> > inputShape;
    std::vector<std::vector<int> > outputShape;

    std::vector<std::string> outputNameFromModel = {"output_eyes_contours_and_brows", "output_iris"};

    cv::Ptr<cv::dnn::Net> net = nullptr;
};


}

#endif //MPP_CMAKE_IRIS_TRACKER_IMPL_H
