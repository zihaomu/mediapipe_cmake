// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_graph.cc

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

#ifndef MPP_REPRODUCE_POSE_LANDMARK_IMPL_H
#define MPP_REPRODUCE_POSE_LANDMARK_IMPL_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "pose_landmark.h"

namespace mpp
{

class PoseLandmarker::PoseLandmarker_Impl
{
public:
    PoseLandmarker_Impl(std::string modelPath, int device = 0);
    PoseLandmarker_Impl(const char* buffer, const long buffer_size, bool isTFlite, int device = 0);
    ~PoseLandmarker_Impl();

    void init();

    /// run the hand landmarker with given croped input image.
    /// \param img the croped input image, required RGB format with size 224x224.
    /// \param landmark_pixel [21 x 3], landmarks provides coordinates (in pixel) of a 3D object projected on
    /// to 2D image surface.
    /// \param landmark_pixel_score handedness score of landmark_pixel
    /// \param landmark_world 21 x 3, landmarks provides coordinates (in meter) of the 3D object itself.
    /// \param landmark_world_score handedness score of landmark_world
    void run(const cv::Mat& img, PointList3f& landmark_pixel, PointList3f& landmark_world,
             PointList2f& visPre, float& landmark_score);

    void getInputWH(int& W, int& H);

    static constexpr int landmarkSize = 33;

private:
    bool isTFlite;
    int device;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;

    // un-simplified version, has heat map and mask output.
    // 1x195, 1x1, 1x256x256x1, 1x64x64x39, 1x117.
//    std::vector<std::string> outputNameFromModel = {"Identity", "Identity_1", "Identity_2", "Identity_3", "Identity_4"};

    // 1x195, 1x1, 1x117.
    // simplified version, has removed heat map and mask output to speed up inference.
//    std::vector<std::string> outputNameFromModel = {"Identity", "Identity_1", "Identity_4"};
//    std::vector<std::string> outputNameFromModel = {"Identity", "Identity_1", "Identity_4"};
    cv::Ptr<cv::dnn::Net> landmarkNet = nullptr;
};

}

#endif //MPP_REPRODUCE_POSE_LANDMARK_IMPL_H
