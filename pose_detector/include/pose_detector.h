// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/pose_detector/pose_detector_graph.cc

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

#ifndef MPP_REPRODUCE_POSE_DETECTOR_H
#define MPP_REPRODUCE_POSE_DETECTOR_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "ssd_post_process.h"

namespace mpp
{

class PoseDetector
{
public:
    /// Create PoseDetector with given model path.
    /// \param modelPath path to hand detect model.
    PoseDetector(std::string modelPath, int maxHumanNum = -1, int device = 0);

    /// override by loading model from memory.
    PoseDetector(const char *buffer, const long buffer_size, bool isTFlite, int maxHumanNum = -1, int device = 0);
    ~PoseDetector();

    void run(const cv::Mat& img, std::vector<BoxKp2>& rects);

private:
    void init();

    int maxHumanNum;
    int device;
    bool isTFlite;

    float min_suppression_threshold = 0.3f;
    float min_detection_confidence = 0.5f;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;

    float expandRatio = 1.25;

    std::vector<std::string> outputNameFromModelMNN = {"Identity:0", "Identity_1:0"};
    std::vector<std::string> outputNameFromModelTF = {"Identity", "Identity_1"};
    cv::Ptr<cv::dnn::Net> netPoseDet = nullptr;
    cv::Ptr<SSDDecoder> ssdDecoder = nullptr;
};

} // namespace mpp

#endif //MPP_REPRODUCE_POSE_DETECTOR_H
