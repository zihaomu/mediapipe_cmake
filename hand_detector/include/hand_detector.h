// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc

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
#ifndef MPP_REPRODUCE_HAND_DETECTOR_H
#define MPP_REPRODUCE_HAND_DETECTOR_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "ssd_post_process.h"

namespace mpp
{

class HandDetector
{
public:
    /// Create HandDetector with given model path.
    /// \param modelPath path to hand detect model.
    HandDetector(std::string modelPath, int maxHandNum = -1, int device = 0);

    /// Override of loading model from buffer.
    /// \param buffer memory buffer pointer.
    /// \param buffer_size buffer size
    /// \param maxHandNum -1 means return all hands that have been detected.
    /// \param device default 0
    HandDetector(const char* buffer, long buffer_size, int maxHandNum = -1, int device = 0);
    ~HandDetector();

    void run(const cv::Mat& img, std::vector<BoxKp2>& rects);

private:
    void init();

    int maxHandNum;
    int device;

    float min_suppression_threshold = 0.2f;
    float min_detection_confidence = 0.5f;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;

    float expandRatio = 2.6;
    float shift_y = -0.25;

    std::vector<std::string> outputNameFromModel = {"Identity", "Identity_1"};
    cv::Ptr<cv::dnn::Net> netHandDet = nullptr;
    cv::Ptr<SSDDecoder> ssdDecoder = nullptr;
};

} // namespace mpp

#endif //MPP_REPRODUCE_HAND_DETECTOR_H
