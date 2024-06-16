// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/face_detector/face_detector.h

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

#ifndef MPP_REPRODUCE_FACE_DETECTOR_H
#define MPP_REPRODUCE_FACE_DETECTOR_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "ssd_post_process.h"

namespace mpp
{

// BlazeFace detector. Output the face bounding box and 6 key points.
class FaceDetector
{
public:
    FaceDetector(std::string modelPath, int maxFaceNum = -1, int device = 0);
    FaceDetector(const char *buffer, long buffer_size, bool isTFlite = false, int maxFaceNum = -1, int device = 0);
    ~FaceDetector();

    void run(const cv::Mat& img, std::vector<BoxKp2>& rects);

private:
    bool shortModel = false; // whether use the short model or full model.
    void init();

    int maxHandNum;
    int device;

    bool isTFlite;

    const int input_size_short = 128;
    const int input_size_full = 192;

    const int num_boxes_full = 2304;
    const int num_boxes_short = 896;

    float min_suppression_threshold = 0.3f;
    float min_detection_confidence = 0.7f;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;

    std::vector<std::string> outputNameFromModel_full = {"reshaped_regressor_face_4", "reshaped_classifier_face_4"};
    std::vector<std::string> outputNameFromModel_short = {"regressors", "classificators"};
    cv::Ptr<cv::dnn::Net> netFaceDet = nullptr;
    cv::Ptr<SSDDecoder> ssdDecoder = nullptr;
};

} // namespace mpp

#endif //MPP_REPRODUCE_FACE_DETECTOR_H
