// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/selfie_segmentation/selfie_segmentation_cpu.pbtxt

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

namespace mpp
{

// This class is used to carry selfie segmentation from the input image.
// ATTETNION: The selfie_segmentation.tflite model only works for opencv_lite with tflite backend.
class SelfieSegmenter
{
public:

    /// create a SelfieSegmenter instance.
    /// \param path model path.
    /// \param device default CPU device num is 0, GPU is 1.
    SelfieSegmenter(std::string modelPath, int device = 0);

    /// Override construct function, try to create instance from memory buffer.
    /// \param buffer
    /// \param buffer_size
    /// \param model_suffix only support tflite now.
    /// \param device
    SelfieSegmenter(const char* buffer, long buffer_size, std::string model_suffix, int device = 0);

    ~SelfieSegmenter();

    /// run the interative segmenter with given input image and points over the input.
    /// \param input input image, it should be the RGB, int8_t data type.
    /// \param output output mask, 2 channels, has the same size of the input image.
    void run(const cv::Mat& input, cv::Mat& output);

private:
    void init();

    int device;
    int inputHeight, inputWidth;

    bool isTFlite = true; // TODO support opencv_lite default MNN backend.
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::vector<int> > outputShape;
    std::vector<std::string> outputName;

    std::vector<std::string> outputNameFromModel = {"activation_10"};
    cv::Ptr<cv::dnn::Net> netSegmenter = nullptr;
};

}

#endif //MPP_HAIR_SEGMENTER_H
