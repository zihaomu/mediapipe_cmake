// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter_graph.cc

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

#ifndef MPP_INTERACTIVE_SEGMENTER_H
#define MPP_INTERACTIVE_SEGMENTER_H

#include "common.h"
#include "opencv2/dnn.hpp"

namespace mpp
{

// Performs interactive segmentation on images.
// Users can represent user interaction through mouse click and scribble over the input image.
class InterativeSegmenter
{
public:

    /// create a interative segmenter instance.
    /// \param path model path.
    /// \param device default device num is 0.
    InterativeSegmenter(std::string modelPath, int device = 0);

    /// Override construct function, try to create instance from memory buffer.
    /// \param buffer
    /// \param buffer_size
    /// \param device
    InterativeSegmenter(const char* buffer, long buffer_size, int device = 0);

    ~InterativeSegmenter();

    /// run the interative segmenter with given input image and points over the input.
    /// \param input input image, it should be the RGB, int8_t data type.
    /// \param points input points generated by the mouse click or scribe over the image.
    /// \param output output mask, 2 channels, has the same size of the input image.
    void run(const cv::Mat& input, std::vector<cv::Point>& points, cv::Mat& output);

    /// Given the point list and input image, it will generate the mask Mat.
    /// \param inputImage input image, RGB
    /// \param pointList roi points
    /// \param outImage output mask which has the same size as the input image.
    void pointsToAlpha(const cv::Mat& inputImage, std::vector<cv::Point>& pointList, cv::Mat& outImage);

private:
    void init();
    // this function will merge the alpha channel to the given input image.
    void setAlpha(const cv::Mat& input, const cv::Mat& mask, cv::Mat& out);

    int device;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::vector<int> > outputShape;
    std::vector<std::string> outputName;

    std::vector<std::string> outputNameFromModel = {"Identity"};
    cv::Ptr<cv::dnn::Net> netSegmenter = nullptr;
};

}

#endif //MPP_INTERACTIVE_SEGMENTER_H