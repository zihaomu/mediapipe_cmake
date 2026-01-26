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

#include "selfie_segmenter.h"

using namespace cv;

namespace mpp
{

SelfieSegmenter::SelfieSegmenter(std::string modelPath, int _device)
: device(_device)
{
    netSegmenter = makePtr<dnn::Net>(dnn::readNet(modelPath));
    this->init();
}

SelfieSegmenter::SelfieSegmenter(const char* buffer, long buffer_size, std::string model_suffix, int _device)
: device(_device)
{
    CV_Assert(model_suffix == "tflite");
    netSegmenter = makePtr<dnn::Net>(dnn::readNetFromTflite(buffer, buffer_size));
    this->init();
}

void SelfieSegmenter::init()
{
    inputName = netSegmenter->getInputName();
    inputShape = netSegmenter->getInputShape();
    outputShape = netSegmenter->getOutputShape();

    netSegmenter->setNumThreads(4);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    inputWidth = inputShape[0][1]; // 256
    inputHeight = inputShape[0][2]; // 256

    outputName = netSegmenter->getOutputName();
}

SelfieSegmenter::~SelfieSegmenter()
{
}

void SelfieSegmenter::run(const cv::Mat &input, cv::Mat &output)
{
    CV_Assert(netSegmenter && "The netSegmenter is null!");

    Mat blob = dnn::blobFromImage(input, 1.0/255.0, Size(inputWidth, inputHeight), Scalar(), true);

    Mat blobNHWC;

    transposeND(blob, {0, 2, 3, 1}, blobNHWC);

    netSegmenter->setInput(blobNHWC);

    Mat tmp = netSegmenter->forward();
    Mat outMask = Mat(inputHeight, inputWidth, CV_32FC1, tmp.data);
    resize(outMask, output, input.size());
}

}
