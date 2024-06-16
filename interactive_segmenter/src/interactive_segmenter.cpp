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

#include "interactive_segmenter.h"

using namespace cv;

namespace mpp
{

// Calculater the point thickness, so that we can generate the image mask that is scale invariant to the input image.
static inline float calculatePointThickness(const Mat& inputImage, int modelInput_W, int modelInput_H)
{
    int img_W = inputImage.cols;
    int img_H = inputImage.rows;

    return std::max(std::max(img_W * 1.f / modelInput_W, img_H * 1.f / modelInput_H), 1.0f);
}

InterativeSegmenter::InterativeSegmenter(std::string modelPath, int _device)
: device(_device)
{
    netSegmenter = makePtr<dnn::Net>(dnn::readNet(modelPath));
    this->init();
}

InterativeSegmenter::InterativeSegmenter(const char* buffer, long buffer_size, int _device)
: device(_device)
{
    netSegmenter = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    this->init();
}

void InterativeSegmenter::init()
{
    inputName = netSegmenter->getInputName();
    inputShape = netSegmenter->getInputShape();
    outputShape = netSegmenter->getOutputShape();
    netSegmenter->setNumThreads(8);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    inputWidth = inputShape[0][2]; // 512
    inputHeight = inputShape[0][3]; // 512

    outputName = netSegmenter->getOutputName();
}

InterativeSegmenter::~InterativeSegmenter()
{
}

void InterativeSegmenter::run(const cv::Mat &input, std::vector<cv::Point> &points, cv::Mat &output)
{
    CV_Assert(netSegmenter && "The netSegmenter is null!");
    Mat mask;
    pointsToAlpha(input, points, mask);
    Mat inpA;
    setAlpha(input, mask, inpA);

    Mat blob = dnn::blobFromImage(inpA, 1.0/255.0, Size(inputWidth, inputHeight), Scalar(), true);

    netSegmenter->setInput(blob);

    output = netSegmenter->forward();
    Mat maskOut = Mat(inputHeight, inputHeight, CV_32FC1, output.data);

    // resize back to raw image.
    resize(maskOut, output, input.size());
}

void InterativeSegmenter::setAlpha(const cv::Mat &input, const cv::Mat &mask, cv::Mat &out)
{
    CV_Assert(!input.empty() && !mask.empty());
    CV_Assert(mask.channels() == 1);
    CV_Assert(input.cols == mask.cols && input.rows == mask.rows);

    int inpChannels = input.channels();
    CV_Assert(inpChannels == 3 || inpChannels == 4);
    CV_Assert(input.depth() == mask.depth());

    std::vector<Mat> matChannels;
    cv::split(input, matChannels);

    // TODO Add the BGR and RGB.
    if (matChannels.size() == 3) // assume the input is BGR
    {
       Mat tmp = matChannels[2];
       matChannels[2] = matChannels[0];
       matChannels[0] = tmp;
       matChannels.push_back(mask);
    }
    else
       matChannels[3] = mask;

    merge(matChannels, out);
}

// Convert the point list to the image mask.
// TODO change the points only to Line and Point.
void InterativeSegmenter::pointsToAlpha(const cv::Mat &inputImage, std::vector<cv::Point> &pointList,
                                        cv::Mat &outImage)
{
    int imgH = inputImage.rows;
    int imgW = inputImage.cols;

    // Generate the empty image.
    outImage = Mat::zeros(imgH, imgW, CV_8UC1);
    float thickness = calculatePointThickness(inputImage, inputWidth, inputHeight);

    int pointNum = pointList.size();
    for (int i = 0; i < pointNum; i++)
    {
        circle(outImage, pointList[i], (int)thickness, Scalar(255), -1);
    }
}

}
