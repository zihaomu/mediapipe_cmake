// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_graph.cc

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

#include "hand_landmark_impl.h"

using namespace cv;
using namespace dnn;
namespace mpp
{

HandLandmarker_Impl::HandLandmarker_Impl(std::string modelPath, int device)
{
    // Load MNN model by opencv.
    netHandDet = makePtr<dnn::Net>(dnn::readNetFromMNN(modelPath));
    this->init();
}

HandLandmarker_Impl::HandLandmarker_Impl(const char* buffer, long buffer_size, std::string model_suffix, int device)
{
    CV_Assert(model_suffix == "mnn");
    // Load MNN model by opencv.
    netHandDet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    this->init();
}

void HandLandmarker_Impl::init()
{
    inputName = netHandDet->getInputName();
    inputShape = netHandDet->getInputShape();
    netHandDet->setNumThreads(4);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    // TODO check if the following code is correct,
    inputWidth = inputShape[0][2]; // 224
    inputHeight = inputShape[0][3]; // 224

    outputName = netHandDet->getOutputName();
}

HandLandmarker_Impl::~HandLandmarker_Impl()
{
}

inline void recoveryLandmark(const Mat& m, PointList3f& landmark)
{
    const float* data = m.ptr<float>();

    landmark.resize(HandLandmarker_Impl::landmarkSize);
    for (int i = 0; i < HandLandmarker_Impl::landmarkSize; i++)
    {
        landmark[i] = Point3f{data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
    }
}

void HandLandmarker_Impl::getInputWH(int& W, int& H)
{
    H = inputHeight;
    W = inputWidth;
}

void HandLandmarker_Impl::run(const cv::Mat& img, PointList3f& landmark_pixel, float& landmark_pixel_score,
                              PointList3f& landmark_world, float& landmark_world_score)
{
    landmark_pixel.clear();
    landmark_world.clear();

    Mat blob = blobFromImage(img, 1.0/255.0, Size(), Scalar(), true);

    netHandDet->setInput(blob);

    std::vector<Mat> outputBlob;
    netHandDet->forward(outputBlob, outputNameFromModel);

    CV_Assert(outputBlob.size() == 4);

    // recovery pixel landmarks
    recoveryLandmark(outputBlob[0], landmark_pixel);

    // recovery world landmarks
    recoveryLandmark(outputBlob[3], landmark_world);

    // set score
    landmark_pixel_score = outputBlob[1].at<float>();
    landmark_world_score = outputBlob[2].at<float>();
}

}