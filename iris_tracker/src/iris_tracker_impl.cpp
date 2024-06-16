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


#include "iris_tracker_impl.h"

using namespace cv;
namespace mpp
{

IrisTracker::IrisLandmarker_Impl::IrisLandmarker_Impl(std::string modelPath, int _device)
: device(_device)
{
    net = makePtr<dnn::Net>(dnn::readNet(modelPath));
    this->init();
}

void IrisTracker::IrisLandmarker_Impl::init()
{
    inputName = net->getInputName();
    inputShape = net->getInputShape();
    outputShape = net->getOutputShape();

    net->setNumThreads(4);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    inputWidth = inputShape[0][2]; // 64
    inputHeight = inputShape[0][3]; // 64

    outputName = net->getOutputName();
}

inline void decodeLandmark(const float* p, int size, PointList3f& landmark)
{
    landmark.clear();
    landmark.resize(size);
    for (int i = 0; i < size; i++)
    {
        landmark[i] = Point3f{p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
    }
}

void IrisTracker::IrisLandmarker_Impl::run(const cv::Mat &img, mpp::PointList3f &landmarkIris,
                                           mpp::PointList3f &landmarkEye)
{
    CV_Assert(net && "The iris landmarker model is null!");
    landmarkIris.clear();
    landmarkEye.clear();

    Mat blob = dnn::blobFromImage(img, 1.0/255.0, Size(inputWidth, inputHeight), Scalar(), true);

    net->setInput(blob);
    std::vector<Mat> outs;
    net->forward(outs, outputNameFromModel);

    CV_Assert(outs.size() == 2);

    decodeLandmark((const float *)outs[0].data, EYE_LANDMARK_NUM, landmarkEye);
    decodeLandmark((const float *)outs[1].data, IRIS_LANDMARK_NUM, landmarkIris);
}

void IrisTracker::IrisLandmarker_Impl::getInputWH(int &W, int &H)
{
    W = inputWidth;
    H = inputHeight;
}


IrisTracker::IrisLandmarker_Impl::~IrisLandmarker_Impl()
{

}



}