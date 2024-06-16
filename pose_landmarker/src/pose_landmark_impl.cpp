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

#include "pose_landmark_impl.h"

using namespace cv;
using namespace dnn;
namespace mpp
{

PoseLandmarker::PoseLandmarker_Impl::PoseLandmarker_Impl(std::string modelPath, int _device)
{
    device = _device;
    // Load MNN model by opencv.
    const std::string modelExt = modelPath.substr(modelPath.rfind('.') + 1);

    if (modelExt == "tflite")
    {
        isTFlite = true;
    }
    else if (modelExt == "mnn")
    {
        isTFlite = false;
    }
    else
        CV_Error(Error::Code::StsNotImplemented, "Not supported model type!");

    landmarkNet = makePtr<dnn::Net>(dnn::readNetFromMNN(modelPath));

    this->init();
}

PoseLandmarker::PoseLandmarker_Impl::PoseLandmarker_Impl(const char *buffer, const long buffer_size, bool _isTFlite , int _device)
{
    device = _device;
    // Load MNN model by opencv from buffer.
    isTFlite = _isTFlite;
    if (isTFlite)
    {
        landmarkNet = makePtr<dnn::Net>(dnn::readNetFromTflite(buffer, buffer_size));
    }
    else
    {
        landmarkNet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    }
    this->init();
}

void PoseLandmarker::PoseLandmarker_Impl::init()
{
    landmarkNet->setNumThreads(4);
    landmarkNet->setPreferablePrecision(Precision::DNN_PRECISION_LOW);

    if (device == 1)
    {
        landmarkNet->setPreferableBackend(Backend::DNN_BACKEND_GPU);
    }

    inputName = landmarkNet->getInputName();
    inputShape = landmarkNet->getInputShape();
    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);
    // TODO check if the following code is correct,

    if (isTFlite)
    {
        inputWidth = inputShape[0][1]; // 256
        inputHeight = inputShape[0][2]; // 256
    }
    else
    {
        inputWidth = inputShape[0][2]; // 256
        inputHeight = inputShape[0][3]; // 256
    }

    outputName = landmarkNet->getOutputName();
}

PoseLandmarker::PoseLandmarker_Impl::~PoseLandmarker_Impl()
{

}

inline void recoveryLandmark3(const Mat& m, int keypointSize, PointList3f& landmark)
{
    const float* data = m.ptr<float>();
    landmark.resize(keypointSize);
    for (int i = 0; i < keypointSize; i++)
    {
        landmark[i] = Point3f{data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
    }
}

inline double sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline void recoveryLandmark5(const Mat& m, int keypointSize, PointList3f& landmark, PointList2f& vp)
{
    const float* data = m.ptr<float>();
    landmark.resize(keypointSize);
    vp.resize(keypointSize);
    for (int i = 0; i < keypointSize; i++)
    {
        landmark[i] = Point3f{data[i * 5], data[i * 5 + 1], data[i * 5 + 2]};
        vp[i] = Point2f {(float )sigmoid(data[i * 5 + 3]), (float )sigmoid(data[i * 5 + 4])};
    }
}

void PoseLandmarker::PoseLandmarker_Impl::getInputWH(int& W, int& H)
{
    H = inputHeight;
    W = inputWidth;
}

void PoseLandmarker::PoseLandmarker_Impl::run(const cv::Mat& img, PointList3f& landmark_pixel, PointList3f& landmark_world,
                                              PointList2f& visPre, float& landmark_score)
{
    landmark_pixel.clear();
    landmark_world.clear();

    Mat blob = blobFromImage(img, 1.0/255.0, Size(), Scalar(), true);

    if (isTFlite)
    {
        transposeND(blob, {0, 2, 3, 1}, blob);
    }
    landmarkNet->setInput(blob);

    std::vector<Mat> outputBlob;

    landmarkNet->forward(outputBlob, outputName);

    CV_Assert(outputBlob.size() == outputName.size());

    // recovery pixel landmarks
    recoveryLandmark5(outputBlob[0], landmarkSize, landmark_pixel, visPre);

    // recovery world landmarks
    recoveryLandmark3(outputBlob[2], landmarkSize, landmark_world);

    // set score
    landmark_score = outputBlob[1].at<float>();
}

}