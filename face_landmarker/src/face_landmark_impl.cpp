// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc

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

#include "face_landmark_impl.h"

using namespace cv;
using namespace dnn;
namespace mpp
{
FaceLandmarker_Impl::FaceLandmarker_Impl(std::string modelPath, int device)
{
    // Load MNN model by opencv.
    netFaceDet = makePtr<dnn::Net>(dnn::readNetFromMNN(modelPath));
    this->init();
}

FaceLandmarker_Impl::FaceLandmarker_Impl(const char* buffer, long buffer_size, int device)
{
    // Load MNN model by opencv from buffer.
    netFaceDet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    this->init();
}

void FaceLandmarker_Impl::init()
{
    inputName = netFaceDet->getInputName();
    inputShape = netFaceDet->getInputShape();
    netFaceDet->setNumThreads(4);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    // TODO check if the following code is correct,
    inputWidth = inputShape[0][2]; // 224
    inputHeight = inputShape[0][3]; // 224

    outputName = netFaceDet->getOutputName();

    CV_Assert(outputName.size() == outputNameFromModel_468.size());

    landmarkSize = 468;
}

FaceLandmarker_Impl::~FaceLandmarker_Impl()
{

}

void FaceLandmarker_Impl::getInputWH(int &W, int &H)
{
    W = inputWidth;
    H = inputHeight;
}

int FaceLandmarker_Impl::getLandmarkSize()
{
    return landmarkSize;
}

static float get_average_z(const std::vector<int>& index, PointList3f& landmark)
{
    float z_average = 0;
    for (int i = 0; i < index.size(); ++i)
    {
        z_average += landmark[index[i]].z;
    }

    return z_average / index.size();
}

void decodeLandmark(const float* p, int size, PointList3f& landmark)
{
    landmark.clear();
    landmark.resize(size);
    for (int i = 0; i < size; i++)
    {
        landmark[i] = Point3f{p[i * 3], p[i * 3 + 1], p[i * 3 + 2]};
    }
}

void FaceLandmarker_Impl::run(const cv::Mat &img,
                              mpp::PointList3f &landmark, float &landmark_score)
{
    landmark.clear();
    Mat blob = blobFromImage(img, 1.0/255.0, Size(), Scalar(), true);
    netFaceDet->setInput(blob);
    
    std::vector<Mat> out;

    {
        netFaceDet->forward(out, outputNameFromModel_468);
        CV_Assert(out.size() == outputNameFromModel_468.size());
        landmark_score = out[1].at<float>(0, 0);
        decodeLandmark((float*)out[0].data, landmarkSize, landmark);
    }
}

} // namespace mpp