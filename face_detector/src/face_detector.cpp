// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc

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

#include "face_detector.h"
#include "opencv2/dnn.hpp"
#include "non_max_suppression.h"

using namespace cv;
using namespace dnn;
namespace mpp
{

FaceDetector::FaceDetector(std::string modelPath, int _maxHandNum, int _device)
: maxHandNum(_maxHandNum), device(_device)
{
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

    netFaceDet = makePtr<dnn::Net>(dnn::readNet(modelPath));
    this->init();
}

FaceDetector::FaceDetector(const char *buffer, long buffer_size, std::string model_suffix, int _maxHandNum, int _device)
: maxHandNum(_maxHandNum), device(_device), isTFlite(model_suffix == "tflite")
{
    CV_Assert(model_suffix == "tflite" || model_suffix == "mnn");
    // TODO Add the different inference type.
    if (isTFlite)
    {
        netFaceDet = makePtr<dnn::Net>(dnn::readNetFromTflite(buffer, buffer_size));
    }
    else
    {
        netFaceDet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    }

    this->init();
}

FaceDetector::~FaceDetector()
{

}

void FaceDetector::init()
{
    // Model pre-processing
    inputName = netFaceDet->getInputName();
    inputShape = netFaceDet->getInputShape();
    netFaceDet->setNumThreads(4);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    // TODO check if the following code is correct,
    if (!isTFlite) // MNN has the NCHW input data layout.
    {
        inputWidth = inputShape[0][2];
        inputHeight = inputShape[0][3];
    }
    else // tflite has NHWC input data layout
    {
        inputWidth = inputShape[0][1];
        inputHeight = inputShape[0][2];
    }


    if (inputWidth == input_size_full)
    {
        ; // do nothing, running in full model.
    }
    else if(inputWidth == input_size_short)
    {
        shortModel = true;
    }
    else
    {
        CV_LOG_ERROR(NULL, "The input size of the model is not correct!")
    }
    outputName = netFaceDet->getOutputName();

    // Generate the pre-defined Anchors
    SSDAnchorOptions ssdAnchorOptions(inputWidth, inputHeight, 0.1484375f, 0.75f, shortModel ? 4 : 1);
    ssdAnchorOptions.anchor_offset_x = 0.5f;
    ssdAnchorOptions.anchor_offset_y = 0.5f;
    ssdAnchorOptions.strides = shortModel ? std::vector<int>({8, 16, 16, 16}) : std::vector<int>({4});
    ssdAnchorOptions.aspect_ratios = {1.0f};
    ssdAnchorOptions.fixed_anchor_size = true;
    ssdAnchorOptions.interpolated_scale_aspect_ratio = shortModel ? 1.0f : 0.0f;

    // generate the Anchor and create SSD Decoder.
    SSDDecoderOption ssdDecoderOption(1, shortModel ? num_boxes_short : num_boxes_full, 16);
    ssdDecoderOption.box_format = SSDDecoderOption::BoxFormat::XYWH;
    ssdDecoderOption.box_coord_offset = 0;
    ssdDecoderOption.keypoint_coord_offset = 4;
    ssdDecoderOption.num_keypoints = 6;
    ssdDecoderOption.num_values_per_keypoint = 2;
    ssdDecoderOption.sigmoid_score = true;
    ssdDecoderOption.score_clipping_thresh = 100.0f;
    ssdDecoderOption.reverse_output_order = true;
    ssdDecoderOption.min_score_thresh = min_detection_confidence;
    ssdDecoderOption.x_scale = inputWidth;
    ssdDecoderOption.y_scale = inputWidth;
    ssdDecoderOption.w_scale = inputWidth;
    ssdDecoderOption.h_scale = inputWidth;

    ssdDecoder = Ptr<SSDDecoder>(new SSDDecoder(ssdAnchorOptions, ssdDecoderOption));
}

void FaceDetector::run(const cv::Mat& img, std::vector<BoxKp2>& outBox)
{
    CV_Assert(netFaceDet && "The netFaceDet is null!");
    outBox.clear();
    if (img.empty())
    {
        CV_LOG_WARNING(NULL, "The image given in FaceDetector is empty!")
        return;
    }

    // step 0: Resize image unscale
    ImgScaleParams imgParams;
    Mat resizedImage;
    resizeUnscale(img, resizedImage, inputWidth, inputHeight, imgParams);

    // Update opencv_ort to opencv 4.8!
    Mat blob = blobFromImage(resizedImage, 1.0/255.0, Size(), Scalar(), true);

    if (isTFlite)  // convert NCHW to NHWC
    {
        transposeND(blob, {0, 2, 3, 1}, blob);
    }

    netFaceDet->setInput(blob);

    std::vector<Mat> outputBlob;
    if (shortModel)
        netFaceDet->forward(outputBlob, outputNameFromModel_short);
    else
        netFaceDet->forward(outputBlob, outputNameFromModel_full);

    CV_LOG_DEBUG(NULL, "Finish model inference!")

    // TODO setDevice! currently, we only support CPU!
    CV_Assert(outputBlob.size() == 2);

    // step 1: Tensor2Detection, process anchor!
    ssdDecoder->run(outputBlob[0], outputBlob[1], outBox);

    // step 2: NMS
    nms_simple(outBox, min_suppression_threshold);

    // step 3: projects detection back input image.
    float div_ratio = 1.0/imgParams.ratio;
    int dw = imgParams.dw;
    int dh = imgParams.dh;

    float imgW = img.cols - 1.f;
    float imgH = img.rows - 1.f;

    int realBoxNum = outBox.size();
    for(int i =0; i < realBoxNum; i++)
    {
        auto& objBox = outBox[i];
        auto& rect = objBox.rect;

        float xmin = rect.x * inputWidth;
        float ymin = rect.y * inputHeight;
        float xmax = (rect.x + rect.width) * inputWidth;
        float ymax = (rect.y + rect.height) * inputHeight;

        // project the face box
        xmin = std::max((xmin - dw) * div_ratio, 0.0f);
        ymin = std::max((ymin - dh) * div_ratio, 0.0f);
        xmax = std::min((xmax - dw) * div_ratio, imgW);
        ymax = std::min((ymax - dh) * div_ratio, imgH);

        rect.x = xmin, rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        for(int j=0; j<objBox.points.size(); j++)
        {
            objBox.points[j].x = std::max(std::min((objBox.points[j].x * inputWidth - dw) * div_ratio, imgW), 0.0f);
            objBox.points[j].y = std::max(std::min((objBox.points[j].y * inputHeight - dh) * div_ratio, imgH), 0.0f);
        }
    }

    // step 5: clip the output rects based on the maxHandNum
    if (maxHandNum != -1 && maxHandNum > 0 && maxHandNum < outBox.size())
    {
        outBox.resize(maxHandNum);
    }
}


} // namespace mpp