// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/pose_detector/pose_detector_graph.cc

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

#include "pose_detector.h"
#include "opencv2/dnn.hpp"
#include "non_max_suppression.h"

using namespace cv;
using namespace dnn;
namespace mpp
{

PoseDetector::PoseDetector(std::string modelPath, int _maxHumanNum, int _device)
: maxHumanNum(_maxHumanNum), device(_device)
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

    netPoseDet = makePtr<dnn::Net>(dnn::readNet(modelPath));
    this->init();
}

PoseDetector::PoseDetector(const char *buffer, const long buffer_size, bool _isTFlite, int _maxHumanNum, int _device)
: maxHumanNum(_maxHumanNum), device(_device), isTFlite(_isTFlite)
{
    if (isTFlite)
    {
        netPoseDet = makePtr<dnn::Net>(dnn::readNetFromTflite(buffer, buffer_size));
    }
    else
    {
        netPoseDet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    }

    this->init();
}

PoseDetector::~PoseDetector()
{
}

void PoseDetector::init()
{
    // Model preprocessing
    inputName = netPoseDet->getInputName();
    inputShape = netPoseDet->getInputShape();
    netPoseDet->setNumThreads(2);
    netPoseDet->setPreferablePrecision(Precision::DNN_PRECISION_LOW);

    if (device == 1)
    {
        netPoseDet->setPreferableBackend(Backend::DNN_BACKEND_GPU);
    }

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    // TODO check if the following code is correct, 224x224
    if (isTFlite)
    {
        inputWidth = inputShape[0][1];
        inputHeight = inputShape[0][2];
    }
    else
    {
        inputWidth = inputShape[0][2];
        inputHeight = inputShape[0][3];
    }

    outputName = netPoseDet->getOutputName();

    // Generate pre-defined anchors.
//    [mediapipe.SsdAnchorsCalculatorOptions.ext] {
//        num_layers: 5
//        min_scale: 0.1484375
//        max_scale: 0.75
//        input_size_height: 224
//        input_size_width: 224
//        anchor_offset_x: 0.5
//        anchor_offset_y: 0.5
//        strides: 8
//        strides: 16
//        strides: 32
//        strides: 32
//        strides: 32
//        aspect_ratios: 1.0
//        fixed_anchor_size: true
//    }
    SSDAnchorOptions ssdAnchorOptions(inputWidth, inputHeight, 0.1484375f, 0.75f, 5);
    ssdAnchorOptions.anchor_offset_x = 0.5f;
    ssdAnchorOptions.anchor_offset_y = 0.5f;
    ssdAnchorOptions.strides = {8, 16, 32, 32, 32};
    ssdAnchorOptions.aspect_ratios = {1.0f};
    ssdAnchorOptions.fixed_anchor_size = true;

    // generate the Anchor and create SSD Decoder.
    SSDDecoderOption ssdDecoderOption(1, 2254, 12);
    ssdDecoderOption.box_format = SSDDecoderOption::BoxFormat::XYWH;
    ssdDecoderOption.box_coord_offset = 0;
    ssdDecoderOption.keypoint_coord_offset = 4;
    ssdDecoderOption.num_keypoints = 4;
    ssdDecoderOption.num_values_per_keypoint = 2;
    ssdDecoderOption.sigmoid_score = true;
    ssdDecoderOption.score_clipping_thresh = 100.0f;
    ssdDecoderOption.reverse_output_order = true;
    ssdDecoderOption.min_score_thresh = min_detection_confidence;
    ssdDecoderOption.x_scale = 224.0f;
    ssdDecoderOption.y_scale = 224.0f;
    ssdDecoderOption.w_scale = 224.0f;
    ssdDecoderOption.h_scale = 224.0f;

    ssdDecoder = Ptr<SSDDecoder>(new SSDDecoder(ssdAnchorOptions, ssdDecoderOption));
}

void PoseDetector::run(const cv::Mat& img, std::vector<BoxKp2>& outBox)
{
    CV_Assert(netPoseDet && "The netPoseDet is null!");
    TickMeter m;
    m.reset();
    double t0, t1, t2;

    outBox.clear();
    if (img.empty())
    {
        CV_LOG_WARNING(NULL, "The image given in PoseDetector is empty!")
        return;
    }

    m.start();
    // step 0: Resize image unscale
    ImgScaleParams imgParams;
    Mat resizedImage;

    resizeUnscale(img, resizedImage, inputWidth, inputHeight, imgParams);
    m.stop();

    t0 = m.getTimeMilli();

    m.reset();
    m.start();
    // Update opencv_ort to opencv 4.8! normalize the (0, 255) to (-1.0, 1.0)
    Mat blob = blobFromImage(resizedImage, 1.0/127.5f, Size(), Scalar(), true);
    blob -= 1.0f;

    if (isTFlite)
    {
        transposeND(blob, {0, 2, 3, 1}, blob);
    }

    netPoseDet->setInput(blob);

    std::vector<Mat> outputBlob;

    if (isTFlite)
    {
        netPoseDet->forward(outputBlob, outputNameFromModelTF);
    }
    else
    {
        netPoseDet->forward(outputBlob, outputNameFromModelMNN);
    }

    m.stop();
    t1 = m.getTimeMilli();

    CV_LOG_DEBUG(NULL, "Finish model inference!")

    // TODO setDevice! currently, we only support CPU!
    CV_Assert(outputBlob.size() == 2);

    // Tensor2Detection: according the anchors pre-set do the

    m.reset(); m.start();
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

        // project the key points.
        for(int j=0; j<objBox.points.size(); j++)
        {
            objBox.points[j].x = std::max(std::min((objBox.points[j].x * inputWidth - dw) * div_ratio, imgW), 0.0f);
            objBox.points[j].y = std::max(std::min((objBox.points[j].y * inputHeight - dh) * div_ratio, imgH), 0.0f);
        }

        // according 4 key point to generate the bounding box
        // Key point 0 - mid hip center
        // Key point 1 - point that encodes size & rotation (for full body)
        // Key point 2 - mid shoulder center
        // Key point 3 - point that encodes size & rotation (for upper body)

        // Convert pose detected key point to rect.
        auto& rect = objBox.rect;

        const float x_center = objBox.points[0].x;
        const float y_center = objBox.points[0].y;

        const float x_scale = objBox.points[1].x;
        const float y_scale = objBox.points[1].y;

        const float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                                         (y_scale - y_center) * (y_scale - y_center)) * 2.0f * expandRatio;

        rect.x = x_center - box_size*0.5f;//std::max(x_center - box_size*0.5f, 0.f);
        rect.y = y_center - box_size*0.5f;//std::max(y_center - box_size*0.5f, 0.f);
        rect.width = box_size;
        rect.height = box_size;
    }

    // step 5: clip the output rects based on the maxHandNum
    if (maxHumanNum != -1 && maxHumanNum > 0 && maxHumanNum < outBox.size())
    {
        outBox.resize(maxHumanNum);
    }
    m.stop();
    t2 = m.getTimeMilli();

    LOGD("LOG of C++, Detector , resize takes %f ms, forward takes %f ms, post-pro takes %f ms!", t0, t1, t2);
}


} // namespace mpp