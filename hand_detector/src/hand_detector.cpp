// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc

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

#include "hand_detector.h"
#include "opencv2/dnn.hpp"
#include "non_max_suppression.h"

using namespace cv;
using namespace dnn;
namespace mpp
{

HandDetector::HandDetector(std::string modelPath, int _maxHandNum, int _device)
: maxHandNum(_maxHandNum), device(_device)
{
    // Load MNN model by opencv.
    netHandDet = makePtr<dnn::Net>(dnn::readNetFromMNN(modelPath));
    this->init();
}

HandDetector::HandDetector(const char* buffer, long buffer_size, int _maxHandNum, int _device)
        : maxHandNum(_maxHandNum), device(_device)
{
    // Load MNN model by opencv from buffer.
    netHandDet = makePtr<dnn::Net>(dnn::readNetFromMNN(buffer, buffer_size));
    this->init();
}

HandDetector::~HandDetector()
{

}

void HandDetector::init()
{
    // model pre-processing.
    inputName = netHandDet->getInputName();
    inputShape = netHandDet->getInputShape();
    netHandDet->setNumThreads(8);

    CV_Assert(inputShape.size() == 1);
    CV_Assert(inputShape[0].size() == 4);

    // TODO check if the following code is correct,
    inputWidth = inputShape[0][2];
    inputHeight = inputShape[0][3];

    outputName = netHandDet->getOutputName();

    // Generate predefined anchors.
    SSDAnchorOptions ssdAnchorOptions(inputWidth, inputHeight, 0.1484375f, 0.75f, 4);
    ssdAnchorOptions.anchor_offset_x = 0.5f;
    ssdAnchorOptions.anchor_offset_y = 0.5f;
    ssdAnchorOptions.strides = {8, 16, 16, 16};
    ssdAnchorOptions.aspect_ratios = {1.0f};
    ssdAnchorOptions.fixed_anchor_size = true;

    // generate the Anchor and create SSD Decoder.
    SSDDecoderOption ssdDecoderOption(1, 2016, 18);
    ssdDecoderOption.box_format = SSDDecoderOption::BoxFormat::XYWH;
    ssdDecoderOption.box_coord_offset = 0;
    ssdDecoderOption.keypoint_coord_offset = 4;
    ssdDecoderOption.num_keypoints = 7;
    ssdDecoderOption.num_values_per_keypoint = 2;
    ssdDecoderOption.sigmoid_score = true;
    ssdDecoderOption.score_clipping_thresh = 100.0f;
    ssdDecoderOption.reverse_output_order = true;
    ssdDecoderOption.min_score_thresh = min_detection_confidence;
    ssdDecoderOption.x_scale = 192.0f;
    ssdDecoderOption.y_scale = 192.0f;
    ssdDecoderOption.w_scale = 192.0f;
    ssdDecoderOption.h_scale = 192.0f;

    ssdDecoder = Ptr<SSDDecoder>(new SSDDecoder(ssdAnchorOptions, ssdDecoderOption));
}

void HandDetector::run(const cv::Mat& img, std::vector<BoxKp2>& outBox)
{
    CV_Assert(netHandDet && "The netHandDet is null!");

    outBox.clear();
    if (img.empty())
    {
        CV_LOG_WARNING(NULL, "The image given in HandDetector is empty!")
        return;
    }

    // step 0: Resize image unscale
    ImgScaleParams imgParams;
    Mat resizedImage;
    resizeUnscale(img, resizedImage, inputWidth, inputHeight, imgParams);

    // Update opencv_ort to opencv 4.8!
    Mat blob = blobFromImage(resizedImage, 1.0/255.0, Size(), Scalar(), true);

    netHandDet->setInput(blob);

    std::vector<Mat> outputBlob;
    netHandDet->forward(outputBlob, outputNameFromModel);

    CV_LOG_DEBUG(NULL, "Finish model inference!")

    // TODO setDevice! currently, we only support CPU!

    CV_Assert(outputBlob.size() == 2);

    // Tensor2Detection: according the anchors pre-set do the

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

        // project the hand box
        xmin = std::max((xmin - dw) * div_ratio, 0.0f);
        ymin = std::max((ymin - dh) * div_ratio, 0.0f);
        xmax = std::min((xmax - dw) * div_ratio, imgW);
        ymax = std::min((ymax - dh) * div_ratio, imgH);

        rect.x = xmin, rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        // TODO refine the bounding box expanding strategy.
        // step 4: expands and shifts the rectangle that contains the palm, so that it's likely to cover the entire hand.
        float w = xmax - xmin, h = ymax - ymin;

        float x_center = xmin + 0.5f * w;
        float y_center = ymin + 0.5f * h;

        float longSide = std::max(w, h) * expandRatio;
        rect.width = longSide;
        rect.height = longSide;

        // min max
        rect.x = std::max(std::min(x_center - rect.width * 0.5f, imgW), 0.0f);
        rect.y = std::max(std::min(y_center - rect.height * 0.5f + shift_y * longSide, imgH), 0.0f);

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