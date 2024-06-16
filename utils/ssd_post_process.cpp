// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/calculators/tensor/tensors_to_detections_calculator.cc

// The following is original lincense:
// Copyright 2019 The MediaPipe Authors.
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

#include "ssd_post_process.h"

namespace mpp
{
SSDDecoderOption::SSDDecoderOption(int _num_classes, int _num_boxes, int _num_coords)
: num_classes(_num_classes), num_boxes(_num_boxes), num_coords(_num_coords)
{}

SSDDecoder::SSDDecoder(const SSDAnchorOptions &anchorOptions, const SSDDecoderOption& _detectorOption)
: detectorOption(_detectorOption)
{
    anchorList = {};
    generateAnchors(&anchorList, anchorOptions);

    num_classes = detectorOption.num_classes;
    num_boxes = detectorOption.num_boxes;
    num_coords = detectorOption.num_coords;
    box_format = detectorOption.box_format;
}

SSDDecoder::~SSDDecoder()
{

}

inline void funYXHW(float& y_center, float& x_center, float& h, float& w, const float* raw_boxes, int box_offset)
{
    y_center = raw_boxes[box_offset];
    x_center = raw_boxes[box_offset + 1];
    h = raw_boxes[box_offset + 2];
    w = raw_boxes[box_offset + 3];
}

inline void funXYWH(float& y_center, float& x_center, float& h, float& w, const float* raw_boxes, int box_offset)
{
    x_center = raw_boxes[box_offset];
    y_center = raw_boxes[box_offset + 1];
    w = raw_boxes[box_offset + 2];
    h = raw_boxes[box_offset + 3];
}

inline void funXYXY(float& y_center, float& x_center, float& h, float& w, const float* raw_boxes, int box_offset)
{
    x_center = (-raw_boxes[box_offset] + raw_boxes[box_offset + 2]) / 2;
    y_center = (-raw_boxes[box_offset + 1] + raw_boxes[box_offset + 3]) / 2;
    w = raw_boxes[box_offset + 2] + raw_boxes[box_offset];
    h = raw_boxes[box_offset + 3] + raw_boxes[box_offset + 1];
}

void SSDDecoder::run(const cv::Mat& boxTensor, const cv::Mat& scoreTensor, std::vector<BoxKp2>& outBox)
{
    CV_Assert(num_boxes != 0 && "Please set num_boxes in calculator options");

    outBox.clear(); // clean data.

    // TODO add shape check, the boxTensor should be 3 dimension.
    const float* rawBox = boxTensor.ptr<float>();
    const float* rawScore = scoreTensor.ptr<float>();

    // Anchor Decoder process
    void (*fun)(float& x_center, float& y_center, float& h, float& w, const float* raw_boxes, int box_offset);

    switch (box_format)
    {
        case SSDDecoderOption::UNSPECIFIED:
        case SSDDecoderOption::YXHW:
        {
            fun = funYXHW;
            break;
        }
        case SSDDecoderOption::XYWH:
        {
            fun = funXYWH;
            break;
        }
        case SSDDecoderOption::XYXY:
        {
            fun = funXYXY;
            break;
        }
        default:
            CV_Error(cv::Error::StsNotImplemented, "Unsupported box format in SSDDecoder::run!");
    }

    // Begin decode Boxes.
    for (int i = 0; i < num_boxes; i++)
    {
        BoxKp2 currentBox = {};

        // Filter class by scores
        int class_id = -1;
        float max_score = -std::numeric_limits<float>::max();

        if (detectorOption.max_results > 0 && outBox.size() == detectorOption.max_results)
        {
            break;
        }

        // Find the top score for box i.
        for (int score_idx = 0; score_idx < num_classes; ++score_idx)
        {
            // TODO check which part use IsClassIndexAllowed function?
//            if (IsClassIndexAllowed(score_idx))
//            {
            auto score = rawScore[i * num_classes + score_idx];

            if (detectorOption.sigmoid_score)
            {
                if (detectorOption.score_clipping_thresh != 0.0f)
                {
                    score = score < -detectorOption.score_clipping_thresh
                            ? -detectorOption.score_clipping_thresh
                            : score;
                    score = score > detectorOption.score_clipping_thresh
                            ? detectorOption.score_clipping_thresh
                            : score;
                }
                score = 1.0f / (1.0f + std::exp(-score));
            }

            if (max_score < score)
            {
                max_score = score;
                class_id = score_idx;
            }
//            }
        }

        if (detectorOption.min_score_thresh != 0.0f && max_score < detectorOption.min_score_thresh)
        {
            continue;
        }

//        if (!IsClassIndexAllowed(detection_classes[i])) {
//            continue;
//        }

        currentBox.classId = class_id;
        currentBox.score = max_score;

        const int box_offset = i * num_coords + detectorOption.box_coord_offset;
        float y_center = 0.0;
        float x_center = 0.0;
        float w = 0.0;
        float h = 0.0;

        fun(y_center, x_center, h, w, rawBox, box_offset);
        x_center =
                x_center / detectorOption.x_scale * anchorList[i].w + anchorList[i].x_center;
        y_center =
                y_center / detectorOption.y_scale * anchorList[i].h + anchorList[i].y_center;

        if (detectorOption.apply_exponential_on_box_size)
        {
            h = std::exp(h / detectorOption.h_scale) * anchorList[i].h;
            w = std::exp(w / detectorOption.w_scale) * anchorList[i].w;
        }
        else
        {
            h = h / detectorOption.h_scale * anchorList[i].h;
            w = w / detectorOption.w_scale * anchorList[i].w;
        }

        const float ymin = y_center - h / 2.f;
        const float xmin = x_center - w / 2.f;
        const float height = y_center + h / 2.f - ymin;
        const float width = x_center + w / 2.f - xmin;

        if (width < 0.0f || height < 0.0f)
            continue;

        currentBox.rect = {xmin, ymin, width, height};

        // Add keypoint if it has.
        if (detectorOption.num_keypoints)
        {
            currentBox.points.resize(detectorOption.num_keypoints);
            for (int k = 0; k < detectorOption.num_keypoints; ++k)
            {
                const int offset = i * num_coords + detectorOption.keypoint_coord_offset +
                                   k * detectorOption.num_values_per_keypoint;

                float keypoint_y = 0.0;
                float keypoint_x = 0.0;
                switch (box_format)
                {
                    case SSDDecoderOption::UNSPECIFIED:
                    case SSDDecoderOption::YXHW:
                    {
                        keypoint_y = rawBox[offset];
                        keypoint_x = rawBox[offset + 1];
                        break;
                    }
                    case SSDDecoderOption::XYWH:
                    case SSDDecoderOption::XYXY:
                    {
                        keypoint_x = rawBox[offset];
                        keypoint_y = rawBox[offset + 1];
                        break;
                    }
                }

                cv::Point2f point2F = {keypoint_x / detectorOption.x_scale * anchorList[i].w +
                                               anchorList[i].x_center,
                                       keypoint_y / detectorOption.y_scale * anchorList[i].h +
                                               anchorList[i].y_center};
                currentBox.points[k] = point2F;
            }
        }

        outBox.emplace_back(currentBox);
    }
}

} // namespace mpp
