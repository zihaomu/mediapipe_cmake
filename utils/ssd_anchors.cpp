// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/calculators/tflite/ssd_anchors_calculator.cc

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

#include "ssd_anchors.h"

namespace mpp
{

int GetNumLayers(const SSDAnchorOptions& options)
{
    if (options.multiscale_anchor_generation)
    {
        return (options.max_level - options.min_level + 1);
    }
    return options.num_layers;
}

FeatureMapDim GetFeatureMapDimensions(
        const SSDAnchorOptions& options, int index)
{
    FeatureMapDim feature_map_dims;
    if (options.feature_map_height.size())
    {
        feature_map_dims.height = options.feature_map_height[index];
        feature_map_dims.width = options.feature_map_width[index];
    }
    else
    {
        const int stride = options.strides[index];
        feature_map_dims.height =
                std::ceil(1.0f * options.input_size_height / stride);
        feature_map_dims.width =
                std::ceil(1.0f * options.input_size_width / stride);
    }
    return feature_map_dims;
}

float CalculateScale(float min_scale, float max_scale, int stride_index,
                     int num_strides)
{
    if (num_strides == 1)
    {
        return (min_scale + max_scale) * 0.5f;
    }
    else
    {
        return min_scale +
               (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
    }
}

// TBD.
void NormalizeAnchor(const int input_height, const int input_width,
                     Anchor* anchor) {
    anchor->h = (anchor->h / (float)input_height);
    anchor->w = (anchor->w / (float)input_width);
    anchor->y_center = (anchor->y_center / (float)input_height);
    anchor->x_center = (anchor->x_center / (float)input_width);
}

Anchor CalculateAnchorBox(const int y_center, const int x_center,
                          const float scale, const float aspect_ratio,
                          const std::pair<float, float> base_anchor_size,
        // y-height first
                          const std::pair<float, float> anchor_stride,
                          const std::pair<float, float> anchor_offset)
{
    Anchor result;
    float ratio_sqrt = std::sqrt(aspect_ratio);
    result.h = (scale * base_anchor_size.first / ratio_sqrt);
    result.w = (scale * ratio_sqrt * base_anchor_size.second);
    result.y_center = (y_center * anchor_stride.first + anchor_offset.first);
    result.x_center = (x_center * anchor_stride.second + anchor_offset.second);
    return result;
}

// Although we have stride for both x and y, only one value is used for offset
// calculation. See
// tensorflow_models/object_detection/anchor_generators/multiscale_grid_anchor_generator.py;l=121
std::pair<float, float> GetMultiScaleAnchorOffset(
        const SSDAnchorOptions& options, const float stride,
        const int level)
{
    std::pair<float, float> result(0., 0.);
    int denominator = std::pow(2, level);

    if (options.input_size_height % denominator == 0 ||
        options.input_size_height == 1)
    {
        result.first = stride / 2.0;
    }

    if (options.input_size_width % denominator == 0 ||
        options.input_size_width == 1)
    {
        result.second = stride / 2.0;
    }
    return result;
}

SSDAnchorOptions::SSDAnchorOptions(int _input_size_height, int _input_size_width, float _min_scale, float _max_scale, int _num_layers)
: input_size_height(_input_size_height), input_size_width(_input_size_width), min_scale(_min_scale), max_scale(_max_scale), num_layers(_num_layers)
{}

int generateMultiScaleAnchors(std::vector<Anchor>* anchors, const SSDAnchorOptions& options)
{
    std::vector<MultiScaleAnchorInfo> anchor_infos;
    for (int i = options.min_level; i <= options.max_level; ++i)
    {
        MultiScaleAnchorInfo current_anchor_info;
        // level
        current_anchor_info.level = i;
        // aspect_ratios
        for (const float aspect_ratio : options.aspect_ratios)
        {
            current_anchor_info.aspect_ratios.push_back(aspect_ratio);
        }

        // scale
        for (int i = 0; i < options.scales_per_octave; ++i)
        {
            current_anchor_info.scales.push_back(
                    std::pow(2.0, (double)i / (double)options.scales_per_octave));
        }

        // anchor stride
        float anchor_stride = std::pow(2.0, i);
        current_anchor_info.anchor_stride =
                std::make_pair(anchor_stride, anchor_stride);

        // base_anchor_size
        current_anchor_info.base_anchor_size =
                std::make_pair(anchor_stride * options.anchor_scale,
                               anchor_stride * options.anchor_scale);
        anchor_infos.push_back(current_anchor_info);
    }

    for (unsigned int i = 0; i < anchor_infos.size(); ++i)
    {
        FeatureMapDim dimensions = GetFeatureMapDimensions(options, i);
        for (int y = 0; y < dimensions.height; ++y)
        {
            for (int x = 0; x < dimensions.width; ++x)
            {
                // loop over combination of scale and aspect ratio
                for (unsigned int j = 0; j < anchor_infos[i].aspect_ratios.size(); ++j)
                {
                    for (unsigned int k = 0; k < anchor_infos[i].scales.size(); ++k)
                    {
                        Anchor anchor = CalculateAnchorBox(
                                /*y_center=*/y, /*x_center=*/x, anchor_infos[i].scales[k],
                                             anchor_infos[i].aspect_ratios[j],
                                             anchor_infos[i].base_anchor_size,
                                /*anchor_stride=*/anchor_infos[i].anchor_stride,
                                /*anchor_offset=*/
                                             GetMultiScaleAnchorOffset(options,
                                                                       anchor_infos[i].anchor_stride.first,
                                                                       anchor_infos[i].level));
                        if (options.normalize_coordinates)
                        {
                            NormalizeAnchor(options.input_size_height,
                                            options.input_size_width, &anchor);
                        }
                        anchors->push_back(anchor);
                    }
                }
            }
        }
    }
    return 1;
}

int generateAnchors(std::vector<Anchor>* anchors, const SSDAnchorOptions& options)
{
    // Verify the options.
    if (!options.feature_map_height.size() && !options.strides.size())
    {
        CV_Error(cv::Error::StsUnmatchedSizes, "Both feature map shape and strides are missing. Must provide either one.");
        return -1;
    }
    const int kNumLayers = GetNumLayers(options);

    if (options.feature_map_height.size())
    {
        if (options.strides.size())
        {
            CV_Error(cv::Error::StsUnmatchedSizes, "Found feature map shapes. Strides will be ignored.");
        }
        CV_Assert(options.feature_map_height.size() == kNumLayers);
        CV_Assert(options.feature_map_height.size() == options.feature_map_width.size());
    }
    else
    {
        CV_Assert(options.strides.size() == kNumLayers);
    }

    if (options.multiscale_anchor_generation)
    {
        return generateMultiScaleAnchors(anchors, options);
    }

    int layer_id = 0;
    while (layer_id < options.num_layers)
    {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // For same strides, we merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < options.strides.size() &&
               options.strides[last_same_stride_layer] ==
               options.strides[layer_id])
        {
            const float scale =
                    CalculateScale(options.min_scale, options.max_scale,
                                   last_same_stride_layer, options.strides.size());

            if (last_same_stride_layer == 0 &&
                options.reduce_boxes_in_lowest_layer)
            {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push_back(1.0);
                aspect_ratios.push_back(2.0);
                aspect_ratios.push_back(0.5);
                scales.push_back(0.1);
                scales.push_back(scale);
                scales.push_back(scale);
            }
            else
            {
                for (int aspect_ratio_id = 0;
                     aspect_ratio_id < options.aspect_ratios.size();
                     ++aspect_ratio_id)
                {
                    aspect_ratios.push_back(options.aspect_ratios[aspect_ratio_id]);
                    scales.push_back(scale);
                }

                if (options.interpolated_scale_aspect_ratio > 0.0)
                {
                    const float scale_next =
                            last_same_stride_layer == options.strides.size() - 1
                            ? 1.0f
                            : CalculateScale(options.min_scale, options.max_scale,
                                             last_same_stride_layer + 1,
                                             options.strides.size());
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer++;
        }

        for (int i = 0; i < aspect_ratios.size(); ++i)
        {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = 0;
        int feature_map_width = 0;
        if (options.feature_map_height.size())
        {
            feature_map_height = options.feature_map_height[layer_id];
            feature_map_width = options.feature_map_width[layer_id];
        }
        else
        {
            const int stride = options.strides[layer_id];
            feature_map_height =
                    std::ceil(1.0f * options.input_size_height / stride);
            feature_map_width = std::ceil(1.0f * options.input_size_width / stride);
        }

        for (int y = 0; y < feature_map_height; ++y)
        {
            for (int x = 0; x < feature_map_width; ++x)
            {
                for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id)
                {
                    // TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    const float x_center =
                            (x + options.anchor_offset_x) * 1.0f / feature_map_width;
                    const float y_center =
                            (y + options.anchor_offset_y) * 1.0f / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x_center = (x_center);
                    new_anchor.y_center = (y_center);

                    if (options.fixed_anchor_size)
                    {
                        new_anchor.w = (1.0f);
                        new_anchor.h = (1.0f);
                    }
                    else
                    {
                        new_anchor.w = (anchor_width[anchor_id]);
                        new_anchor.h = (anchor_height[anchor_id]);
                    }
                    anchors->push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    return 1;
}



} // namespace mpp