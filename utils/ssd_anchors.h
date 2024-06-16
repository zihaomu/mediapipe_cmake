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

#ifndef MPP_REPRODUCE_SSD_ANCHORS_H
#define MPP_REPRODUCE_SSD_ANCHORS_H

#include <cmath>
#include <utility>
#include <vector>

#include "common.h"

namespace mpp
{

// The normalized anchor coordinates of a sticker
struct Anchor {
//    float x;  // [0.0-1.0]
//    float y;  // [0.0-1.0]
//    float z;  // Centered around 1.0 [current_scale = z * initial_scale]

    // box
    float w;
    float h;
    float x_center;
    float y_center;
    int sticker_id;
};

struct MultiScaleAnchorInfo {
    int level;
    std::vector<float> aspect_ratios;
    std::vector<float> scales;
    std::pair<float, float> base_anchor_size;
    std::pair<float, float> anchor_stride;
};

struct FeatureMapDim {
    int height;
    int width;
};

struct SSDAnchorOptions
{
    SSDAnchorOptions(int input_size_width, int input_size_height, float min_scale, float max_scale, int num_layers);
    int input_size_height;
    int input_size_width;
    float min_scale;
    float max_scale;
    int num_layers;

    float anchor_offset_x = 0.5f;
    float anchor_offset_y = 0.5f;

    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;
    std::vector<int> strides;
    std::vector<float> aspect_ratios;

    bool reduce_boxes_in_lowest_layer = false;
    float interpolated_scale_aspect_ratio = 1.0f;
    bool fixed_anchor_size = false;
    bool multiscale_anchor_generation = false;

    int min_level = 3;
    int max_level = 7;
    float anchor_scale = 4.0f;
    int scales_per_octave = 2;
    bool normalize_coordinates = true;
};

int generateMultiScaleAnchors(std::vector<Anchor>* anchors, const SSDAnchorOptions& options);
int generateAnchors(std::vector<Anchor>* anchors, const SSDAnchorOptions& options);

} // namespace mpp



#endif //MPP_REPRODUCE_SSD_ANCHORS_H
