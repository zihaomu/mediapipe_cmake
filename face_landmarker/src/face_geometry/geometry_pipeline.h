// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc

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

#ifndef MPP_CMAKE_GEOMETRY_PIPELINE_H
#define MPP_CMAKE_GEOMETRY_PIPELINE_H

#include "geometry_utils.h"

struct CameraEnviroment
{
    float vertical_fov_degrees = 63.0f; // degrees
    float near = 1.0f;    // cm
    float far = 10000.0f; // 100 m
};

// FaceGeometry Output
struct FaceGeometry
{
    std::vector<float> metric_face_landmark;
    std::vector<float> pose_transform_mat;    // for saving Eigen::Matrix4f
};

class GeometryPipeline {
public:
    virtual ~GeometryPipeline() = default;

    virtual std::vector<float> ComputePerspectiveTransformMatrix(int image_width, int image_height) = 0;

    virtual std::vector<FaceGeometry> EstimateFaceGeometry(const std::vector<std::vector<float> >& multi_face_landmarks,
                                     int frame_width, int frame_height) const = 0;
};

// Create an instance of GeometryPipeline
// Both the enviroment and metadata should be valid.
// Canonical face mesh (defined as a part of `metadata`) must have the POSITION and TEX_COORD vertex components.
std::unique_ptr<GeometryPipeline> CreateGeometryPipeline(const CameraEnviroment& enviroment,
                                                         const GeometryPipelineMeta& metadata);

#endif //MPP_CMAKE_GEOMETRY_PIPELINE_H
