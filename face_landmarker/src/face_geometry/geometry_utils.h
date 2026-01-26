// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_geometry/libs/mesh_3d_utils.h

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


#ifndef MPP_CMAKE_GEOMETRY_UTILS_H
#define MPP_CMAKE_GEOMETRY_UTILS_H

#include "vector"
#include "iostream"

struct Mesh3d
{
    enum VertexType
    {
        VERTEX_PT = 0
    };

    enum PrimitiveType
    {
        TRIANGLE = 0
    };

    VertexType vertex_type;
    PrimitiveType primitive_type;

    uint32_t vertex_buffer_size;

    std::vector<float> vertex_buffer;
    std::vector<int> index_buffer;
};

struct GeometryPipelineMeta
{
    GeometryPipelineMeta();

    std::vector<double> procrustes_landmark_basis_weight = {
            0.070909939706326, 0.032100144773722, 0.008446550928056, 0.058724168688059, 0.007667080033571,
            0.009078059345484, 0.009791937656701, 0.014565368182957, 0.018591361120343, 0.005197994410992,
            0.120625205338001, 0.005560018587857, 0.05328618362546 , 0.066890455782413, 0.014816547743976,
            0.014262833632529,0.025462191551924, 0.047252278774977, 0.058724168688059, 0.007667080033571,
            0.009078059345484, 0.009791937656701, 0.014565368182957, 0.018591361120343, 0.005197994410992,
            0.120625205338001, 0.005560018587857, 0.05328618362546 , 0.066890455782413, 0.014816547743976,
            0.014262833632529, 0.025462191551924, 0.047252278774977
    };

    std::vector<int> procrustes_landmark_id = {
            4  , 6  , 10 , 33 , 54 , 67 , 117, 119, 121, 127, 129, 132, 133, 136, 143, 147, 198, 205, 263, 284, 297,
            346, 348, 350, 356, 358, 361, 362, 365, 372, 376, 420, 425,
    };

    Mesh3d canonical_mesh; // canonical mesh.
};

void InitializePerspectiveMatrix(std::vector<float>& matrix,
                                 float aspect_ratio, float fov_degrees, float z_near, float z_far);

enum class VertexComponent { POSITION, TEX_COORD };

std::size_t GetVertexSize(Mesh3d::VertexType vertex_type);

std::size_t GetPrimitiveSize(Mesh3d::PrimitiveType primitive_type);

bool HasVertexComponent(Mesh3d::VertexType vertex_type,
                        VertexComponent vertex_component);

uint32_t GetVertexComponentOffset(
        Mesh3d::VertexType vertex_type, VertexComponent vertex_component);

uint32_t GetVertexComponentSize(
        Mesh3d::VertexType vertex_type, VertexComponent vertex_component);

uint32_t GetVertexComponentOffsetVertexPT(VertexComponent vertex_component);

uint32_t GetVertexComponentSizeVertexPT(VertexComponent vertex_component);

bool HasVertexComponentVertexPT(VertexComponent vertex_component);

#endif //MPP_CMAKE_GEOMETRY_UTILS_H
