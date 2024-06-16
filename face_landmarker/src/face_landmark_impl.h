// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/face_landmarker/tensors_to_face_landmarks_graph.cc

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

#ifndef MPP_CMAKE_FACE_LANDMARK_IMPL_H
#define MPP_CMAKE_FACE_LANDMARK_IMPL_H

#include "common.h"
#include "opencv2/dnn.hpp"

namespace mpp
{
struct FaceLandmarkIndex
{
    const std::vector<int> lips_idxs_xy = {
        // Lower outer.
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        // Upper outer (excluding corners).
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        // Lower inner.
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        // Upper inner (excluding corners).
        191, 80, 81, 82, 13, 312, 311, 310, 415,
        // Lower semi-outer.
        76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
        // Upper semi-outer (excluding corners).
        184, 74, 73, 72, 11, 302, 303, 304, 408,
        // Lower semi-inner.
        62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
        // Upper semi-inner (excluding corners).
        183, 42, 41, 38, 12, 268, 271, 272, 407
        };

    const std::vector<int> left_eye_idxs_xy = {
        //# Lower contour.
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        // # upper contour (excluding corners).
        246, 161, 160, 159, 158, 157, 173,
        //# Halo x2 lower contour.
        130, 25, 110, 24, 23, 22, 26, 112, 243,
        //# Halo x2 upper contour (excluding corners).
        247, 30, 29, 27, 28, 56, 190,
        //# Halo x3 lower contour.
        226, 31, 228, 229, 230, 231, 232, 233, 244,
        //# Halo x3 upper contour (excluding corners).
        113, 225, 224, 223, 222, 221, 189,
        35, 124, 46, 53, 52, 65,
        //# Halo x5 lower contour.
        143, 111, 117, 118, 119, 120, 121, 128, 245,
        //# Halo x5 upper contour (excluding corners) or eyebrow outer contour.
        156, 70, 63, 105, 66, 107, 55, 193
        };

    const std::vector<int> right_eye_idxs_xy = {
        // Lower contour.
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        // Upper contour (excluding corners).
        466, 388, 387, 386, 385, 384, 398,
        // Halo x2 lower contour.
        359, 255, 339, 254, 253, 252, 256, 341, 463,
        // Halo x2 upper contour (excluding corners).
        467, 260, 259, 257, 258, 286, 414,
        // Halo x3 lower contour.
        446, 261, 448, 449, 450, 451, 452, 453, 464,
        // Halo x3 upper contour (excluding corners).
        342, 445, 444, 443, 442, 441, 413,
        // Halo x4 upper contour (no lower because of mesh structure) or
        // eyebrow inner contour.
        265, 353, 276, 283, 282, 295,
        // Halo x5 lower contour.
        372, 340, 346, 347, 348, 349, 350, 357, 465,
        // Halo x5 upper contour (excluding corners) or eyebrow outer contour.
        383, 300, 293, 334, 296, 336, 285, 417
        };

    const std::vector<int> left_iris_xy = {
        // Center.
        468,
        // Iris right edge.
        469,
        // Iris top edge.
        470,
        // Iris left edge.
        471,
        // Iris bottom edge.
        472
        };

    const std::vector<int> left_iris_z = {
        // Lower contour.
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        // Upper contour (excluding corners).
        246, 161, 160, 159, 158, 157, 173
        };

    const std::vector<int> right_iris_xy = {
        // Center.
        473,
        // Iris right edge.
        474,
        // Iris top edge.
        475,
        // Iris left edge.
        476,
        // Iris bottom edge.
        477
        };
    const std::vector<int> right_iris_z = {
        // Lower contour.
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        // Upper contour (excluding corners).
        466, 388, 387, 386, 385, 384, 398
        };
};

class FaceLandmarker_Impl
{
public:
    FaceLandmarker_Impl(std::string modelPath, int device = 0);
    FaceLandmarker_Impl(const char* buffer, long buffer_size, int device = 0);
    ~FaceLandmarker_Impl();

    void run(const cv::Mat& img, PointList3f& landmark, float& landmark_score);

    void getInputWH(int& W, int& H);
    int getLandmarkSize();

private:
    void init();

    int landmarkSize = 468;
    FaceLandmarkIndex faceIndex;
    cv::Scalar mean;
    cv::Scalar scalefactor;
    int inputHeight, inputWidth;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;

    // 1x1x1x1404, 1x1x1x1
    std::vector<std::string> outputNameFromModel_468 = {"conv2d_21", "conv2d_31"};

    cv::Ptr<cv::dnn::Net> netFaceDet = nullptr;
};

} // namespace mpp
#endif //MPP_CMAKE_FACE_LANDMARK_IMPL_H
