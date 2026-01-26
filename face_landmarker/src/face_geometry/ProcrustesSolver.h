// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_geometry/libs/procrustes_solver.h

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

#ifndef MPP_CMAKE_PROCRUSTESSOLVER_H
#define MPP_CMAKE_PROCRUSTESSOLVER_H

#include "Eigen/Dense"

// Encapsulates a stateless solver for the Weighted Extended Orthogonal
// Procrustes (WEOP) Problem, as defined in Section 2.4 of
// https://doi.org/10.3929/ethz-a-004656648.
//
// Given the source and the target point clouds, the algorithm estimates
// a 4x4 transformation matrix featuring the following semantic components:
//
//   * Uniform scale
//   * Rotation
//   * Translation
//
// The matrix maps the source point cloud into the target point cloud minimizing
// the Mean Squared Error.
class ProcrustesSolver {
public:
    virtual ~ProcrustesSolver() = default;

    /// All soruce points, target points and point weights should be defined the same number of points.
    /// A too small diameter of either of the point clouds will likely lead to numerical instabilities
    /// and failure to estimate the transformations.
    /// \param source_points
    /// \param target_points
    /// \param point_weights
    /// \param transform_mat
    /// \return
    virtual int SolveWeightedOrthogonalProblem(
            const Eigen::Matrix3Xf& source_points,
            const Eigen::Matrix3Xf& target_points,
            const Eigen::VectorXf& point_weights,
            Eigen::Matrix4f& transform_mat) = 0;
};

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver();

#endif //MPP_CMAKE_PROCRUSTESSOLVER_H
