// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_geometry/libs/procrustes_solver.cc

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

#include "ProcrustesSolver.h"
#include "iostream"

#include <cmath>
#include <memory>

#include <cstdint>

#include <vector>
#include <utility>

#include "Eigen/Dense"

class FloatPrecisionProcrustesSolver : public ProcrustesSolver
{
public:
    FloatPrecisionProcrustesSolver() = default;

    int SolveWeightedOrthogonalProblem(
            const Eigen::Matrix3Xf& source_points,
            const Eigen::Matrix3Xf& target_points,
            const Eigen::VectorXf& point_weights,
            Eigen::Matrix4f& transform_mat) override
    {
        // Check if source points has the same shape as target points.
        if (!(source_points.cols() > 0 && source_points.cols() == target_points.cols()))
        {
            std::cerr<<"Failed to validate weighted orthogonal problem input points!"<<std::endl;
            return -1;
        }

        Eigen::VectorXf sqrt_weights = ExtractSquareRoot(point_weights);
        int ret = InternalSolveWeightedOrthogonalProblem(
                source_points, target_points, sqrt_weights, transform_mat);

        if (ret < 0)
        {
            std::cerr<<"Failed to solve the WEOP problem!"<<std::endl;
            return -1;
        }

        return 0;
    }

private:
    static constexpr float kAbsoluteErrorEps = 1e-9f;

    static Eigen::VectorXf ExtractSquareRoot(
            const Eigen::VectorXf& point_weights) {
        Eigen::VectorXf sqrt_weights(point_weights);
        for (int i = 0; i < sqrt_weights.size(); ++i) {
            sqrt_weights(i) = std::sqrt(sqrt_weights(i));
        }

        return sqrt_weights;
    }

    // Combines a 3x3 rotation-and-scale matrix and a 3x1 translation vector into
    // a single 4x4 transformation matrix.
    static Eigen::Matrix4f CombineTransformMatrix(const Eigen::Matrix3f& r_and_s,
                                                  const Eigen::Vector3f& t)
    {
        Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
        result.leftCols(3).topRows(3) = r_and_s;
        result.col(3).topRows(3) = t;

        return result;
    }

    static int InternalSolveWeightedOrthogonalProblem(
            const Eigen::Matrix3Xf& sources, const Eigen::Matrix3Xf& targets,
            const Eigen::VectorXf& sqrt_weights, Eigen::Matrix4f& transform_mat)
    {
        // tranposed(A_w).
        Eigen::Matrix3Xf weighted_sources =
                sources.array().rowwise() * sqrt_weights.array().transpose();
        // tranposed(B_w).
        Eigen::Matrix3Xf weighted_targets =
                targets.array().rowwise() * sqrt_weights.array().transpose();

        // w = tranposed(j_w) j_w.
        float total_weight = sqrt_weights.cwiseProduct(sqrt_weights).sum();

        // Let C = (j_w tranposed(j_w)) / (tranposed(j_w) j_w).
        // Note that C = tranposed(C), hence (I - C) = tranposed(I - C).
        //
        // tranposed(A_w) C = tranposed(A_w) j_w tranposed(j_w) / w =
        // (tranposed(A_w) j_w) tranposed(j_w) / w = c_w tranposed(j_w),
        //
        // where c_w = tranposed(A_w) j_w / w is a k x 1 vector calculated here:
        Eigen::Matrix3Xf twice_weighted_sources =
                weighted_sources.array().rowwise() * sqrt_weights.array().transpose();
        Eigen::Vector3f source_center_of_mass =
                twice_weighted_sources.rowwise().sum() / total_weight;
        // tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
        // tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
        Eigen::Matrix3Xf centered_weighted_sources =
                weighted_sources - source_center_of_mass * sqrt_weights.transpose();

        Eigen::Matrix3f rotation;

        int ret = ComputeOptimalRotation(
                weighted_targets * centered_weighted_sources.transpose(), rotation);

        if (ret < 0)
        {
            std::cerr<<"Failed to compute the optimal rotation!"<<std::endl;
            return -1;
        }

        float scale = ComputeOptimalScale(centered_weighted_sources, weighted_sources,
                                    weighted_targets, rotation);

        // R = c tranposed(T).
        Eigen::Matrix3f rotation_and_scale = scale * rotation;

        // Compute optimal translation for the weighted problem.

        // tranposed(B_w - c A_w T) = tranposed(B_w) - R tranposed(A_w) in (54).
        const auto pointwise_diffs =
                weighted_targets - rotation_and_scale * weighted_sources;
        // Multiplication by j_w is a respectively weighted column sum.
        // (54) from the paper.
        const auto weighted_pointwise_diffs =
                pointwise_diffs.array().rowwise() * sqrt_weights.array().transpose();
        Eigen::Vector3f translation =
                weighted_pointwise_diffs.rowwise().sum() / total_weight;

        transform_mat = CombineTransformMatrix(rotation_and_scale, translation);

        return 0;
    }

    static int ComputeOptimalRotation(const Eigen::Matrix3f& design_matrix, Eigen::Matrix3f& rotation)
    {
        if (design_matrix.norm() < kAbsoluteErrorEps)
        {
            std::cerr<< "Design matrix norm is too small!"<<std::endl;
            return -1;
        }

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(
                design_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f postrotation = svd.matrixU();
        Eigen::Matrix3f prerotation = svd.matrixV().transpose();

        // Disallow reflection by ensuring that det(`rotation`) = +1 (and not -1),
        // see "4.6 Constrained orthogonal Procrustes problems"
        // in the Gower & Dijksterhuis's book "Procrustes Analysis".
        // We flip the sign of the least singular value along with a column in W.
        //
        // Note that now the sum of singular values doesn't work for scale
        // estimation due to this sign flip.
        if (postrotation.determinant() * prerotation.determinant() <
            static_cast<float>(0))
        {
            postrotation.col(2) *= static_cast<float>(-1);
        }

        // Transposed (52) from the paper.
        rotation = postrotation * prerotation;
        return 0;
    }

    static float ComputeOptimalScale(
            const Eigen::Matrix3Xf& centered_weighted_sources,
            const Eigen::Matrix3Xf& weighted_sources,
            const Eigen::Matrix3Xf& weighted_targets,
            const Eigen::Matrix3f& rotation)
    {
        // tranposed(T) tranposed(A_w) (I - C).
        const auto rotated_centered_weighted_sources =
                rotation * centered_weighted_sources;
        // Use the identity trace(A B) = sum(A * B^T)
        // to avoid building large intermediate matrices (* is Hadamard product).
        // (53) from the paper.
        float numerator =
                rotated_centered_weighted_sources.cwiseProduct(weighted_targets).sum();
        float denominator =
                centered_weighted_sources.cwiseProduct(weighted_sources).sum();

        if (denominator < kAbsoluteErrorEps)
            std::cerr<<"Scale expression denominator is too small!"<<std::endl;
        float scale = numerator / denominator;
        if (scale < kAbsoluteErrorEps)
            std::cerr<<"Scale is too small!"<<std::endl;

        return scale;
    }
};

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver()
{
    return std::make_unique<FloatPrecisionProcrustesSolver>();
};