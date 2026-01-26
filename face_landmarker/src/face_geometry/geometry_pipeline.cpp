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

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "geometry_pipeline.h"
#include "ProcrustesSolver.h"
#include "geometry_utils.h"

struct PerspectiveCameraFrustum {
    // NOTE: all arguments must be validated prior to calling this constructor.
    PerspectiveCameraFrustum(const CameraEnviroment& perspective_camera,
                             int frame_width, int frame_height) {
        static constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.f;

        const float height_at_near =
                2.f * perspective_camera.near *
                std::tan(0.5f * kDegreesToRadians *
                         perspective_camera.vertical_fov_degrees);

        const float width_at_near = frame_width * height_at_near / frame_height;

        left = -0.5f * width_at_near;
        right = 0.5f * width_at_near;
        bottom = -0.5f * height_at_near;
        top = 0.5f * height_at_near;
        near = perspective_camera.near;
        far = perspective_camera.far;
    }

    float left;
    float right;
    float bottom;
    float top;
    float near;
    float far;
};

enum OriginPointLocation
{
    BOTTOM_LEFT_CORNER = 1,
    TOP_LEFT_CORNER = 2
};

enum InputSource {
    DEFAULT = 0,  // FACE_LANDMARK_PIPELINE
    FACE_LANDMARK_PIPELINE = 1,
    FACE_DETECTION_PIPELINE = 2
};

static void ConvertLandmarkListToEigenMatrix(
        const std::vector<float>& landmark_list,
        Eigen::Matrix3Xf& eigen_matrix)
{
    int landmark_size = landmark_list.size()/3;
    eigen_matrix = Eigen::Matrix3Xf(3, landmark_size);

    for (int i = 0; i < landmark_size; ++i)
    {
        eigen_matrix(0, i) = landmark_list[i * 3 + 0];
        eigen_matrix(1, i) = landmark_list[i * 3 + 1];
        eigen_matrix(2, i) = landmark_list[i * 3 + 2];
    }
}

static void ConvertEigenMatrixToLandmarkList(
        const Eigen::Matrix3Xf& eigen_matrix,
        std::vector<float>& landmark_list)
{
    landmark_list.clear();
    size_t point_size = eigen_matrix.cols();
    landmark_list.resize(point_size * 3, 0.f);

    for (int i = 0; i < point_size; ++i)
    {
        landmark_list[i * 3 + 0] = eigen_matrix(0, i);
        landmark_list[i * 3 + 1] = eigen_matrix(1, i);
        landmark_list[i * 3 + 2] = eigen_matrix(2, i);
    }
}

static void ChangeHandedness(Eigen::Matrix3Xf& landmarks)
{
    landmarks.row(2) *= -1.f;
}

static void UnprojectXY(const PerspectiveCameraFrustum& pcf,
                        Eigen::Matrix3Xf& landmarks)
{
    landmarks.row(0) =
            landmarks.row(0).cwiseProduct(landmarks.row(2)) / pcf.near;
    landmarks.row(1) =
            landmarks.row(1).cwiseProduct(landmarks.row(2)) / pcf.near;
}

static void MoveAndRescaleZ(const PerspectiveCameraFrustum& pcf,
                            float depth_offset, float scale,
                            Eigen::Matrix3Xf& landmarks)
{
    landmarks.row(2) =
            (landmarks.array().row(2) - depth_offset + pcf.near) / scale;
}


// 屏幕空间到度量空间的转换
class ScreenToMetricSpaceConverter {
public:
    ScreenToMetricSpaceConverter(
            // TODO to figure out what the first two arguments are.
            OriginPointLocation origin_point_location,
            InputSource input_source,
            Eigen::Matrix3Xf&& canonical_metric_landmarks,
            Eigen::VectorXf&& landmark_weights,
            std::unique_ptr<ProcrustesSolver> procrustes_solver)
            : origin_point_location_(origin_point_location),
              input_source_(input_source),
              canonical_metric_landmarks_(std::move(canonical_metric_landmarks)),
              landmark_weights_(std::move(landmark_weights)),
              procrustes_solver_(std::move(procrustes_solver))
    {
    }


    // Converts `screen landmark list` into `metric landmark list` and estimates the `pose transform mat`
    // Here's the algorithm summary
    // step1: project X and Y screen landmark coordinate at Z near plane.
    //
    // step2: Estimate a canonical-to-runtime landmark set scale by running the Procrustes solver using the screen
    // runtime landmarks.
    // step3:
    int Convert(
            const std::vector<float>& screen_landmark_list,
            const PerspectiveCameraFrustum& pcf,
            std::vector<float>& metric_landmark_list,
            Eigen::Matrix4f& pose_transform_mat) const

    {
        if (screen_landmark_list.size() != canonical_metric_landmarks_.cols() * 3)
        {
            std::cerr<<"The number of landmarks doesn't match the number passed upon initialization!"<<std::endl;
            return -1;
        }

        Eigen::Matrix3Xf screen_landmarks;
        ConvertLandmarkListToEigenMatrix(screen_landmark_list, screen_landmarks);

        ProjectXY(pcf, screen_landmarks);
        const float depth_offset = screen_landmarks.row(2).mean();

        // step1: iteration: don't unproject XY because it's unsafe to do so due to the relative nature of the Z coordinate.
        // Instead, run the first estimate on the projected XY and use that scale to unproject for second iteration.
        Eigen::Matrix3Xf intermediate_landmarks(screen_landmarks);
        ChangeHandedness(intermediate_landmarks);

        const float first_iteration_scale = EstimateScale(intermediate_landmarks);
        if (first_iteration_scale == 0.f)
        {
            std::cerr<<"Failed to estimate first iteration scale!"<<std::endl;
            return -1;
        }

        // step2: iteration unproject XY using the scale from the 1st iteration.
        intermediate_landmarks = screen_landmarks;
        MoveAndRescaleZ(pcf, depth_offset, first_iteration_scale, intermediate_landmarks);
        UnprojectXY(pcf, intermediate_landmarks);
        ChangeHandedness(intermediate_landmarks);

        const float second_iteration_scale = EstimateScale(intermediate_landmarks);

        if (second_iteration_scale == 0.f)
        {
            std::cerr<<"Failed to estimate second iteration scale!"<<std::endl;
            return -1;
        }

        // Use the total scale to unproject the screen landmarks.
        const float total_scale = first_iteration_scale * second_iteration_scale;
        MoveAndRescaleZ(pcf, depth_offset, total_scale, screen_landmarks);
        UnprojectXY(pcf, screen_landmarks);
        ChangeHandedness(screen_landmarks);

        // At this point, screen landmarks are converted into metric landmarks
        Eigen::Matrix3Xf& metric_landmarks = screen_landmarks;
        int ret = procrustes_solver_->SolveWeightedOrthogonalProblem(
                canonical_metric_landmarks_, metric_landmarks, landmark_weights_,
                pose_transform_mat);

        if (ret < 0)
        {
            std::cerr<<"Failed to estimate pose transform matrix!"<<std::endl;
            return -1;
        }

        // TODO check
        metric_landmarks = (pose_transform_mat.inverse() *
                            metric_landmarks.colwise().homogeneous())
                .topRows(3);
//        auto dataInternal = metric_landmarks.colwise().homogeneous();
//        metric_landmarks = (pose_transform_mat.inverse()).topRows(3);
        ConvertEigenMatrixToLandmarkList(metric_landmarks, metric_landmark_list);

        return 0; // success!
    }

    void ProjectXY(const PerspectiveCameraFrustum& pcf,
                   Eigen::Matrix3Xf& landmarks) const {
        float x_scale = pcf.right - pcf.left;
        float y_scale = pcf.top - pcf.bottom;
        float x_translation = pcf.left;
        float y_translation = pcf.bottom;

        if (origin_point_location_ == OriginPointLocation::TOP_LEFT_CORNER) {
            landmarks.row(1) = 1.f - landmarks.row(1).array();
        }

        landmarks =
                landmarks.array().colwise() * Eigen::Array3f(x_scale, y_scale, x_scale);
        landmarks.colwise() += Eigen::Vector3f(x_translation, y_translation, 0.f);
    }

    float EstimateScale(Eigen::Matrix3Xf& landmarks) const
    {
        Eigen::Matrix4f transform_mat;
        int ret = procrustes_solver_->SolveWeightedOrthogonalProblem(
                canonical_metric_landmarks_, landmarks, landmark_weights_,
                transform_mat);

        if (ret < 0)
        {
            std::cerr<<"Failed to estimate canonical-to-runtime landmark set transform!"<<std::endl;
            return 0.f;
        }

        return transform_mat.col(0).norm();
    }


    const OriginPointLocation origin_point_location_;
    const InputSource input_source_;
    Eigen::Matrix3Xf canonical_metric_landmarks_;
    Eigen::VectorXf landmark_weights_;

    std::unique_ptr<ProcrustesSolver> procrustes_solver_;
};

static bool IsScreenLandmarkListTooCompact(
        const std::vector<float>& screen_landmarks) {
    float mean_x = 0.f;
    float mean_y = 0.f;
    int landmark_size = screen_landmarks.size() / 3;

    for (int i = 0; i < landmark_size; ++i)
    {
        mean_x += (screen_landmarks[i * 3] - mean_x) / static_cast<float>(i + 1);
        mean_y += (screen_landmarks[i * 3 + 1] - mean_y) / static_cast<float>(i + 1);
    }

    float max_sq_dist = 0.f;
    for (int i = 0; i < landmark_size; i++)
    {
        const float d_x = screen_landmarks[i * 3] - mean_x;
        const float d_y = screen_landmarks[i * 3 + 1] - mean_y;
        max_sq_dist = std::max(max_sq_dist, d_x * d_x + d_y * d_y);
    }

    static constexpr float kIsScreenLandmarkListTooCompactThreshold = 1e-3f;
    return std::sqrt(max_sq_dist) <= kIsScreenLandmarkListTooCompactThreshold;
}

class GeometryPipelineImpl : public GeometryPipeline
{
public:
    GeometryPipelineImpl(
            const CameraEnviroment& perspective_camera,
            const Mesh3d& canonical_mesh,
            uint32_t canonical_mesh_vertex_size,
            uint32_t canonical_mesh_num_vertices,
            uint32_t canonical_mesh_vertex_position_offset,
            std::unique_ptr<ScreenToMetricSpaceConverter> space_converter)
            : perspective_camera_(perspective_camera),
            canonical_mesh_(canonical_mesh),
            canonical_mesh_vertex_size_(canonical_mesh_vertex_size),
            canonical_mesh_num_vertices_(canonical_mesh_num_vertices),
            canonical_mesh_vertex_position_offset_(canonical_mesh_vertex_position_offset),
            space_converter_(std::move(space_converter))
    {
    }

    std::vector<float> ComputePerspectiveTransformMatrix(int image_width, int image_height) override
    {
        std::vector<float> matrix(16, 0.f);

        float ration = image_width * 1.0 / image_height;
        InitializePerspectiveMatrix(matrix, ration, perspective_camera_.vertical_fov_degrees,
                                    perspective_camera_.near, perspective_camera_.far);

        return matrix;
    }

    /// TODO check if we need to change the output of program to feed our need.
    /// \param multi_face_landmarks
    /// \param frame_width
    /// \param frame_height
    /// \return contain the metric face landmark, pose matrix, camera matrix!
    std::vector<FaceGeometry> EstimateFaceGeometry(
            const std::vector<std::vector<float> >& multi_face_landmarks,
            int frame_width, int frame_height) const override
    {
        if (frame_height <= 0 || frame_width <= 0)
        {
            std::cerr<<"Frame width and height must be positive!"<<std::endl;
            return {};
        }

        // Create a perspective camera frustum to be shared for geometry estimation
        // per each face.
        PerspectiveCameraFrustum pcf(perspective_camera_, frame_width, frame_height);

        std::vector<FaceGeometry> multi_face_geometry;

        // From this point, the meaning of face landmarks is clarified further as screen face landmarks
        // This is done by distinguish from metric face landmarks that are derived during the face geometry estimation process
        for (const std::vector<float>& screen_face_landmarks : multi_face_landmarks)
        {
            // Having a too compact screen landmark list will result in numerical instabilities,
            // therefore such faces are filtered.
            if (IsScreenLandmarkListTooCompact(screen_face_landmarks))
            {
                continue;
            }

            // Convert the screen landmarks into the metric landmarks and get the pose transformation matrix.
            std::vector<float> metric_face_landmarks = {};
            Eigen::Matrix4f pose_transform_mat;

            int ret = space_converter_->Convert(screen_face_landmarks, pcf, metric_face_landmarks, pose_transform_mat);

            if (ret < 0)
            {
                std::cerr<<"Failed to convert landmarks from the screen to the metric space!"<<std::endl;
            }

            // TODO: if this part is enough for our program!

            // Pack geometry data for this face.
            // TODO: fix the error relation about geometry with mutable mesh.
            FaceGeometry face_geometry;

            face_geometry.metric_face_landmark = metric_face_landmarks;
            face_geometry.pose_transform_mat = std::vector<float>(pose_transform_mat.data(), pose_transform_mat.data() + pose_transform_mat.size());

            multi_face_geometry.push_back(face_geometry);
        }

        return multi_face_geometry;
    }

private:
    const CameraEnviroment perspective_camera_;
    const Mesh3d canonical_mesh_;
    const uint32_t canonical_mesh_vertex_size_;
    const uint32_t canonical_mesh_num_vertices_;
    const uint32_t canonical_mesh_vertex_position_offset_;

    std::unique_ptr<ScreenToMetricSpaceConverter> space_converter_;
};

std::unique_ptr<GeometryPipeline> CreateGeometryPipeline(const CameraEnviroment& enviroment,
                                                         const GeometryPipelineMeta& metadata)
{
    // Validate Environment
    static constexpr float kAbsoluteErrorEps = 1e-9f;

    if (enviroment.near < kAbsoluteErrorEps)
    {
        std::cerr<<"Near Z must be greater than 0 with a margin of 10^{-9}!"<<std::endl;
        return {};
    }

    if (enviroment.far < kAbsoluteErrorEps + enviroment.near)
    {
        std::cerr<<"Far Z must be greater than Near Z with a margin of 10^{-9}!"<<std::endl;
        return {};
    }

    if (enviroment.vertical_fov_degrees < kAbsoluteErrorEps)
    {
        std::cerr<<"Vertical FOV must be positive with a margin of 10^{-9}!"<<std::endl;
        return {};
    }

    if (enviroment.vertical_fov_degrees + kAbsoluteErrorEps > 180.f)
    {
        std::cerr<<"Vertical FOV must be less than 180 degrees with a margin of 10^{-9}!"<<std::endl;
        return {};
    }

    const auto& canonical_mesh = metadata.canonical_mesh;
    // Validate Geometry Pipeline Metadata
    uint32_t canonical_mesh_vertex_size =
            GetVertexSize(canonical_mesh.vertex_type);
    uint32_t canonical_mesh_num_vertices =
            canonical_mesh.vertex_buffer_size / canonical_mesh_vertex_size;
    uint32_t canonical_mesh_vertex_position_offset =
            GetVertexComponentOffset(canonical_mesh.vertex_type,VertexComponent::POSITION);

    // Put the Procrustes landmark basis into Eigen matrices for an easier access.
    Eigen::Matrix3Xf canonical_metric_landmarks =
            Eigen::Matrix3Xf::Zero(3, canonical_mesh_num_vertices);
    Eigen::VectorXf landmark_weights =
            Eigen::VectorXf::Zero(canonical_mesh_num_vertices);

    for (int i = 0; i < canonical_mesh_num_vertices; ++i)
    {
        uint32_t vertex_buffer_offset =
                canonical_mesh_vertex_size * i + canonical_mesh_vertex_position_offset;

        canonical_metric_landmarks(0, i) =
                canonical_mesh.vertex_buffer[vertex_buffer_offset];
        canonical_metric_landmarks(1, i) =
                canonical_mesh.vertex_buffer[vertex_buffer_offset + 1];
        canonical_metric_landmarks(2, i) =
                canonical_mesh.vertex_buffer[vertex_buffer_offset + 2];
    }

    int procrustes_landmark_size = metadata.procrustes_landmark_id.size();

    for (int i = 0; i < procrustes_landmark_size; i++)
    {
        landmark_weights(metadata.procrustes_landmark_id[i]) = metadata.procrustes_landmark_basis_weight[i];
    }

    // create unique pointer to
    std::unique_ptr<ScreenToMetricSpaceConverter> screen_metric_ptr = std::make_unique<ScreenToMetricSpaceConverter>(
            OriginPointLocation::TOP_LEFT_CORNER,
            InputSource::FACE_LANDMARK_PIPELINE,
            std::move(canonical_metric_landmarks),
            std::move(landmark_weights),
            CreateFloatPrecisionProcrustesSolver());

    std::unique_ptr<GeometryPipeline> result =
            std::make_unique<GeometryPipelineImpl>(
                    enviroment, canonical_mesh,
                    canonical_mesh_vertex_size, canonical_mesh_num_vertices,
                    canonical_mesh_vertex_position_offset, std::move(screen_metric_ptr));

    return result;
}