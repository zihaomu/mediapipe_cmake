// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/util/filtering/one_euro_filter.h

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

#ifndef MPP_REPRODUCE_SMOOTHING_H
#define MPP_REPRODUCE_SMOOTHING_H

#include <cmath>
#include <vector>
#include "common.h"

namespace mpp
{
// TODO! Add the RelativeVelocityFilter
// Ref: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/relative_velocity_filter.cc

class LowPassSmoother
{
public:
    LowPassSmoother(float alpha_) : alpha(alpha_) {}
    ~LowPassSmoother() {}
    void reset();

    void apply(const std::vector<float>& input, std::vector<float> &output);

    // overload for 2D point.
    void apply(const PointList2f& input, PointList2f &output);

    // overload for 3D point.
    void apply(const PointList3f& input, PointList3f&output);

private:
    cv::Point2f applyImpl(const cv::Point2f& x_cur, const cv::Point2f& x_pre, const int loc);
    cv::Point3f applyImpl(const cv::Point3f& x_cur, const cv::Point3f& x_pre, const int loc);
    float applyImpl(const float x_cur, const float x_pre, int loc);
    float alpha = 1.0f;

    std::vector<float> x_prev;
    PointList2f x_prev_2f;
    PointList3f x_prev_3f;
};

// class for vector, Point2f and Point3f smoothing.
// Original paper link: https://inria.hal.science/hal-00670496/document
class OneEuroSmoother
{
public:
    /// Create smoothing instance
    /// \param _fc_min  // min cutoff frequncy
    /// \param _beta  // Speed ratio
    /// \param _fc_d  // constant cut-off frequency, in the original paper, it was set to 1.0.
    OneEuroSmoother(float _fc_min, float _beta, float _fc_d = 1.0f)
            : fc_min(_fc_min), beta(_beta), fc_d(_fc_d) {}

    ~OneEuroSmoother() noexcept;

    // clean the smoother.
    void reset();

    /// Perform the OneEuro Smoothing for giving landmark or vector.
    /// \param input input data
    /// \param timestamp the timestamp should be the microsecond.
    /// \param output output data
    void apply(const std::vector<float>& input, int64_t timestamp, std::vector<float> &output);

    // overload for 2D point.
    void apply(const std::vector<cv::Point2f>& input, int64_t timestamp, std::vector<cv::Point2f> &output);

    // overload for 3D point.
    void apply(const std::vector<cv::Point3f>& input, int64_t timestamp, std::vector<cv::Point3f> &output);

private:
    cv::Point2f applyImpl(const cv::Point2f& x_cur, const cv::Point2f& x_pre, const int loc);
    cv::Point3f applyImpl(const cv::Point3f& x_cur, const cv::Point3f& x_pre, const int loc);
    float applyImpl(const float x_cur, const float x_pre, int loc);
    float fc_min = 0.; // min cutoff effector
    float beta = 1.;   // speed effector
    float fc_d = 1.;   // derivative cutoff.

    float frequency_; // Sampling rate, computed by the inverse of two time stamp.
    int64_t last_timestamp = 0;
    float alpha = 0.;

    std::vector<float> x_prev_hat;
    std::vector<float> dx_prev_hat;

    PointList3f x_prev_hat_3f;
    PointList3f dx_prev_hat_3f;

    PointList2f x_prev_hat_2f;
    PointList2f dx_prev_hat_2f;
};

} // namespace

#endif //MPP_REPRODUCE_SMOOTHING_H
