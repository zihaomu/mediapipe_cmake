// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/util/filtering/one_euro_filter.cc

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

#include "smoothing.h"

using namespace cv;
namespace mpp
{

inline float getAlpha(float rate, float cutoff)
{
    float tau = 1.0/ (2* M_PI * cutoff);
    float te = 1.0 / rate;
    return 1.0/(1.0 + tau / te);
}

// return Hertz
inline float compute_frequency(const int64_t t0, const int64_t t1)
{
    static constexpr double kNanoSecondsToSecond = 1e-6;
    return float(1.0 / ((t1 - t0) * kNanoSecondsToSecond));
}

inline float exponential_smoothing(float a, float x, float x_prev)
{
    return a * x + (1 - a) * x_prev;
}

// >>>>>>>>>>>>>>>>>>>>> LowPassSmoother <<<<<<<<<<<<<<<<<<<<<<<<<
void LowPassSmoother::apply(const std::vector<float>& input, std::vector<float> &output)
{
    if (x_prev.empty())
    {
        output = input;
        x_prev = input;
    }
    else
    {
        int pointNum = input.size();
        output.resize(pointNum, 0.0f);

        for (int i = 0; i < pointNum; ++i)
        {
            output[i] = applyImpl(input[i], x_prev[i], i);
        }
    }
}

void LowPassSmoother::apply(const mpp::PointList2f &input, mpp::PointList2f &output)
{
    if (x_prev.empty())
    {
        output = input;
        x_prev_2f = input;
    }
    else
    {
        int pointNum = input.size();
        output.resize(pointNum, Point2f {0.0f, 0.0f});

        for (int i = 0; i < pointNum; ++i)
        {
            output[i] = applyImpl(input[i], x_prev_2f[i], i);
        }
    }
}

void LowPassSmoother::apply(const mpp::PointList3f &input, mpp::PointList3f &output)
{
    if (x_prev.empty())
    {
        output = input;
        x_prev_3f = input;
    }
    else
    {
        int pointNum = input.size();
        output.resize(pointNum, Point3f {0.0f, 0.0f, 0.0f});

        for (int i = 0; i < pointNum; ++i)
        {
            output[i] = applyImpl(input[i], x_prev_3f[i], i);
        }
    }
}

float LowPassSmoother::applyImpl(const float x_cur, const float x_pre, int loc)
{
    float out = exponential_smoothing(alpha, x_cur, x_pre);
    x_prev[loc] = out;
    return out;
}

cv::Point2f LowPassSmoother::applyImpl(const cv::Point2f& x_cur, const cv::Point2f& x_pre, const int loc)
{
    float out_x = exponential_smoothing(alpha, x_cur.x, x_pre.x);
    float out_y = exponential_smoothing(alpha, x_cur.y, x_pre.y);
    Point2f out(out_x, out_y);
    x_prev_2f[loc] = out;
    return out;
}

cv::Point3f LowPassSmoother::applyImpl(const cv::Point3f& x_cur, const cv::Point3f& x_pre, const int loc)
{
    float out_x = exponential_smoothing(alpha, x_cur.x, x_pre.x);
    float out_y = exponential_smoothing(alpha, x_cur.y, x_pre.y);
    float out_z = exponential_smoothing(alpha, x_cur.z, x_pre.z);
    Point3f out(out_x, out_y, out_z);
    x_prev_3f[loc] = out;
    return out;
}

void LowPassSmoother::reset()
{
    if (!x_prev.empty())
        x_prev.clear();
    if (!x_prev_2f.empty())
        x_prev_2f.clear();
    if (!x_prev_3f.empty())
        x_prev_3f.clear();
}

// >>>>>>>>>>>>>>>>>>>>> OneEuroSmoother <<<<<<<<<<<<<<<<<<<<<<<<<

void OneEuroSmoother::apply(const std::vector<float>& point, std::vector<float> &pointOut)
{
    int64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
    apply(point, timestamp, pointOut);
}
    
void OneEuroSmoother::apply(const std::vector<float>& point, int64_t timestamp, std::vector<float> &pointOut)
{
    if (dx_prev_hat.empty())
    {
        // copy the input as output, and save the data.
        dx_prev_hat.resize(point.size(), 0.0f);
        pointOut = point;
        x_prev_hat = point;
    }
    else
    {
        // Check if we need to update frequency.
        if(last_timestamp != 0 && timestamp != 0)
            frequency_ = compute_frequency(last_timestamp, timestamp);

        CV_Assert(dx_prev_hat.size() == point.size());
        CV_Assert(x_prev_hat.size() == point.size());

        pointOut.resize(point.size(), 0.0f);
        int pointNum = point.size();
        for (int i = 0; i < pointNum; ++i)
        {
            pointOut[i] = applyImpl(point[i], x_prev_hat[i], i);
        }
    }
    last_timestamp = timestamp;
}

void OneEuroSmoother::apply(const PointList2f& point, PointList2f &pointOut)
{
    int64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
    apply(point, timestamp, pointOut);
}

void OneEuroSmoother::apply(const PointList2f& point, int64_t timestamp, PointList2f &pointOut)
{
    if (dx_prev_hat_2f.empty())
    {
        // copy the input as output, and save the data.
        pointOut = point;
        x_prev_hat_2f = point;
        dx_prev_hat_2f.resize(point.size(), Point2f {0.0f, 0.0f});
    }
    else
    {
        // Check if we need to update frequency.
        if(last_timestamp != 0 && timestamp != 0)
            frequency_ = compute_frequency(last_timestamp, timestamp);

        CV_Assert(dx_prev_hat_2f.size() == point.size());
        CV_Assert(x_prev_hat_2f.size() == point.size());

        pointOut.resize(point.size(), Point2f{0.0f, 0.0f});
        int pointNum = point.size();

        for (int i = 0; i < pointNum; ++i)
        {
            pointOut[i] = applyImpl(point[i], x_prev_hat_2f[i], i);
        }
    }
    last_timestamp = timestamp;
}

void OneEuroSmoother::apply(const PointList3f& point, PointList3f &pointOut)
{
    int64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
    apply(point, timestamp, pointOut);
}

void OneEuroSmoother::apply(const PointList3f& point, int64_t timestamp, PointList3f &pointOut)
{
    if (dx_prev_hat_3f.empty())
    {
        // copy the input as output, and save the data.
        dx_prev_hat_3f.resize(point.size(), Point3f {0.0f, 0.0f, 0.0f});
        x_prev_hat_3f = point;
        pointOut = point;
    }
    else
    {
        CV_Assert(dx_prev_hat_3f.size() == point.size());
        CV_Assert(x_prev_hat_3f.size() == point.size());

        // Check if we need to update frequency.
        if(last_timestamp != 0 && timestamp != 0)
            frequency_ = compute_frequency(last_timestamp, timestamp);

        pointOut.resize(point.size(), Point3f{0.0f, 0.0f, 0.0f});
        int pointNum = point.size();
        for (int i = 0; i < pointNum; ++i)
        {
            pointOut[i] = applyImpl(point[i], x_prev_hat_3f[i], i);
        }

        last_timestamp = timestamp;
    }
}

OneEuroSmoother::~OneEuroSmoother() noexcept
{
}

float OneEuroSmoother::applyImpl(const float x_cur, const float x_pre, const int loc)
{
    this->alpha = getAlpha(frequency_, this->fc_d);
    float dx_cur = (x_cur - x_pre) * frequency_;
    float dx_cur_hat =
            exponential_smoothing(this->alpha, dx_cur, this->dx_prev_hat[loc]);

    float fc = this->fc_min + this->beta * abs(dx_cur_hat);
    this->alpha = getAlpha(frequency_, fc);
    float x_cur_hat = exponential_smoothing(this->alpha, x_cur, x_pre);
    this->x_prev_hat[loc] = x_cur_hat;
    this->dx_prev_hat[loc] = dx_cur_hat;
    return x_cur_hat;
}

// TODO check if we need Add time stamp!
Point2f OneEuroSmoother::applyImpl(const Point2f& x_cur, const Point2f& x_pre, const int loc)
{
    this->alpha = getAlpha(frequency_, this->fc_d);
    float dx_cur_x = (x_cur.x - x_pre.x) * frequency_;
    float dx_cur_y = (x_cur.y - x_pre.y) * frequency_;

    float dx_cur_hat_x =
            exponential_smoothing(this->alpha, dx_cur_x, this->dx_prev_hat_2f[loc].x);
    float dx_cur_hat_y =
            exponential_smoothing(this->alpha, dx_cur_y, this->dx_prev_hat_2f[loc].y);

    float fc_x = this->fc_min + this->beta * abs(dx_cur_hat_x);
    float fc_y = this->fc_min + this->beta * abs(dx_cur_hat_y);

    float alpha_x = getAlpha(frequency_, fc_x);
    float alpha_y = getAlpha(frequency_, fc_y);

    float x_cur_hat_x = exponential_smoothing(alpha_x, x_cur.x, x_pre.x);
    float x_cur_hat_y = exponential_smoothing(alpha_y, x_cur.y, x_pre.y);
    Point2f outPoint = {x_cur_hat_x, x_cur_hat_y};

    this->x_prev_hat_2f[loc] = outPoint;
    this->dx_prev_hat_2f[loc] = Point2f{dx_cur_hat_x, dx_cur_hat_y};
    return outPoint;
}

Point3f OneEuroSmoother::applyImpl(const Point3f& x_cur, const Point3f& x_pre, const int loc)
{
    this->alpha = getAlpha(frequency_, this->fc_d);
    float dx_cur_x = (x_cur.x - x_pre.x) * frequency_;
    float dx_cur_y = (x_cur.y - x_pre.y) * frequency_;
    float dx_cur_z = (x_cur.z - x_pre.z) * frequency_;

    float dx_cur_hat_x =
            exponential_smoothing(this->alpha, dx_cur_x, this->dx_prev_hat_3f[loc].x);
    float dx_cur_hat_y =
            exponential_smoothing(this->alpha, dx_cur_y, this->dx_prev_hat_3f[loc].y);
    float dx_cur_hat_z =
            exponential_smoothing(this->alpha, dx_cur_z, this->dx_prev_hat_3f[loc].z);

    float fc_x = this->fc_min + this->beta * abs(dx_cur_hat_x);
    float fc_y = this->fc_min + this->beta * abs(dx_cur_hat_y);
    float fc_z = this->fc_min + this->beta * abs(dx_cur_hat_z);

    float alpha_x = getAlpha(frequency_, fc_x);
    float alpha_y = getAlpha(frequency_, fc_y);
    float alpha_z = getAlpha(frequency_, fc_z);

    float x_cur_hat_x = exponential_smoothing(alpha_x, x_cur.x, x_pre.x);
    float x_cur_hat_y = exponential_smoothing(alpha_y, x_cur.y, x_pre.y);
    float x_cur_hat_z = exponential_smoothing(alpha_z, x_cur.z, x_pre.z);

    Point3f outPoint = {x_cur_hat_x, x_cur_hat_y, x_cur_hat_z};

    this->x_prev_hat_3f[loc] = outPoint;
    this->dx_prev_hat_3f[loc] = Point3f{dx_cur_hat_x, dx_cur_hat_y, dx_cur_hat_z};
    return outPoint;
}

void OneEuroSmoother::reset()
{
    x_prev_hat.clear();
    x_prev_hat_2f.clear();
    x_prev_hat_3f.clear();

    dx_prev_hat.clear();
    dx_prev_hat_2f.clear();
    dx_prev_hat_3f.clear();
}

}
