// C API implementation for HandLandmarker.
// This wrapper mirrors the face_landmarker C API style for consistency.

#include "hand_landmark_c_api.h"
#include "hand_landmark.h"
#include "common.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace mpp;

static HandLandmarker *g_hand_landmarker = nullptr;

namespace
{
    bool make_bgr_image(const char *data, int width, int height, int stride, int img_type, int flip, int rotate, cv::Mat &img)
    {
        if (!data)
            return false;

        const int step3 = stride > 0 ? stride : width * 3;
        const int step1 = stride > 0 ? stride : width;

        switch (img_type)
        {
            case MPP_HAND_IMAGE_TYPE_RGB:
            {
                cv::Mat tmp(height, width, CV_8UC3, const_cast<char *>(data), step3);
                cv::cvtColor(tmp, img, cv::COLOR_RGB2BGR);
                break;
            }
            case MPP_HAND_IMAGE_TYPE_BGR:
            {
                img = cv::Mat(height, width, CV_8UC3, const_cast<char *>(data), step3);
                break;
            }
            case MPP_HAND_IMAGE_TYPE_RGBA:
            {
                cv::Mat tmp(height, width, CV_8UC4, const_cast<char *>(data), stride > 0 ? stride : width * 4);
                cv::cvtColor(tmp, img, cv::COLOR_RGBA2BGR);
                break;
            }
            case MPP_HAND_IMAGE_TYPE_BGRA:
            {
                cv::Mat tmp(height, width, CV_8UC4, const_cast<char *>(data), stride > 0 ? stride : width * 4);
                cv::cvtColor(tmp, img, cv::COLOR_BGRA2BGR);
                break;
            }
            case MPP_HAND_IMAGE_TYPE_YUV420:
            {
                cv::Mat tmp(height + height / 2, width, CV_8UC1, const_cast<char *>(data), step1);
                cv::cvtColor(tmp, img, cv::COLOR_YUV2BGR_NV21);
                break;
            }
            case MPP_HAND_IMAGE_TYPE_GRAY:
            {
                cv::Mat tmp(height, width, CV_8UC1, const_cast<char *>(data), step1);
                cv::cvtColor(tmp, img, cv::COLOR_GRAY2BGR);
                break;
            }
            default:
                return false;
        }

        if (img.empty())
            return false;

        if (rotate != MPP_HAND_ROTATION_0)
        {
            if (rotate == MPP_HAND_ROTATION_90 || rotate == MPP_HAND_ROTATION_180 || rotate == MPP_HAND_ROTATION_270)
                cv::rotate(img, img, rotate);
            else
                return false;
        }

        if (flip != MPP_HAND_NO_FLIP)
        {
            if (flip == MPP_HAND_FLIP_X || flip == MPP_HAND_FLIP_Y)
                cv::flip(img, img, flip);
            else
                return false;
        }

        return true;
    }

    int run_hand_landmark_internal(bool video_mode, const char *data, int width, int height, int stride, int flip, int rotate,
                                   int img_type, HandLandmarkResult *results, int results_capacity)
    {
        if (!g_hand_landmarker)
            return -1;

        if (!results || !data || results_capacity <= 0)
            return -2;

        cv::Mat img;
        if (!make_bgr_image(data, width, height, stride, img_type, flip, rotate, img))
            return -3;

        std::vector<BoxKp3> boxLandmark;
        if (video_mode)
            g_hand_landmarker->runVideo(img, boxLandmark);
        else
            g_hand_landmarker->runImage(img, boxLandmark);

        int hand_count = std::min<int>(boxLandmark.size(), results_capacity);

        for (int i = 0; i < hand_count; ++i)
        {
            HandLandmarkResult &dst = results[i];
            std::memset(&dst, 0, sizeof(HandLandmarkResult));

            const auto &src = boxLandmark[i];
            int copy_landmark = std::min<int>(static_cast<int>(src.points.size()), MPP_HAND_LANDMARK_MAX_POINTS);

            dst.landmark_count = copy_landmark;
            dst.rect[0] = src.rect.x;
            dst.rect[1] = src.rect.y;
            dst.rect[2] = src.rect.width;
            dst.rect[3] = src.rect.height;
            dst.score = src.score;
            dst.radians = src.radians;

            // Copy pixel coordinates
            for (int k = 0; k < copy_landmark; ++k)
            {
                dst.points[k * 3] = src.points[k].x;
                dst.points[k * 3 + 1] = src.points[k].y;
                dst.points[k * 3 + 2] = src.points[k].z;
            }
        }

        return hand_count;
    }
}

int initHandLandmarker(int max_hand_num, int device)
{
    if (g_hand_landmarker)
    {
        delete g_hand_landmarker;
        g_hand_landmarker = nullptr;
    }

    g_hand_landmarker = new HandLandmarker(max_hand_num, device);
    return g_hand_landmarker ? 0 : -1;
}

int loadModelHandDetect(const char *buffer, long buffer_size, const char *model_suffix, int device)
{
    if (!g_hand_landmarker || !buffer || buffer_size <= 0 || !model_suffix)
        return -1;

    std::string suffix(model_suffix);
    g_hand_landmarker->loadDetectModel(buffer, buffer_size, suffix);
    return 0;
}

int loadModelHandLandmark(const char *buffer, long buffer_size, const char *model_suffix, int device)
{
    if (!g_hand_landmarker || !buffer || buffer_size <= 0 || !model_suffix)
        return -1;

    std::string suffix(model_suffix);
    g_hand_landmarker->loadLandmarkModel(buffer, buffer_size, suffix);
    return 0;
}

int loadModelHandDetectFromFile(const char *model_path, int device)
{
    if (!g_hand_landmarker || !model_path)
        return -1;

    g_hand_landmarker->loadDetectModel(std::string(model_path));
    return 0;
}

int loadModelHandLandmarkFromFile(const char *model_path, int device)
{
    if (!g_hand_landmarker || !model_path)
        return -1;

    g_hand_landmarker->loadLandmarkModel(std::string(model_path));
    return 0;
}

int runHandLandmarkImage(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, HandLandmarkResult *results, int results_capacity)
{
    return run_hand_landmark_internal(false, data, width, height, stride, flip, rotate, img_type, results, results_capacity);
}

int runHandLandmarkVideo(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, HandLandmarkResult *results, int results_capacity)
{
    return run_hand_landmark_internal(true, data, width, height, stride, flip, rotate, img_type, results, results_capacity);
}

int releaseHandLandmarker()
{
    if (g_hand_landmarker)
    {
        delete g_hand_landmarker;
        g_hand_landmarker = nullptr;
    }
    return 0;
}
