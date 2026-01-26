// C API implementation for FaceLandmarker.
// This wrapper mirrors the pose_landmarker C API style so it can be used from pure C environments.

#include "face_landmark_c_api.h"
#include "face_landmark.h"
#include "common.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace mpp;

static FaceLandmarker *g_face_landmarker = nullptr;

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
            case MPP_FACE_IMAGE_TYPE_RGB:
            {
                cv::Mat tmp(height, width, CV_8UC3, const_cast<char *>(data), step3);
                cv::cvtColor(tmp, img, cv::COLOR_RGB2BGR);
                break;
            }
            case MPP_FACE_IMAGE_TYPE_BGR:
            {
                img = cv::Mat(height, width, CV_8UC3, const_cast<char *>(data), step3);
                break;
            }
            case MPP_FACE_IMAGE_TYPE_RGBA:
            {
                cv::Mat tmp(height, width, CV_8UC4, const_cast<char *>(data), stride > 0 ? stride : width * 4);
                cv::cvtColor(tmp, img, cv::COLOR_RGBA2BGR);
                break;
            }
            case MPP_FACE_IMAGE_TYPE_BGRA:
            {
                cv::Mat tmp(height, width, CV_8UC4, const_cast<char *>(data), stride > 0 ? stride : width * 4);
                cv::cvtColor(tmp, img, cv::COLOR_BGRA2BGR);
                break;
            }
            case MPP_FACE_IMAGE_TYPE_YUV420:
            {
                cv::Mat tmp(height + height / 2, width, CV_8UC1, const_cast<char *>(data), step1);
                cv::cvtColor(tmp, img, cv::COLOR_YUV2BGR_NV21);
                break;
            }
            case MPP_FACE_IMAGE_TYPE_GRAY:
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

        if (rotate != MPP_FACE_ROTATION_0)
        {
            if (rotate == MPP_FACE_ROTATION_90 || rotate == MPP_FACE_ROTATION_180 || rotate == MPP_FACE_ROTATION_270)
                cv::rotate(img, img, rotate);
            else
                return false;
        }

        if (flip != MPP_FACE_NO_FLIP)
        {
            if (flip == MPP_FACE_FLIP_X || flip == MPP_FACE_FLIP_Y)
                cv::flip(img, img, flip);
            else
                return false;
        }

        return true;
    }

    int run_face_landmark_internal(bool video_mode, const char *data, int width, int height, int stride, int flip, int rotate,
                                   int img_type, FaceLandmarkResult *results, int results_capacity)
    {
        if (!g_face_landmarker)
            return -1;

        if (!results || !data || results_capacity <= 0)
            return -2;

        cv::Mat img;
        if (!make_bgr_image(data, width, height, stride, img_type, flip, rotate, img))
            return -3;

        std::vector<BoxKp3> boxLandmark;
        if (video_mode)
            g_face_landmarker->runVideo(img, boxLandmark);
        else
            g_face_landmarker->runImage(img, boxLandmark);

        int face_count = std::min<int>(boxLandmark.size(), results_capacity);
        int landmark_dim = g_face_landmarker->getLandmarkSize();
        if (landmark_dim <= 0 && face_count > 0)
            landmark_dim = static_cast<int>(std::min<size_t>(boxLandmark[0].points.size(), MPP_FACE_LANDMARK_MAX_POINTS));

        for (int i = 0; i < face_count; ++i)
        {
            FaceLandmarkResult &dst = results[i];
            std::memset(&dst, 0, sizeof(FaceLandmarkResult));

            const auto &src = boxLandmark[i];
            int copy_landmark = std::min<int>(landmark_dim > 0 ? landmark_dim : static_cast<int>(src.points.size()), MPP_FACE_LANDMARK_MAX_POINTS);

            dst.landmark_count = copy_landmark;
            dst.rect[0] = src.rect.x;
            dst.rect[1] = src.rect.y;
            dst.rect[2] = src.rect.width;
            dst.rect[3] = src.rect.height;
            dst.score = src.score;
            dst.radians = src.radians;

            int point_limit = std::min<int>(copy_landmark, static_cast<int>(src.points.size()));
            for (int k = 0; k < point_limit; ++k)
            {
                dst.points[k * 3] = src.points[k].x;
                dst.points[k * 3 + 1] = src.points[k].y;
                dst.points[k * 3 + 2] = src.points[k].z;
            }
        }

        return face_count;
    }
}

int initFaceLandmarker(int max_face_num, int device)
{
    if (g_face_landmarker)
    {
        delete g_face_landmarker;
        g_face_landmarker = nullptr;
    }

    g_face_landmarker = new FaceLandmarker(max_face_num, device);
    return g_face_landmarker ? 0 : -1;
}

int loadModelFaceDetect(const char *buffer, long buffer_size, const char *model_suffix, int device)
{
    if (!g_face_landmarker || !buffer || buffer_size <= 0 || !model_suffix)
        return -1;

    g_face_landmarker->setDevice(device);
    std::string suffix(model_suffix);
    g_face_landmarker->loadDetectModel(buffer, buffer_size, suffix);
    return 0;
}

int loadModelFaceLandmark(const char *buffer, long buffer_size, const char *model_suffix, int device)
{
    if (!g_face_landmarker || !buffer || buffer_size <= 0 || !model_suffix)
        return -1;

    g_face_landmarker->setDevice(device);
    std::string suffix(model_suffix);
    g_face_landmarker->loadLandmarkModel(buffer, buffer_size, suffix);
    return 0;
}

int loadModelFaceDetectFromFile(const char *model_path, int device)
{
    if (!g_face_landmarker || !model_path)
        return -1;

    g_face_landmarker->setDevice(device);
    g_face_landmarker->loadDetectModel(std::string(model_path));
    return 0;
}

int loadModelFaceLandmarkFromFile(const char *model_path, int device)
{
    if (!g_face_landmarker || !model_path)
        return -1;

    g_face_landmarker->setDevice(device);
    g_face_landmarker->loadLandmarkModel(std::string(model_path));
    return 0;
}

int runFaceLandmarkImage(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, FaceLandmarkResult *results, int results_capacity)
{
    return run_face_landmark_internal(false, data, width, height, stride, flip, rotate, img_type, results, results_capacity);
}

int runFaceLandmarkVideo(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, FaceLandmarkResult *results, int results_capacity)
{
    return run_face_landmark_internal(true, data, width, height, stride, flip, rotate, img_type, results, results_capacity);
}

int getFaceLandmarkDimension()
{
    if (!g_face_landmarker)
        return 0;
    return g_face_landmarker->getLandmarkSize();
}

int releaseFaceLandmarker()
{
    if (g_face_landmarker)
    {
        delete g_face_landmarker;
        g_face_landmarker = nullptr;
    }
    return 0;
}
