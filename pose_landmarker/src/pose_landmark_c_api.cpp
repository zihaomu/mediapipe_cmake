//
// Created by mzh on 2024/1/4.
//

#include "pose_landmark_c_api.h"
#include "pose_landmark.h"
#include <algorithm>

enum MPP_ImageType {
    MPP_IMAGE_TYPE_RGB = 0,
    MPP_IMAGE_TYPE_BGR = 1,
    MPP_IMAGE_TYPE_RGBA = 2,
    MPP_IMAGE_TYPE_BGRA = 3,
    MPP_IMAGE_TYPE_YUV420 = 4, // store in YYYY UU VV.
    MPP_IMAGE_TYPE_GRAY = 5,
};

enum MPP_RotationType {
    MPP_ROTATION_0 = -1,
    MPP_ROTATION_90 = 0,
    MPP_ROTATION_180 = 1,
    MPP_ROTATION_270 = 2,
};

enum MPP_FlipType {
    MPP_FLIP_X = -1, // flip image along the x axis
    MPP_NO_FLIP = 0,
    MPP_FLIP_Y = 1, // flip image along the y axis
};

#define MAX_POSE_NUM 1

__attribute__((visibility("default"))) __attribute__((used))
static mpp::PoseLandmarker* landmarker = nullptr;

int initPoseLandmarker()
{
    landmarker = new mpp::PoseLandmarker(1);

    if (landmarker)
        return 0;
    else
        return -1;
}

int loadModelPoseDetect(const char *buffer, const long buffer_size, const char *model_suffix, int device)
{
    // recovery model_suffix from  char* to std::string
    std::string model_suffix_str(model_suffix);

    LOGD("LOG of C++, PoseDetect use %s", model_suffix_str.c_str());

    if (!landmarker)
        return -1;
    landmarker->loadDetectModel(buffer, buffer_size, model_suffix_str, device);
    return 0;
}

int loadModelPoseDetectFromFile(const char *model_path, int device)
{
    if (!model_path || !landmarker)
        return -1;

    std::string path(model_path);
    landmarker->loadDetectModel(path, device);
    return 0;
}

int loadModelPoseLandmark(const char *buffer, const long buffer_size, const char *model_suffix, int device)
{
    // recovery model_suffix from  char* to std::string
    std::string model_suffix_str(model_suffix);

    LOGD("LOG of C++, PoseDetect use %s", model_suffix_str.c_str());

    if (!landmarker)
        return -1;
    landmarker->loadLandmarkModel(buffer, buffer_size, model_suffix_str, device);
    return 0;
}

int loadModelPoseLandmarkFromFile(const char *model_path, int device)
{
    if (!model_path || !landmarker)
        return -1;

    std::string path(model_path);
    landmarker->loadLandmarkModel(path, device);
    return 0;
}

int runPoseLandmark(const char* data, const int width, const int height, const int stride, const int flip, const int rotate,
        const int _img_type, PoseLandmarkResult *result)
{
    cv::TickMeter m;
    m.reset();

    if (!result || !data)
        return -1;

    LOGD("LOG of C++, img W = %d, H = %d, rotate = %d!", width, height, rotate);
    memset(result, 0, sizeof(PoseLandmarkResult));

    MPP_ImageType imageType = (MPP_ImageType)_img_type;
    m.start();
    cv::Mat img;
    if (imageType == MPP_IMAGE_TYPE_RGB)
    {
        cv::Mat _img = cv::Mat(height, width, CV_8UC3, (void *)data);
        cv::cvtColor(_img, img, cv::COLOR_RGB2BGR);
    }
    else if (imageType == MPP_IMAGE_TYPE_BGR)
    {
        img = cv::Mat(height, width, CV_8UC3, (void *)data);
    }
    else if (imageType == MPP_IMAGE_TYPE_RGBA)
    {
        cv::Mat _img = cv::Mat(height, width, CV_8UC4, (void *)data);
        cv::cvtColor(_img, img, cv::COLOR_RGBA2BGR);
    }
    else if (imageType == MPP_IMAGE_TYPE_BGRA)
    {
        cv::Mat _img = cv::Mat(height, width, CV_8UC4, (void *)data);
        cv::cvtColor(_img, img, cv::COLOR_BGRA2BGR);
    }
    else if (imageType == MPP_IMAGE_TYPE_YUV420)
    {
        cv::Mat _img = cv::Mat(height * 1.5, width, CV_8UC1, (void *)data);
        cv::cvtColor(_img, img, cv::COLOR_YUV2BGR_NV21);
    }
    else
        return -2;

    if (rotate != -1)
    {
        CV_Assert(rotate == 0 || rotate == 1 || rotate == 2);
        cv::rotate(img, img, rotate);
    }

    if (flip != 0)
    {
        CV_Assert(flip == 1 || flip == -1);
        if (flip == 1)
            cv::flip(img, img, MPP_FLIP_X);
        else if (flip == -1)
            cv::flip(img, img, MPP_FLIP_Y);
    }
    m.stop();
    LOGD("LOG of C++, Image Finish preprocessing, w = %d, h = %d! takes time %f ms!", img.cols, img.rows, m.getTimeMilli());

    std::vector<BoxKp3> boxLandmark;
    landmarker->runVideo(img, boxLandmark);

    if (boxLandmark.size() > 0)
    {
        auto& p = boxLandmark[0];
        result->poseNum = 1;
        result->score = p.score;
        result->rect[0] = p.rect.x;
        result->rect[1] = p.rect.y;
        result->rect[2] = p.rect.width;
        result->rect[3] = p.rect.height;

        int copy_num = std::min<int>(MAX_POSE_LANDMARK_NUM, static_cast<int>(p.points.size()));
        for (int i = 0; i < copy_num; i++)
        {
            result->points[i * 3] = p.points[i].x;
            result->points[i * 3 + 1] = p.points[i].y;
            result->points[i * 3 + 2] = p.points[i].z;
        }
    }

    return 0;
}

int releasePoseLandmark()
{
    if (landmarker)
        delete landmarker;
    return 0;
}