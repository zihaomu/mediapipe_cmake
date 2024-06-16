//
// Created by mzh on 2023/8/21.
//

#ifndef MPP_REPRODUCE_COMMON_H
#define MPP_REPRODUCE_COMMON_H

#include <iostream>

// base data type
#include "base_type.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"
#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace mpp
{

#ifdef __ANDROID__
#define LOG_TAG "aisdk_native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
// TODO, delete comment to print log.
#elif 0 && (defined(__unix__) || defined(_WIN32) || defined(__APPLE__))
#define LOGD(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#define LOGI(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#define LOGW(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#else
#define LOGD(...);
#define LOGI(...);
#define LOGW(...);
//#define LOGE(...);
#define LOGE(fmt, ...) printf(fmt"\n", ##__VA_ARGS__)
#endif
// Define the output data structure.

typedef struct
{
    float ratio;
    int dw;
    int dh;
    bool flag;
} ImgScaleParams;

typedef std::vector<cv::Point3f> PointList3f;
typedef std::vector<cv::Point2f> PointList2f;

float normalizeRadius(float r);

// resize image and keep its ratio.
void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_width, int target_height, ImgScaleParams& params);

// project landmark based on the inverse affine transformation matrix.
void projectLandmarkBack(const PointList3f& srcLandmark, int imageW, int imageH, const cv::Mat& transMatInv, PointList3f& dstLandmark, float ratio_z = 1.0f);

/// face alignment or hand alignment.
/// \param src the input image
/// \param dstW expected image width
/// \param dstH expected image height
/// \param box detected bonding box
/// \param angle rotated angle
/// \param transMatInv inverse affine transformation matrix, it is needed in the postprocessing.
/// \param dst output, aligned image.
/// \param center rotated center. optional, by default we use the bounding box center as the center.
void imageAlignment(const cv::Mat& src, int dstW, int dstH, const BoxKp2& box, float angle, cv::Mat& transMatInv, cv::Mat& dst, cv::Point2f center = {0, 0});

// overload
void imageAlignment(const cv::Mat& src, int dstW, int dstH, const cv::Rect2f& roi, float angle, cv::Mat& transMatInv, cv::Mat& dst, bool flip = false);


/// DrawConnection based on given image, key
/// \param src input image.
/// \param points all the points list.
/// \param connection the connection points one by one.
/// \param x_scale use to scale the normalized x-axis value
/// \param y_scale use to scale the normalized y-axis value
void drawConnection(cv::Mat& src, const std::vector<cv::Point3f>& points, const std::vector<std::vector<int > > connection,
                    float x_scale = 1.0f, float y_scale = 1.0f);


/// Draw the opencv Rotation Rect based on  on the given img.
/// \param src input img
/// \param roi rect
/// \param radian rotation radian
/// \param center rotation center point, the default value mean rotate the rect based on rect center.
void drawRotateRect(const cv::Mat& src, const cv::Rect2f& roi, const float radian, cv::Point2f center = {0, 0});

cv::Rect2f scaleRect(const cv::Rect2f& rect, const int _imgW, const int _imgH, float scaleFactor, bool square_long = false);

cv::Rect2f getRoiFromLandmarkBoundary(const PointList3f& faceLandmark, const int imgH, const int imgW,
                                      const std::vector<int> boundary_index = {});

}

#endif //MPP_REPRODUCE_COMMON_H
