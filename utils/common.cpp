//
// Created by mzh on 2023/8/23.
//

#include "common.h"

namespace mpp
{

float normalizeRadius(float r)
{
    return r = r - 2 * M_PI * std::floor((r - (-M_PI)) / (2 * M_PI)); // normalize radians, get the angle to (+/- π)
}

void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_width, int target_height)
{
    ImgScaleParams params;
    resizeUnscale(mat, mat_rs, target_width, target_height, params);
}

void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_width, int target_height, ImgScaleParams& params)
{
    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(0, 0, 0));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float) target_width / (float) img_width;
    float h_r = (float) target_height / (float) img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float) img_width * r); // floor
    int new_unpad_h = static_cast<int>((float) img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    params.ratio = r;
    params.dw = dw;
    params.dh = dh;
    params.flag = true;
}

void projectLandmarkBack(const PointList3f& srcLandmark, int imageW, int imageH, const cv::Mat& transMatInv, PointList3f& dstLandmark, float ratio_z)
{
    CV_Assert(srcLandmark.data() != nullptr);

    if (dstLandmark.data() != nullptr)
        CV_Assert(srcLandmark.data() != dstLandmark.data() && "srcLandmark and dstLandmark should not be the same.");

    int sizeLandmark = srcLandmark.size();
    dstLandmark.resize(sizeLandmark);

    float div_w = 1.0/imageW;
    float div_h = 1.0/imageH;

    for (int i = 0; i < sizeLandmark; i++)
    {
        float x = srcLandmark[i].x;
        float y = srcLandmark[i].y;

        dstLandmark[i].x = (x * transMatInv.at<double>(0, 0) + y * transMatInv.at<double>(0, 1) + transMatInv.at<double>(0, 2)) * div_w;
        dstLandmark[i].y = (x * transMatInv.at<double>(1, 0) + y * transMatInv.at<double>(1, 1) + transMatInv.at<double>(1, 2)) * div_h;
        dstLandmark[i].z = srcLandmark[i].z * ratio_z;
    }
}

void imageAlignment(const cv::Mat& src, int dstW, int dstH, const BoxKp2& box, float angle, cv::Mat& transMatInv, cv::Mat& dst, cv::Point2f center)
{
    // use box center as center
    if (center.x == 0 && center.y == 0)
    {
        center.x = (box.rect.x + box.rect.width/2);
        center.y = (box.rect.y + box.rect.height/2);
    }

    float cos_r = std::cos(angle);
    float sin_r = std::sin(angle);

    std::vector<cv::Point2f> dstPts(4);
    std::vector<cv::Point2f> srcPts(4);

    dstPts[0] = cv::Point2f(0, 0);
    dstPts[1] = cv::Point2f(dstW, 0);
    dstPts[2] = cv::Point2f(dstW, dstH);
    dstPts[3] = cv::Point2f(0, dstH);

    auto& rect = box.rect;

    // NOTE: Instead of using the center of the hand detection box, use the center of the palm part.
    // This is done because the hand and finger rotates around the palm of the hand.
    float center_x = center.x;
    float center_y = center.y;

    srcPts[0] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[1] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[2] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r +
            center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r +
            center_y
    );

    srcPts[3] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r + center_y
    );

    cv::Mat transMat = cv::getAffineTransform(srcPts.data(), dstPts.data());

    cv::warpAffine(src, dst, transMat, cv::Size(dstW, dstH), 1, 0);
    cv::invertAffineTransform(transMat, transMatInv);
}

void imageAlignment(const cv::Mat& src, int dstW, int dstH, const cv::Rect2f& rect, float angle, cv::Mat& transMatInv, cv::Mat& dst, bool flip)
{
    cv::Point2f center;
    center.x = (rect.x + rect.width/2);
    center.y = (rect.y + rect.height/2);

    float cos_r = std::cos(angle);
    float sin_r = std::sin(angle);

    std::vector<cv::Point2f> dstPts(4);
    std::vector<cv::Point2f> srcPts(4);

    if (flip)
    {
        dstPts[0] = cv::Point2f(dstW, 0);
        dstPts[1] = cv::Point2f(0, 0);
        dstPts[2] = cv::Point2f(0, dstH);
        dstPts[3] = cv::Point2f(dstW, dstH);
    }
    else
    {
        dstPts[0] = cv::Point2f(0, 0);
        dstPts[1] = cv::Point2f(dstW, 0);
        dstPts[2] = cv::Point2f(dstW, dstH);
        dstPts[3] = cv::Point2f(0, dstH);
    }

    // NOTE: Instead of using the center of the hand detection box, use the center of the palm part.
    // This is done because the hand and finger rotates around the palm of the hand.
    float center_x = center.x;
    float center_y = center.y;

    srcPts[0] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[1] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y - center_y) * sin_r + center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y - center_y) * cos_r + center_y
    );

    srcPts[2] = cv::Point2f(
            (rect.x + rect.width - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r +
            center_x,
            (rect.x + rect.width - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r +
            center_y
    );

    srcPts[3] = cv::Point2f(
            (rect.x - center_x) * cos_r - (rect.y + rect.height - center_y) * sin_r + center_x,
            (rect.x - center_x) * sin_r + (rect.y + rect.height - center_y) * cos_r + center_y
    );

    cv::Mat transMat = cv::getAffineTransform(srcPts.data(), dstPts.data());

    cv::warpAffine(src, dst, transMat, cv::Size(dstW, dstH), 1, 0);
    cv::invertAffineTransform(transMat, transMatInv);
}

void drawConnection(cv::Mat& src, const std::vector<cv::Point3f>& points, const std::vector<std::vector<int > > connection,
                        float x_scale, float y_scale)
{
    int lineSize = connection.size();
    for (int i = 0; i < lineSize; i++)
    {
        CV_Assert(connection[i].size() == 2);
        auto data = connection[i];
        auto& p0 = points[connection[i][0]];
        auto& p1 = points[connection[i][1]];

        cv::Point2f startPoint{p0.x * x_scale, p0.y * y_scale};
        cv::Point2f endPoint{p1.x * x_scale, p1.y * y_scale};
        cv::line(src, startPoint, endPoint, cv::Scalar(0, 255, 0), 2);
    }
}

// TODO check if fix the normalized of landmark value
cv::Rect2f getRoiFromLandmarkBoundary(const PointList3f& faceLandmark, const int imgW, const int imgH,
                                       const std::vector<int> boundary_index)
{
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    if (boundary_index.empty())
    {
        for (int i = 0; i < faceLandmark.size(); i++)
        {
            auto& p = faceLandmark[i];

            min_x = min_x > p.x ? p.x : min_x;
            min_y = min_y > p.y ? p.y : min_y;

            max_x = max_x < p.x ? p.x : max_x;
            max_y = max_y < p.y ? p.y : max_y;
        }
    }
    else
    {
        for (int i = 0; i < boundary_index.size(); i++)
        {
            auto& p = faceLandmark[boundary_index[i]];

            min_x = min_x > p.x ? p.x : min_x;
            min_y = min_y > p.y ? p.y : min_y;

            max_x = max_x < p.x ? p.x : max_x;
            max_y = max_y < p.y ? p.y : max_y;
        }
    }

    return {min_x * imgW, min_y * imgH, (max_x - min_x) * imgW, (max_y - min_y) * imgH};
}

cv::Rect2f scaleRect(const cv::Rect2f& rect, const int _imgW, const int _imgH, float scaleFactor, bool square_long)
{
    const int imgW = _imgW - 1;
    const int imgH = _imgH - 1;
    cv::Point2f center = {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
    cv::Rect2f rectScaled = rect;

    if (square_long)
    {
        float longSquare = std::max(rectScaled.width, rectScaled.height);
        rectScaled.width = std::min(longSquare * scaleFactor, (imgW - center.x) * 2.f);
        rectScaled.height = std::min(longSquare * scaleFactor, (imgH - center.y) * 2.f);

        rectScaled.x = std::max(center.x - rectScaled.width * 0.5f, 0.f);
        rectScaled.y = std::max(center.y - rectScaled.height * 0.5f, 0.f);
    }
    else
    {
        rectScaled.width = std::min(rectScaled.width * scaleFactor, (imgW - center.x) * 2.f);
        rectScaled.height = std::min(rectScaled.height * scaleFactor, (imgH - center.y) * 2.f);

        rectScaled.x = std::max(center.x - rectScaled.width * 0.5f, 0.f);
        rectScaled.y = std::max(center.y - rectScaled.height * 0.5f, 0.f);
    }

    return rectScaled;
}

// Function to rotate a point around a center by a given angle (in degrees)
inline cv::Point2f rotatePoint(const cv::Point2f& pt, const cv::Point2f& center, float angle) {
    // Convert angle from degrees to radians
    float rad = angle * CV_PI / 180.0;

    // Translate point back to origin
    float x_new = pt.x - center.x;
    float y_new = pt.y - center.y;

    // Apply rotation
    float x_rot = x_new * cos(rad) - y_new * sin(rad);
    float y_rot = x_new * sin(rad) + y_new * cos(rad);

    // Translate point back to original location
    x_rot += center.x;
    y_rot += center.y;

    return cv::Point2f(x_rot, y_rot);
}


void drawRotateRect(const cv::Mat& src, const cv::Rect2f& roi, const float radian, cv::Point2f center)
{
    // use box center as center
    bool useRectCenter = false;
    if (center.x == 0 && center.y == 0)
    {
        center.x = (roi.x + roi.width/2);
        center.y = (roi.y + roi.height/2);
        useRectCenter = true;
    }
    const float angle = radian * 180 / M_PI; // TODO check

    cv::Point2f vertices[4];
    if (useRectCenter)
    {
        cv::RotatedRect rRect = cv::RotatedRect(center, cv::Size(roi.width, roi.height), angle);
        rRect.points(vertices);
    }
    else
    {
        // the order bottomLeft, topLeft, topRight, bottomRight.
        vertices[1] = rotatePoint(cv::Point2f(roi.x, roi.y), center, angle);
        vertices[2] = rotatePoint(cv::Point2f(roi.x + roi.width, roi.y), center, angle);
        vertices[0] = rotatePoint(cv::Point2f(roi.x, roi.y + roi.height), center, angle);
        vertices[3] = rotatePoint(cv::Point2f(roi.x + roi.width, roi.y + roi.height), center, angle);
    }

    for (int i = 0; i < 4; i++)
        cv::line(src, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0));
}

} // namespace mpp