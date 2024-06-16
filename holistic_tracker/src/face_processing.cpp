//
// Created by moo on 2024/4/6.
//

#include "face_processing.h"

namespace mpp
{

FaceProcessor::FaceProcessor(std::string face_detector, std::string face_landmarker, int _device)
:device(_device)
{
    detectImpl = makePtr<FaceDetector>(face_detector, device);
    lmImpl = makePtr<FaceLandmarker_Impl>(face_landmarker, device);
}

inline Rect2f getRectFromFaceLandmark(const BoxKp3& points, int imgW, int imgH)
{
    // Find boundaries of landmarks.
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    auto& pointList = points.points;
    int pointsSize = pointList.size();

    for (int i = 0; i < pointsSize; ++i)
    {
        max_x = std::max(max_x, pointList[i].x);
        max_y = std::max(max_y, pointList[i].y);
        min_x = std::min(min_x, pointList[i].x);
        min_y = std::min(min_y, pointList[i].y);
    }

    const float center_x = (max_x + min_x) * imgW / 2.f;
    const float center_y = (max_y + min_y) * imgH / 2.f;

    // bounding box
    float width = (max_x - min_x) * imgW;
    float height = (max_y - min_y) * imgH;

    // TO FINE TUNing the following params.
    float expandRatio =1.10f;
    float longEdge = std::max(width, height) * expandRatio;

    Rect2f rect;
    rect.width = longEdge;
    rect.height = longEdge;

    rect.x = center_x - 0.5f * longEdge;
    rect.y = center_y - 0.5f * longEdge;

    return rect;
}

float FaceProcessor::computeFaceRotationAngle(const std::vector<BoxKp3> &poseLandmark)
{
    CV_Assert(poseLandmark.size() == 1);
    float x0 = poseLandmark[0].points[leftEyeLd].x;
    float y0 = poseLandmark[0].points[leftEyeLd].y;
    float x1 = poseLandmark[0].points[rightEyeLd].x;
    float y1 = poseLandmark[0].points[rightEyeLd].y;

    double r = -std::atan2(-(y0 - y1), x0 - x1);   // compute the radians
    return normalizeRadius(r);
}

Rect2f FaceProcessor::faceRectFromDect(const cv::Mat &input, const cv::Rect2f roi)
{
    // crop img from frame
    Mat inpCrop = input(roi).clone();

    std::vector<BoxKp2> boxs;
    detectImpl->run(inpCrop, boxs);
    const int faceNum = boxs.size();

    Rect2f r;
    if (faceNum == 0)
    {
        return roi;
    }
    else if (faceNum >= 1)
    {
        r = boxs[0].rect;
        r.x += roi.x;
        r.y += roi.y;
        return r;
    }
}

static Rect2f scaleBox(const Rect2f& rect, float scaleFactor)
{
    Rect2f out = rect;
    float extendFactor = (scaleFactor - 1.0) * 0.5;

    auto& rectScaled = out;
    rectScaled.x = rectScaled.x - rectScaled.width * extendFactor;
    rectScaled.y = rectScaled.y - rectScaled.height * extendFactor;

    rectScaled.width = rectScaled.width * scaleFactor;
    rectScaled.height = rectScaled.height * scaleFactor;

    return out;
}

void FaceProcessor::run(const cv::Mat &input, HolisticOutput& output)
{
    output.faceLandmark.clear();
    if (output.poseLandmark.empty()) // No pose no face
        return;

    CV_Assert(output.poseLandmark.size() == 1);

    Rect2f faceRectPose = faceRectFromPoseLandmark(output.poseLandmark, input.cols, input.rows);
    Rect2f faceRectDetect = faceRectFromDect(input, faceRectPose);
    Rect2f faceRect = scaleBox(faceRectDetect, detectBoxExpandRation);
#ifdef HOLISTIC_DEBUG
    rectangle(input, faceRectPose, Scalar(255, 0, 0));
    rectangle(input, faceRect, Scalar(0, 255, 0));
    imshow("face  faceRectPose", input);
    waitKey(1);
#endif

    Mat imgCrop, tranMatInv;
    float angle = computeFaceRotationAngle(output.poseLandmark); // TODO compute angle based on eye.
    int H, W;
    lmImpl->getInputWH(W, H);
    imageAlignment(input, W, H, faceRect, angle, tranMatInv, imgCrop);

    PointList3f landmark, landmarkProjected;
    float landmark_score = 0;
    lmImpl->run(imgCrop, landmark, landmark_score);
    float ratio_z = faceRect.width / (float)W / float(input.cols);
    projectLandmarkBack(landmark, input.cols, input.rows, tranMatInv, landmarkProjected, ratio_z);

    BoxKp3 outLd = {};
    outLd.score = landmark_score;
    outLd.points = landmarkProjected;
    outLd.radians = 0;
    outLd.rect = getRectFromFaceLandmark(outLd, input.cols, input.rows);
    output.faceLandmark.push_back(outLd);
}

Rect2f FaceProcessor::faceRectFromPoseLandmark(const std::vector<BoxKp3> &poseLandmark, const int imgW, const int imgH)
{
    Rect2f rect = getRoiFromLandmarkBoundary(poseLandmark[0].points, imgW, imgH, faceLdFromPose);
    return scaleRect(rect, imgW, imgH, scaleFactor, true);
}

}