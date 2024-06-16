// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt

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

#include "iris_tracker.h"
#include "iris_tracker_impl.h"
using namespace cv;

namespace mpp
{

void draw_debug(const Mat& img, const PointList3f& points)
{
    int w = img.cols;
    int h = img.rows;

    for (int k = 0; k < points.size(); k++)
    {
        std::string scoreText = std::to_string(k);
        auto p = Point2f(points[k].x, points[k].y);
//            putText(img, scoreText, p, 2, 1, cv::Scalar(0, 255, 0));
        circle(img, p, 0.5, Scalar(255, 0, 255), 2);
    }
}

IrisTracker::IrisTracker(std::string iris_landmark_path, std::string face_detect_path,
                         std::string face_landmark_path, int _device)
: device(_device)
{
    irisLandmarker_impl = makePtr<IrisLandmarker_Impl>(iris_landmark_path, device);
    faceLandmarker = makePtr<FaceLandmarker>(face_detect_path, face_landmark_path, 1, device);
}

IrisTracker::~IrisTracker()
{
}

#define INTERNAL_IRIS_MIN(x, y) x > y ? y : x
#define INTERNAL_IRIS_MAX(x, y) x > y ? x : y

inline void projectBackResizeUnscale(const ImgScaleParams& params, float imgW, float imgH, const Rect2f& roi,
                                     PointList3f& landmarkIris, PointList3f& landmarkEye)
{
    float div_ratio = 1.0/params.ratio;
    for(int j=0; j<landmarkIris.size(); j++)
    {
        landmarkIris[j].x = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkIris[j].x - params.dw) * div_ratio + roi.x, imgW), 0.0f);
        landmarkIris[j].y = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkIris[j].y - params.dh) * div_ratio + roi.y, imgH), 0.0f);
    }

    for(int j=0; j<landmarkEye.size(); j++)
    {
        landmarkEye[j].x = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkEye[j].x - params.dw) * div_ratio + roi.x, imgW), 0.0f);
        landmarkEye[j].y = INTERNAL_IRIS_MAX(INTERNAL_IRIS_MIN((landmarkEye[j].y - params.dh) * div_ratio + roi.y, imgH), 0.0f);
    }
}

inline void flipLandmark(float imgW, PointList3f& landmarkIris, PointList3f& landmarkEye)
{
    for(int j=0; j<landmarkIris.size(); j++)
    {
        landmarkIris[j].x = imgW - landmarkIris[j].x;
    }

    for(int j=0; j<landmarkEye.size(); j++)
    {
        landmarkEye[j].x = imgW - landmarkEye[j].x;
    }
}

void IrisTracker::runImage(const cv::Mat &input, std::vector<IrisOutput> &output)
{
    CV_Assert(!input.empty());
    output.clear();

    const int imgH = input.rows;
    const int imgW = input.cols;
    std::vector<BoxKp3> boxLandmark;
    faceLandmarker->runImage(input, boxLandmark);

    // currently, IrisTracker can only handle single face scene.
    if (boxLandmark.size() != 1)
        return;

    Rect2f leftEyeRoi = getRoiFromLandmarkBoundary(boxLandmark[0].points, imgW, imgH, left_eye_boundary_index);
    Rect2f rightEyeRoi = getRoiFromLandmarkBoundary(boxLandmark[0].points, imgW, imgH, right_eye_boundary_index);

    leftEyeRoi = scaleRect(leftEyeRoi, imgW, imgH, scaleFactorEyeRoi, true);
    rightEyeRoi = scaleRect(rightEyeRoi, imgW, imgH, scaleFactorEyeRoi, true);

    int modelH, modelW;
    irisLandmarker_impl->getInputWH(modelW, modelH);

    output.resize(2);
    IrisOutput& leftResult = output[0];
    leftResult.eyeType = LEFT_EYE;
    IrisOutput& rightResult = output[1];
    rightResult.eyeType = RIGHT_EYE;

    // Processing left eye.
    Mat leftEyeImg;
    ImgScaleParams leftParams;
    resizeUnscale(input(leftEyeRoi), leftEyeImg, modelW, modelH, leftParams);

    flip(leftEyeImg, leftEyeImg, 1); // flip for left eye

    irisLandmarker_impl->run(leftEyeImg, leftResult.landmarkIris, leftResult.landmarkEye);
    flipLandmark(modelW, leftResult.landmarkIris, leftResult.landmarkEye); // flip the eye and iris landmark back.

    projectBackResizeUnscale(leftParams, imgW - 1, imgH - 1, leftEyeRoi, leftResult.landmarkIris, leftResult.landmarkEye);

    // Processing right eye.
    Mat rightEyeImg;
    ImgScaleParams rightParams;
    resizeUnscale(input(rightEyeRoi), rightEyeImg, modelW, modelH, rightParams);

    imshow("rightEyeImg", rightEyeImg);
    irisLandmarker_impl->run(rightEyeImg, rightResult.landmarkIris, rightResult.landmarkEye);
    projectBackResizeUnscale(rightParams, imgW - 1, imgH - 1, rightEyeRoi, rightResult.landmarkIris, rightResult.landmarkEye);
}

void IrisTracker::runVideo(const cv::Mat &input, std::vector<IrisOutput> &output)
{
    CV_Assert(!input.empty());
    output.clear();

    const int imgH = input.rows;
    const int imgW = input.cols;
    std::vector<BoxKp3> boxLandmark;
    faceLandmarker->runVideo(input, boxLandmark);

    // currently, IrisTracker can only handle single face scene.
    if (boxLandmark.size() != 1)
        return;

    Rect2f leftEyeRoi = getRoiFromLandmarkBoundary(boxLandmark[0].points, imgH, imgW, left_eye_boundary_index);
    Rect2f rightEyeRoi = getRoiFromLandmarkBoundary(boxLandmark[0].points, imgH, imgW, right_eye_boundary_index);

    leftEyeRoi = scaleRect(leftEyeRoi, imgW, imgH, scaleFactorEyeRoi, true);
    rightEyeRoi = scaleRect(rightEyeRoi, imgW, imgH, scaleFactorEyeRoi, true);

    int modelH, modelW;
    irisLandmarker_impl->getInputWH(modelW, modelH);

    output.resize(2);
    IrisOutput& leftResult = output[0];
    leftResult.eyeType = LEFT_EYE;
    IrisOutput& rightResult = output[1];
    rightResult.eyeType = RIGHT_EYE;

    // Processing left eye.
    Mat leftEyeImg;
    ImgScaleParams leftParams;
    resizeUnscale(input(leftEyeRoi), leftEyeImg, modelW, modelH, leftParams);

    flip(leftEyeImg, leftEyeImg, 1); // flip for left eye

    irisLandmarker_impl->run(leftEyeImg, leftResult.landmarkIris, leftResult.landmarkEye);
    flipLandmark(modelW, leftResult.landmarkIris, leftResult.landmarkEye); // flip the eye and iris landmark back.

    projectBackResizeUnscale(leftParams, imgW - 1, imgH - 1, leftEyeRoi, leftResult.landmarkIris, leftResult.landmarkEye);

    // Processing right eye.
    Mat rightEyeImg;
    ImgScaleParams rightParams;
    resizeUnscale(input(rightEyeRoi), rightEyeImg, modelW, modelH, rightParams);

    irisLandmarker_impl->run(rightEyeImg, rightResult.landmarkIris, rightResult.landmarkEye);
    projectBackResizeUnscale(rightParams, imgW - 1, imgH - 1, rightEyeRoi, rightResult.landmarkIris, rightResult.landmarkEye);
}

}
