//
// Created by moo on 2024/4/6.
//

#ifndef MPP_CMAKE_FACE_PROCESSING_H
#define MPP_CMAKE_FACE_PROCESSING_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "holistic_tracker.h"
#include "face_landmark_impl.h"
#include "face_detector.h"

using namespace cv;

namespace mpp
{

class FaceProcessor
{
public:
    FaceProcessor(std::string face_detector, std::string face_landmarker, int device = 0);
//    ~FaceProcessor() = default;

    /// Get the face landmark output based on the output.poseLandmark and input image.
    /// \param input input img.
    /// \param output It must contain the pose landmark value, and do the face_landmark
    /// based on face related pose landmark.
    void run(const cv::Mat& input, HolisticOutput& output);

private:
    Rect2f faceRectFromPoseLandmark(const std::vector<BoxKp3> &poseLandmark, const int imgW, const int imgH);

    // detect face with given roi.
    Rect2f faceRectFromDect(const cv::Mat& input, const Rect2f roi);
    float computeFaceRotationAngle(const std::vector<BoxKp3>& poseLandmark);
    // TODO add smoothing
//    void get

    int device;
    const std::vector<int> faceLdFromPose = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int rightEyeLd = 5;
    const int leftEyeLd = 2;

    const float detectBoxExpandRation = 1.35;
    const float scaleFactor = 3.0; // scale face rect factor.

    Ptr<FaceDetector> detectImpl = nullptr;    // face detection
    Ptr<FaceLandmarker_Impl> lmImpl = nullptr; // face landmarker
};

}

#endif //MPP_CMAKE_FACE_PROCESSING_H
