// The code of this file is modified from:
// https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h
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

#ifndef MPP_CMAKE_FACE_LANDMARK_H
#define MPP_CMAKE_FACE_LANDMARK_H


#include "common.h"
#include "opencv2/dnn.hpp"
#include "face_detector.h"
#include "smoothing.h"

namespace mpp
{

class FaceLandmarker_Impl;
class FaceLandmarker
{
public:

    /// Construct the FaceLandmarker instance.
    /// NOTE: there are two version face_landmark model from mediapipe.
    /// the name has "with_attetion" suffix will output 478 face landmarks while w/o such suffix will output 468 points.
    /// This class will automatically switch the two different model, don't worry about the postprocessing difference.
    /// \param detector_path the face detect model path
    /// \param landmark_path the face landmark model path
    /// \param maxFaceNum -1 means return all detected face landmark.
    /// \param device 0 means cpu.
    FaceLandmarker(std::string detector_path, std::string landmark_path, int maxFaceNum = 1, int device = 0);

    FaceLandmarker(int maxFaceNum = 1, int device = 0);

    void setDevice(int deviceId);
    int getDevice() const;

    void loadDetectModel(std::string detector_path);
    // Only mnn model supported
    void loadDetectModel(const char* buffer, long buffer_size, std::string model_suffix);

    void loadLandmarkModel(std::string landmark_path);
    // Only mnn model supported
    void loadLandmarkModel(const char* buffer, long buffer_size, std::string model_suffix);

    ~FaceLandmarker();

    /// Run face landmark model on the given image. It will detect the face first
    /// and then estimate the face landmark from the given bonding box.
    /// \param img the input image.
    /// \param boxLandmark face landmark and the face bonding box.
    void runImage(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

    /// Run face landmark on the given image sequence. It will detect the face for first frame.
    /// And use the face traker to generate the bounding box for the next frame.
    /// When the detected face number hit the maxFaceNum, we will only use the face tracker instead of face detector.
    /// \param img the input image, frame by frame.
    /// \param boxLandmark face landmark and the face bonding box.
    void runVideo(const cv::Mat& img, std::vector<BoxKp3>& boxLandmark);

    /// Return landmark dimension (468 or 478 depending on model variant). Returns 0 if landmark model is not loaded.
    int getLandmarkSize();

private:
    int maxFaceNum;
    int device;

    const float detectBoxExpandRation = 1.35f;

    // Run face detector model.
    void runDetect(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles);

    /// Try to track faces with given image. And if tracker fail to track faces, we will run faceDetector.
    /// \param[in] img input image.
    /// \param[out] boxes output
    /// \param[out] angles output boxes angle one by one.
    void runTrack(const cv::Mat& img, std::vector<BoxKp2>& boxes, std::vector<float>& angles);

    // TODO check the threshold by mediapipe.
    float threshold = 0.5f;

    cv::Ptr<OneEuroSmoother> smoother; // only use in video mode.
    std::vector<BoxKp3> preBoxPoints;
    cv::Ptr<FaceLandmarker_Impl> faceLanmark_impl = nullptr;
    cv::Ptr<FaceDetector> faceDetector = nullptr;
};

} // namespace mpp

#endif //MPP_CMAKE_FACE_LANDMARK_H
