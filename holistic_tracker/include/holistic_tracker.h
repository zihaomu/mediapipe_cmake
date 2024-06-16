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

#ifndef MPP_HOLISTI_TRACKER_H
#define MPP_HOLISTI_TRACKER_H

#include "common.h"
#include "opencv2/dnn.hpp"

namespace mpp
{

struct HolisticOutput
{
    // pose landmark
    std::vector<BoxKp3> poseLandmark;

    // face landmark
    std::vector<BoxKp3> faceLandmark;

    // hand landmark
    std::vector<BoxKp3> leftHandLandmark;
    std::vector<BoxKp3> rightHandLandmark;
};

class PoseLandmarker;
class HandProcessor;
class FaceProcessor;

// The HolisticTracker class can give the face + hand + pose landmark with given input image.
// NOTE: HolisticTracker can only handle the single person scene.
class HolisticTracker
{
public:

    /// create a HolisticTracker instance.
    /// \param pose_detector
    /// \param pose_landmarker
    /// \param face_detector
    /// \param face_landmarker
    /// \param hand_recrop
    /// \param hand_landmark
    /// \param device
    HolisticTracker(std::string pose_detector, std::string pose_landmarker, std::string face_detector, std::string face_landmarker,
                    std::string hand_recrop, std::string hand_landmark, int device = 0);

    ~HolisticTracker();

    /// run the interative segmenter with given input image and points over the input.
    /// \param input input image, it should be the RGB, int8_t data type.
    /// \param output output mask, 2 channels, has the same size of the input image.
    void runImage(const cv::Mat& input, HolisticOutput& output);

    void runVideo(const cv::Mat& input, HolisticOutput& output);

private:
    int device;
    cv::Ptr<PoseLandmarker> poseLandmarker = nullptr;
    cv::Ptr<HandProcessor> handProcessor = nullptr;
    cv::Ptr<FaceProcessor> faceProcessor = nullptr;
};

}

#endif //MPP_HOLISTI_TRACKER_H
