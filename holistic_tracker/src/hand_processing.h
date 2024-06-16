//
// Created by moo on 2024/4/6.
//

#ifndef MPP_CMAKE_HAND_PROCESSING_H
#define MPP_CMAKE_HAND_PROCESSING_H

#include "common.h"
#include "opencv2/dnn.hpp"
#include "holistic_tracker.h"
#include "hand_landmark_impl.h"

using namespace cv;

namespace mpp
{

class HandProcessor
{
public:
    /// Construct the HandProcessor instance.
    /// \param recrop_path the hand re-crop model path
    /// \param landmark_path the hand landmark model path
    /// \param device 0 CPU, 1 GPU.
    HandProcessor(std::string recrop_path, std::string landmark_path, int device = 0);
    ~HandProcessor();

    /// Get the hand landmark output based on the output.poseLandmark and input image.
    /// \param input input img.
    /// \param output It must contain the pose landmark value.
    void run(const cv::Mat& input, HolisticOutput& output);

private:
    Rect2f handRectFromPoseLandmark(const BoxKp3 &poseLandmark, const std::vector<int>& handIndex, const int imgW, const int imgH);
    float rotateAngleFromPoseLandmark(const BoxKp3 &poseLandmark, const std::vector<int>& handIndex, const int imgW, const int imgH);
    void processHand(const cv::Mat &input, const std::vector<int>& handIndex, const BoxKp3& poseLm, std::vector<BoxKp3>& handOutput);

    class HandRecrop;

    std::vector<int> leftHandIndex = {15, 17, 19};
    std::vector<int> rightHandIndex = {16, 18, 20};

    const float thresholdHandLmThreshold= 0.5f;
    const float thresholdWirstVisibility = 0.1f;
    const float handRectScale = 2.7f;
    Ptr<HandRecrop> handRecrop = nullptr;
    Ptr<HandLandmarker_Impl> handLandmark = nullptr;
};

}

#endif //MPP_CMAKE_HAND_PROCESSING_H
