//
// Created by moo on 2024/4/6.
//

#include "hand_processing.h"

using namespace cv;
namespace mpp
{

// Refine the hand roi by the hand recrop model.
class HandProcessor::HandRecrop
{
public:
    HandRecrop(std::string recrop_path, int device = 0)
    {
        // Load MNN model by opencv.
        const std::string modelExt = recrop_path.substr(recrop_path.rfind('.') + 1);

        if (modelExt == "tflite")
        {
            isTFlite = true;
        }
        else if (modelExt == "mnn")
        {
            isTFlite = false;
        }
        else
            CV_Error(Error::Code::StsNotImplemented, "Not supported model type!");

        net = makePtr<dnn::Net>(dnn::readNet(recrop_path));

        // Model pre-processing
        inputName = net->getInputName();
        inputShape = net->getInputShape();
        net->setNumThreads(4);

        CV_Assert(inputShape.size() == 1);
        CV_Assert(inputShape[0].size() == 4);

        if (isTFlite)
        {
            inputWidth = inputShape[0][1];
            inputHeight = inputShape[0][2];
        }
        else
        {
            inputWidth = inputShape[0][2];
            inputHeight = inputShape[0][3];
        }

#ifdef HOLISTIC_DEBUG
        std::cout<<"HandRecrop W x H = ["<<inputWidth<<"x"<<inputHeight<<"]."<<std::endl;
#endif
        outputName = net->getOutputName();
    }

    ~HandRecrop() = default;

    /// run model with given input and roiFrom Pose, generate the roiFromRecrop.
    /// \param [in] input
    /// \param [in] roiFromPose
    /// \param [in] angleFromPose
    /// \param [out] roiFromRecrop
    /// \param [out] angleFromRecrop
    void run(const Mat &input, const Rect2f& roiFromPose, const float angleFromPose, Rect2f& roiFromRecrop, float& angleFromRecrop)
    {
        Mat imgCrop;
        Mat transMatInv;

        imageAlignment(input, inputWidth, inputHeight, roiFromPose, angleFromPose, transMatInv, imgCrop);

#ifdef HOLISTIC_DEBUG
        imshow("hand imgCrop", imgCrop);
        waitKey(1);
#endif

        // step 2: forward img
        // check if need set the swap RB?
        Mat blob = dnn::blobFromImage(imgCrop, 1/255.0, Size(), Scalar(), true);

        if (isTFlite)
        {
            transposeND(blob, {0, 2, 3, 1}, blob);
        }

        net->setInput(blob);
        Mat out = net->forward();

        CV_Assert(out.total() == netOutputTotal);

        float* p = (float* )out.data;
        Point2f p0{p[0], p[1]};
        Point2f p1{p[2], p[3]};

        // project back
        p0.x = (p[0] * transMatInv.at<double>(0, 0) + p[1] * transMatInv.at<double>(0, 1) + transMatInv.at<double>(0, 2));
        p0.y = (p[0] * transMatInv.at<double>(1, 0) + p[1] * transMatInv.at<double>(1, 1) + transMatInv.at<double>(1, 2));

        p1.x = (p[2] * transMatInv.at<double>(0, 0) + p[3] * transMatInv.at<double>(0, 1) + transMatInv.at<double>(0, 2));
        p1.y = (p[2] * transMatInv.at<double>(1, 0) + p[3] * transMatInv.at<double>(1, 1) + transMatInv.at<double>(1, 2));

        // construct roi by two points.
        float longEdge = std::max(std::abs(p1.x - p0.x), std::abs(p1.y - p0.y)); // 1/2 rect length of edge.

        roiFromRecrop.x = std::max(p0.x - longEdge, 0.f);
        roiFromRecrop.y = std::max(p0.y - longEdge, 0.f);
        roiFromRecrop.width = longEdge * 2;
        roiFromRecrop.height = longEdge * 2;

        angleFromRecrop = normalizeRadius(M_PI * 0.5 - std::atan2(-(p0.y - p1.y), p0.x - p1.x));   // compute the radians

#ifdef HOLISTIC_DEBUG
        drawRotateRect(input, roiFromRecrop, angleFromRecrop);
        drawRotateRect(input, roiFromPose, angleFromPose);

        std::cout<<"angleFromRecrop = "<<angleFromRecrop<<", angleFromPose = "<<angleFromPose<<std::endl;
        std::cout<<"p0 = "<<p0<<std::endl;
        std::cout<<"p1 = "<<p1<<std::endl;
        circle(input, p0, 1.5, Scalar(0, 128, 0), 2);
        circle(input, p1, 1.5, Scalar(0, 128, 0), 2);
        std::cout<<"rect = "<<roiFromRecrop<<std::endl;
        imshow("reCrop hand rect", input);
        waitKey(1);
#endif
    }

private:
    int inputWidth, inputHeight;
    const int netOutputTotal = 4;
    std::vector<std::string> inputName;
    std::vector<std::vector<int> > inputShape;
    std::vector<std::string> outputName;
    bool isTFlite = false;

    cv::Ptr<cv::dnn::Net> net = nullptr;
};

HandProcessor::HandProcessor(std::string recrop_path, std::string landmark_path, int device)
{
    handRecrop = makePtr<HandRecrop>(recrop_path, device);
    handLandmark = makePtr<HandLandmarker_Impl>(landmark_path, device);
}

HandProcessor::~HandProcessor()
{
}

static bool isVisibility(float visValue, float visThreshold)
{
    return visValue > visThreshold;
}

void HandProcessor::processHand(const cv::Mat &input, const std::vector<int>& handIndex, const BoxKp3& poseLm, std::vector<BoxKp3>& handOutput)
{
    handOutput.clear();
    if (!isVisibility(poseLm.vis_pre[handIndex[0]].x, thresholdWirstVisibility))
        return;

    int imgW = input.cols;
    int imgH = input.rows;
    Rect2f rectFromPose = handRectFromPoseLandmark(poseLm, handIndex, imgW, imgH);
    float radian = rotateAngleFromPoseLandmark(poseLm, handIndex, imgW, imgH);

    // shift y
    rectFromPose.y = std::max(rectFromPose.y - rectFromPose.height * 0.1f, 0.f);
    Rect2f rectFromRecrop;
    float radianFromRecrop;

    handRecrop->run(input, rectFromPose, radian, rectFromRecrop, radianFromRecrop);

#ifdef HOLISTIC_DEBUG
    rectangle(input, rectFromPose, Scalar(0, 255, 0));
    rectangle(input, rectFromRecrop, Scalar(0, 0, 255));
    imshow("reCrop processLeftHand", input);
    waitKey(1);
#endif

    // If object didn't not move a lot comparing to previous frame, we'll keep tracking it and will return previous hand box.
    // TODO Add tracking.

    PointList3f landmark_pixel, landmark_world;
    float score_pixel, score_world;

    // crop img based on hand box
    Mat imgCrop;
    Mat tranMatInv;

    int modelW, modelH;
    handLandmark->getInputWH(modelW, modelH);

    imageAlignment(input, modelW, modelH, rectFromRecrop, radianFromRecrop, tranMatInv, imgCrop);

    handLandmark->run(imgCrop, landmark_pixel, score_pixel, landmark_world, score_world);

    if (score_pixel < thresholdHandLmThreshold)
        return;

    // projects back
    PointList3f landmark_pixel_out;
    float ration_z = 1.0f/rectFromRecrop.width;
    projectLandmarkBack(landmark_pixel, input.cols, input.rows, tranMatInv, landmark_pixel_out, ration_z);

    Rect2f rectFromLandmark = getRoiFromLandmarkBoundary(landmark_pixel_out, input.cols, input.rows);
    BoxKp3 box = {};
    box.rect = rectFromLandmark;
    box.points = landmark_pixel_out;
    box.score = score_pixel;
    box.radians = radianFromRecrop;

    handOutput.push_back(box);
}

void HandProcessor::run(const cv::Mat &input, mpp::HolisticOutput &output)
{
    // step0: visibility check
    if (output.poseLandmark.empty())
        return;
    CV_Assert(output.poseLandmark.size() == 1 && "Can only handle single persone case!");

    BoxKp3& poseLm = output.poseLandmark[0];
    CV_Assert(poseLm.vis_pre.size() == poseLm.points.size() && "Poselandmark must has the visibility and presence information!");

    // step1: process left hand
    processHand(input, leftHandIndex, output.poseLandmark[0], output.leftHandLandmark);

    // step1: process right hand
    processHand(input, rightHandIndex, output.poseLandmark[0], output.rightHandLandmark);
}

Rect2f HandProcessor::handRectFromPoseLandmark(const BoxKp3 &poseLandmark, const std::vector<int>& handIndex,  const int imgW, const int imgH)
{
    Rect2f rect = getRoiFromLandmarkBoundary(poseLandmark.points, imgW, imgH, handIndex);
    return scaleRect(rect, imgW, imgH, handRectScale, true);
}

float HandProcessor::rotateAngleFromPoseLandmark(const BoxKp3 &poseLandmark, const std::vector<int>& handIndex, const int imgW, const int imgH)
{
    constexpr int kWrist = 0;
    constexpr int kPinky = 1;
    constexpr int kIndex = 2;

    cv::Point2f wristP = {poseLandmark.points[handIndex[kWrist]].x * imgW, poseLandmark.points[handIndex[kWrist]].y * imgH};
    cv::Point2f pinkyP = {poseLandmark.points[handIndex[kPinky]].x * imgW, poseLandmark.points[handIndex[kPinky]].y * imgH};
    cv::Point2f indexP = {poseLandmark.points[handIndex[kIndex]].x * imgW, poseLandmark.points[handIndex[kIndex]].y * imgH};

    // Estimate middle finger.
    const float x_middle = (2.f * indexP.x + pinkyP.x) / 3.f;
    const float y_middle = (2.f * indexP.y + pinkyP.y) / 3.f;

    double r = M_PI * 0.5 - std::atan2(-(y_middle - wristP.y), x_middle - wristP.x);   // compute the radians
    return normalizeRadius(r);
}

}