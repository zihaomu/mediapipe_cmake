//
// Created by mzh on 2024/1/4.
// For android platform, we need rewrite the Cpp API as C-API.
//

#ifndef MPP_CMAKE_POSE_LANDMARK_C_H
#define MPP_CMAKE_POSE_LANDMARK_C_H

extern "C" {

#define MAX_POSE_LANDMARK_NUM 33

// The return value
struct PoseLandmarkResult {
    int poseNum; // 0 means no pose has been found, 1 means 1 pose has been found.
    int rect[4]; // [x, y, w, h]
    float score;
    float points[MAX_POSE_LANDMARK_NUM * 2];
};

// Test Dart result
__attribute__((visibility("default"))) __attribute__((used))
int testStruct(PoseLandmarkResult *result);

// Create PoseLandmark Instance
__attribute__((visibility("default"))) __attribute__((used))
int initPoseLandmarker();

// Load PoseDetect Model
__attribute__((visibility("default"))) __attribute__((used))
int loadModelPoseDetect(const char *buffer, const long buffer_size, bool isTFlite, int device);

// Load PoseLandmark Model from buffer
__attribute__((visibility("default"))) __attribute__((used))
int loadModelPoseLandmark(const char *buffer, const long buffer_size, bool isTFlite, int device);

// Run function
__attribute__((visibility("default"))) __attribute__((used))
/// run the function of pose landmark.
/// \param data input data buffer
/// \param width image w
/// \param height image h
/// \param stride image stride equal to mat.step
/// \param flip -1 and 1 mean flipping along the x and y axes respectively, and 0 means do not flip
/// \param rotate -1 means do not rotate. 0, 1, and 2 means rotate image 90, 180, and 270 clockwise.
/// \param img_type
/// \param result
/// \return -1 means error, 0 means success.
int runPoseLandmark(const char *data, const int width, const int height, const int stride, const int flip, const int rotate,
        const int img_type, PoseLandmarkResult *result);

__attribute__((visibility("default"))) __attribute__((used))
int releasePoseLandmark();

//#include "pose_landmark_c.cpp"
}
#endif //MPP_CMAKE_POSE_LANDMARK_C_H
