// C API wrapper for HandLandmarker to enable pure C consumers.

#ifndef MPP_CMAKE_HAND_LANDMARK_C_API_H
#define MPP_CMAKE_HAND_LANDMARK_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#define MPP_HAND_LANDMARK_MAX_POINTS 21

// Supported image input formats.
enum MPP_Hand_ImageType {
    MPP_HAND_IMAGE_TYPE_RGB = 0,
    MPP_HAND_IMAGE_TYPE_BGR = 1,
    MPP_HAND_IMAGE_TYPE_RGBA = 2,
    MPP_HAND_IMAGE_TYPE_BGRA = 3,
    MPP_HAND_IMAGE_TYPE_YUV420 = 4, // NV21 layout: YYYY UU VV
    MPP_HAND_IMAGE_TYPE_GRAY = 5
};

// Rotation flags used by OpenCV rotate: -1 means no rotation.
enum MPP_Hand_RotationType {
    MPP_HAND_ROTATION_0 = -1,
    MPP_HAND_ROTATION_90 = 0,
    MPP_HAND_ROTATION_180 = 1,
    MPP_HAND_ROTATION_270 = 2
};

// Flip flags used by OpenCV flip: 0 means no flip.
enum MPP_Hand_FlipType {
    MPP_HAND_FLIP_X = -1,
    MPP_HAND_NO_FLIP = 0,
    MPP_HAND_FLIP_Y = 1
};

// Output for one detected hand.
struct HandLandmarkResult {
    int landmark_count;        // Always 21 for hand landmarks
    float rect[4];             // [x, y, w, h]
    float score;               // detection score
    float radians;             // hand rotation angle
    float points[MPP_HAND_LANDMARK_MAX_POINTS * 3]; // packed as x,y,z,...
};

// Initialize HandLandmarker instance.
// max_hand_num: -1 for all detected hands or a positive cap.
// device: 0 for CPU.
__attribute__((visibility("default"))) __attribute__((used))
int initHandLandmarker(int max_hand_num, int device);

// Load hand detector model from memory buffer.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelHandDetect(const char *buffer, long buffer_size, const char *model_suffix, int device);

// Load hand landmark model from memory buffer.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelHandLandmark(const char *buffer, long buffer_size, const char *model_suffix, int device);

// Load hand detector model from filesystem path.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelHandDetectFromFile(const char *model_path, int device);

// Load hand landmark model from filesystem path.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelHandLandmarkFromFile(const char *model_path, int device);

// Run landmark on a single frame (no temporal smoothing).
// Returns number of hands written to results, negative on error.
__attribute__((visibility("default"))) __attribute__((used))
int runHandLandmarkImage(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, struct HandLandmarkResult *results, int results_capacity);

// Run landmark in video mode (tracking + smoothing).
// Returns number of hands written to results, negative on error.
__attribute__((visibility("default"))) __attribute__((used))
int runHandLandmarkVideo(const char *data, int width, int height, int stride, int flip, int rotate,
        int img_type, struct HandLandmarkResult *results, int results_capacity);

// Release internal resources.
__attribute__((visibility("default"))) __attribute__((used))
int releaseHandLandmarker();

#ifdef __cplusplus
}
#endif

#endif // MPP_CMAKE_HAND_LANDMARK_C_API_H
