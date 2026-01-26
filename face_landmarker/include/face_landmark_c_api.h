// C API wrapper for FaceLandmarker to enable pure C consumers.

#ifndef MPP_CMAKE_FACE_LANDMARK_C_API_H
#define MPP_CMAKE_FACE_LANDMARK_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#define MPP_FACE_LANDMARK_MAX_POINTS 478

// Supported image input formats.
enum MPP_Face_ImageType {
	MPP_FACE_IMAGE_TYPE_RGB = 0,
	MPP_FACE_IMAGE_TYPE_BGR = 1,
	MPP_FACE_IMAGE_TYPE_RGBA = 2,
	MPP_FACE_IMAGE_TYPE_BGRA = 3,
	MPP_FACE_IMAGE_TYPE_YUV420 = 4, // NV21 layout: YYYY UU VV
	MPP_FACE_IMAGE_TYPE_GRAY = 5
};

// Rotation flags used by OpenCV rotate: -1 means no rotation.
enum MPP_Face_RotationType {
	MPP_FACE_ROTATION_0 = -1,
	MPP_FACE_ROTATION_90 = 0,
	MPP_FACE_ROTATION_180 = 1,
	MPP_FACE_ROTATION_270 = 2
};

// Flip flags used by OpenCV flip: 0 means no flip.
enum MPP_Face_FlipType {
	MPP_FACE_FLIP_X = -1,
	MPP_FACE_NO_FLIP = 0,
	MPP_FACE_FLIP_Y = 1
};

// Output for one detected face.
struct FaceLandmarkResult {
	int landmark_count;        // 468 or 478 depending on model variant.
	float rect[4];             // [x, y, w, h]
	float score;               // detection score
	float radians;             // face rotation angle if available
	float points[MPP_FACE_LANDMARK_MAX_POINTS * 3]; // packed as x,y,z,...
};

// Initialize FaceLandmarker instance.
// max_face_num: -1 for all detected faces or a positive cap.
// device: 0 for CPU.
__attribute__((visibility("default"))) __attribute__((used))
int initFaceLandmarker(int max_face_num, int device);

// Load face detector model from memory buffer.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelFaceDetect(const char *buffer, long buffer_size, const char *model_suffix, int device);

// Load face landmark model from memory buffer.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelFaceLandmark(const char *buffer, long buffer_size, const char *model_suffix, int device);

// Load face detector model from filesystem path.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelFaceDetectFromFile(const char *model_path, int device);

// Load face landmark model from filesystem path.
__attribute__((visibility("default"))) __attribute__((used))
int loadModelFaceLandmarkFromFile(const char *model_path, int device);

// Run landmark on a single frame (no temporal smoothing).
// Returns number of faces written to results, negative on error.
__attribute__((visibility("default"))) __attribute__((used))
int runFaceLandmarkImage(const char *data, int width, int height, int stride, int flip, int rotate,
		int img_type, struct FaceLandmarkResult *results, int results_capacity);

// Run landmark in video mode (tracking + smoothing).
// Returns number of faces written to results, negative on error.
__attribute__((visibility("default"))) __attribute__((used))
int runFaceLandmarkVideo(const char *data, int width, int height, int stride, int flip, int rotate,
		int img_type, struct FaceLandmarkResult *results, int results_capacity);

// Query active landmark dimension (468 or 478). Returns 0 if the landmark model is not loaded.
__attribute__((visibility("default"))) __attribute__((used))
int getFaceLandmarkDimension();

// Release internal resources.
__attribute__((visibility("default"))) __attribute__((used))
int releaseFaceLandmarker();

#ifdef __cplusplus
}
#endif

#endif // MPP_CMAKE_FACE_LANDMARK_C_API_H
