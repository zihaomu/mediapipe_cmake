//
// Created by mzh on 2023/8/21.
//

#ifndef MPP_REPRODUCE_NON_MAX_SUPPRESSION_H
#define MPP_REPRODUCE_NON_MAX_SUPPRESSION_H

#include <cmath>
#include <utility>
#include <vector>

#include "common.h"

namespace mpp
{
float iou(const cv::Rect2f& rect0, const cv::Rect2f & rect1);
float iou(const BoxKp2& rect0, const BoxKp2& rect1);

/// NMS Very naive NMS implementation
/// \param rects the input bounding box with detection score.
/// \param iouThr iou threshold, default = 0.5
void nms_simple(std::vector<BoxKp2>& rects, float iouThr = 0.5, int topK = -1);

// TODO finis the complicate NMS
//struct NMSProcessorOption
//{
//    enum OverlapType {
//        UNSPECIFIED_OVERLAP_TYPE = 0,
//        JACCARD = 1,
//        MODIFIED_JACCARD = 2,
//        INTERSECTION_OVER_UNION = 3
//    };
//
//    enum NmsAlgorithm {
//        DEFAULT = 0,
//        // Only supports relative bounding box for weighted NMS.
//        WEIGHTED = 1,
//    };
//
//    int num_detection_streams = 1;
//    int max_num_detections = -1;
//    float min_score_threshold = -1.0f;
//    float min_suppression_threshold = 1.0f;
//
//    OverlapType overlap_type = JACCARD;
//    NmsAlgorithm algorithm = DEFAULT;
//};
//
//// Show with class or api??
//class NMSProcessor
//{
//    NMSProcessor(std::vector<>());
//
//
//};

} // namespace mpp

#endif //MPP_REPRODUCE_NON_MAX_SUPPRESSION_H
