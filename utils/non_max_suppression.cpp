//
// Created by mzh on 2023/8/21.
//

#include "non_max_suppression.h"
#include "opencv2/opencv.hpp"

using namespace cv;
namespace mpp {

float iou(const BoxKp2& rect0, const BoxKp2& rect1)
{
    return iou(rect0.rect, rect1.rect);
}

float iou(const cv::Rect2f& box1, const cv::Rect2f & box2)
{
    if (box1.x > box2.x+box2.width) { return 0.0; }
    if (box1.y > box2.y+box2.height) { return 0.0; }
    if (box1.x+box1.width < box2.x) { return 0.0; }
    if (box1.y+box1.height < box2.y) { return 0.0; }

    cv::Rect2f interesectBox  = box1 & box2;
    cv::Rect2f unionBox  = box1 | box2;
    return interesectBox.area() / unionBox.area();
}

void nms_simple(std::vector<BoxKp2>& rects, float iouThr, int topK)
{
    if (topK == -1)
        topK = rects.size();

    std::sort(rects.begin(), rects.end(),[](BoxKp2 a, BoxKp2 b){return a.score > b.score;});

    std::vector<BoxKp2> new_boxes;
    std::vector<bool> del(rects.size(), false);
    for (size_t i = 0; i < rects.size(); i++)
    {
        if (!del[i])
        {
            for (size_t j = i + 1; j < rects.size(); j++)
            {
                float ou = iou(rects[i], rects[j]);
                if ( ou > iouThr)
                {
                    del[j] = true;
                }
            }
            new_boxes.push_back(rects[i]);

            if (new_boxes.size() > topK)
                break;
        }
    }

    rects = new_boxes;
}

//float OverlapSimilarity(const NMSProcessorOption::OverlapType overlap_type,
//                        const Rect2f & rect1, const Rect2f& rect2)
//{
//    if (!rect1.Intersects(rect2)) return 0.0f;
//    const float intersection_area = Rect2f(rect1).Intersect(rect2).Area();
//    float normalization;
//    switch (overlap_type)
//    {
//        case NMSProcessorOption::JACCARD:
//            normalization = Rect2f(rect1).Union(rect2).Area();
//            break;
//        case NMSProcessorOption::MODIFIED_JACCARD:
//            normalization = rect2.Area();
//            break;
//        case NMSProcessorOption::INTERSECTION_OVER_UNION:
//            normalization = rect1.Area() + rect2.Area() - intersection_area;
//            break;
//        default:
//            LOG(FATAL) << "Unrecognized overlap type: " << overlap_type;
//    }
//    return normalization > 0.0f ? intersection_area / normalization : 0.0f;
//}
//
//// Computes an overlap similarity between two locations by first extracting the
//// relative box (dimension normalized by frame width/height) from the location.
//float OverlapSimilarity(
//        const int frame_width, const int frame_height,
//        const NMSProcessorOption::OverlapType overlap_type,
//        const Location& location1, const Location& location2)
//{
//    const auto rect1 = location1.ConvertToRelativeBBox(frame_width, frame_height);
//    const auto rect2 = location2.ConvertToRelativeBBox(frame_width, frame_height);
//    return OverlapSimilarity(overlap_type, rect1, rect2);
//}

} // namespace