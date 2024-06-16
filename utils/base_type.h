//
// Created by mzh on 2023/8/22.
// This file contains base data type
//

#ifndef MPP_REPRODUCE_BASE_DATA_H
#define MPP_REPRODUCE_BASE_DATA_H

#include "opencv2/opencv.hpp"

// Box with 2D key points.
struct BoxKp2
{
    cv::Rect2f rect;
    float radians;
    float score;
    int classId;
    std::vector<cv::Point2f> points;
};

// Box with 3D key points.
struct BoxKp3
{
    cv::Rect2f rect;
    float radians;
    float score;
    std::vector<cv::Point3f> points; //x,y,z
    std::vector<cv::Point2f> vis_pre; // optional, visibility, and presence. Only Pose_landmark model support this part.
};


#endif //MPP_REPRODUCE_BASE_DATA_H
