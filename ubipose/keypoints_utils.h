#ifndef UBIPOSE_KEYPOINTS_UTILS_H
#define UBIPOSE_KEYPOINTS_UTILS_H

#include <optional>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

#include "types.h"

namespace ubipose {

Eigen::Vector2d CvKeyPointToColmapPoint2d(const cv::KeyPoint &keypoint);

cv::KeyPoint ColmapPoint2dToCvKeyPoint(const Eigen::Vector2d &point_2d);

cv::KeyPoint SuperPointOutputToCvKeyPoint(
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &superpoint_output,
    size_t col_index);
} // namespace ubipose

#endif
