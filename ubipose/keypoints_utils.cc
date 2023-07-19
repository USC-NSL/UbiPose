#include "keypoints_utils.h"

namespace ubipose {

Eigen::Vector2d CvKeyPointToColmapPoint2d(const cv::KeyPoint &keypoint) {
  return Eigen::Vector2d(keypoint.pt.x + 0.5, keypoint.pt.y + 0.5);
}

cv::KeyPoint ColmapPoint2dToCvKeyPoint(const Eigen::Vector2d &point_2d) {
  return cv::KeyPoint(point_2d.x() - 0.5, point_2d.y() - 0.5, 8, -1, 1.0);
}

cv::KeyPoint SuperPointOutputToCvKeyPoint(
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &superpoint_output,
    size_t col_index) {
  return cv::KeyPoint(
      /*x = */ superpoint_output(1, col_index),
      /*y = */ superpoint_output(2, col_index), 8, -1,
      /*score = */ superpoint_output(0, col_index));
}

} // namespace ubipose
