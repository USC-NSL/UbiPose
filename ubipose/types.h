#ifndef UBIPOSE_TYPES_H
#define UBIPOSE_TYPES_H

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>

namespace ubipose {

using EigenGl4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
typedef Eigen::Matrix<double, 259, Eigen::Dynamic> SuperPointOutput;

struct UbiposeStats {
  Eigen::Vector4d vio_est_qvec;
  Eigen::Vector3d vio_est_tvec;
  int num_projected_in_pose = 0;

  // Full localization
  int num_query_features = 0;
  int num_total_mesh_features = 0;
  int num_matches = 0;
  int num_inliers = 0;

  int preprocess_latency_ms = 0;
  int superpoint_latency_ms = 0;
  int superglue_latency_ms = 0;
  int postprocess_latency_ms = 0;
  int match_projection_latency_ms = 0;
  int register_latency_ms = 0;
  int total_latency_ms = 0;

  Eigen::Vector4d localized_qvec;
  Eigen::Vector3d localized_tvec;

  // Cache map points
  bool cache_localized = false;
  int cache_localized_num_query_features = 0;
  int cache_localized_num_total_mesh_features = 0;
  int cache_localized_num_projected_matched_features = 0;
  int cache_localized_num_inliers = 0;

  int cache_localized_preprocess_latency_ms = 0;
  int cache_localized_superpoint_latency_ms = 0;
  int cache_localized_superglue_latency_ms = 0;
  int cache_localized_match_projection_latency_ms = 0;
  int cache_localized_register_latency_ms = 0;
  int cache_localized_total_latency_ms = 0;

  Eigen::Vector4d cache_localized_qvec;
  Eigen::Vector3d cache_localized_tvec;

  bool early_exited = false;
  bool localized = false;
  bool accepted = false;
  bool accepted_cache = false;
};

class MapPoint;

struct MatchedPoint {
  Eigen::Vector3d point_3d;
  // keypoint on the cam image
  cv::KeyPoint keypoint;
  // keypoint on the mesh image
  cv::KeyPoint mesh_keypoint;
  Eigen::Matrix<double, 256, 1> cam_image_feature;
  Eigen::Matrix<double, 256, 1> mesh_image_feature;
  MapPoint *map_point_ = nullptr;
};
} // namespace ubipose

#endif
