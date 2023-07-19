#ifndef UBIPOSE_MATCHING_H
#define UBIPOSE_MATCHING_H

#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

#include "modules/superpointglue.h"
#include "modules/unprojector.h"
#include "types.h"

namespace ubipose {

class MapPoint {
public:
  MapPoint(
      const Eigen::Vector3d &point_3d,
      const std::vector<Eigen::Matrix<double, 256, 1>> &cam_image_features,
      const std::vector<Eigen::Matrix<double, 256, 1>> &mesh_image_features)
      : point_3d_(point_3d), cam_image_features_(cam_image_features),
        mesh_image_features_(mesh_image_features) {
    num_observations_ = 1;
    num_visible_ = 1;
  }

  Eigen::Vector3d point_3d() const { return point_3d_; }
  void AddObservation(const Eigen::Matrix<double, 256, 1> &cam_image_feature);
  void AddVisible();
  double ObservedRatio();
  Eigen::Matrix<double, 256, 1> ComputeDistinctiveCameraImageFeature() const;
  Eigen::Matrix<double, 256, 1> ComputeDistinctiveMeshImageFeature() const;

private:
  Eigen::Vector3d point_3d_;
  std::vector<Eigen::Matrix<double, 256, 1>> cam_image_features_;
  std::vector<Eigen::Matrix<double, 256, 1>> mesh_image_features_;
  size_t num_observations_;
  size_t num_visible_;
};

void AddNewSuperglueMatches(
    const SuperPointOutput &query_keypoints,
    const SuperPointOutput &frame_keypoints,
    const std::vector<cv::DMatch> &superglue_matches,
    const ubipose::UnprojectHelper &unprojected_result, cv::Mat query_image,
    std::vector<Eigen::Vector2d> *outputs_2d,
    std::vector<Eigen::Vector3d> *outputs_3d,
    std::vector<ubipose::MatchedPoint> *output_matched_points);

void AddNewSuperglueMatches(
    const std::unordered_map<size_t, Eigen::Vector3d> &merged_index,
    const SuperPointOutput &query_keypoints,
    const std::vector<SuperPointOutput> &frame_keypoints_vec,
    const std::vector<std::vector<cv::DMatch>> &superglue_matches_vec,
    const std::vector<ubipose::UnprojectHelper> &unprojected_result_vec,
    cv::Mat query_image, std::vector<Eigen::Vector2d> *outputs_2d,
    std::vector<Eigen::Vector3d> *outputs_3d,
    std::vector<ubipose::MatchedPoint> *output_matched_points);

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>,
           std::unordered_map<size_t, Eigen::Vector3d>>
SuperpointFlow(
    std::vector<ubipose::MapPoint *> &projected_map_points,
    std::vector<cv::KeyPoint> &projected_cv_keypoints, cv::Mat query_image,
    const SuperPointOutput &query_keypoints,
    const std::vector<SuperPointOutput> &frame_keypoints_vec,
    const std::vector<std::vector<cv::DMatch>> &superglue_matches_vec,
    const std::vector<UnprojectHelper> &unprojected_result_vec);

// Overloaded for single frame input
std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>,
           std::unordered_map<size_t, Eigen::Vector3d>>
SuperpointFlow(std::vector<ubipose::MapPoint *> &projected_map_points,
               std::vector<cv::KeyPoint> &projected_cv_keypoints,
               cv::Mat query_image, const SuperPointOutput &query_keypoints,
               const SuperPointOutput &frame_keypoints,
               const std::vector<cv::DMatch> &superglue_matches,
               const UnprojectHelper &unprojected_result);

std::vector<ubipose::MapPoint>
FilterByInlier(std::vector<ubipose::MatchedPoint> &input,
               const std::vector<char> &inlier_mask);

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>>
ProjectedSuperGlue(
    cv::Mat query_image, cv::Mat rendered_img, const Eigen::Vector4d &qvec,
    const Eigen::Vector3d &tvec,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, ubipose::SuperPointOutput &query_keypoints,
    std::vector<ubipose::SuperPointOutput> &frame_keypoints_vec,
    const std::vector<ubipose::UnprojectHelper> &unprojected_result_vec,
    ubipose::SuperGlueFeatureMatcher *superglue);

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>>
ProjectedSuperGlueWithStructuralPoints(
    cv::Mat query_image, cv::Mat rendered_img, const Eigen::Vector4d &qvec,
    const Eigen::Vector3d &tvec,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, std::unordered_map<size_t, Eigen::Vector3d> merged_index,
    ubipose::SuperPointOutput &query_keypoints,
    std::vector<ubipose::SuperPointOutput> &frame_keypoints_vec,
    const std::vector<ubipose::UnprojectHelper> &unprojected_result_vec,
    ubipose::SuperGlueFeatureMatcher *superglue);

} // namespace ubipose

#endif
