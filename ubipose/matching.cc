#include "matching.h"

#include <cstddef>
#include <iostream>

#include <Eigen/Dense>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <colmap/base/camera.h>
#include <colmap/base/projection.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/flann/miniflann.hpp>

#include "keypoints_utils.h"
#include "plotting.h"

namespace {

struct hash_pair {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2> &p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);

    if (hash1 != hash2) {
      return hash1 ^ hash2;
    }

    // If hash1 == hash2, their XOR is zero.
    return hash1;
  }
};

} // namespace

namespace ubipose {

void AddNewSuperglueMatches(
    const SuperPointOutput &query_keypoints,
    const SuperPointOutput &frame_keypoints,
    const std::vector<cv::DMatch> &superglue_matches,
    const ubipose::UnprojectHelper &unprojected_result, cv::Mat query_image,
    std::vector<Eigen::Vector2d> *outputs_2d,
    std::vector<Eigen::Vector3d> *outputs_3d,
    std::vector<ubipose::MatchedPoint> *output_matched_points) {
  for (const auto &match : superglue_matches) {
    auto query_index = match.queryIdx;
    auto train_index = match.trainIdx;

    cv::KeyPoint query_keypoint(
        /*x = */ query_keypoints(1, query_index),
        /*y = */ query_keypoints(2, query_index), 8, -1,
        /*score = */ query_keypoints(0, query_index));
    Eigen::Vector2d image_point =
        ubipose::CvKeyPointToColmapPoint2d(query_keypoint);

    cv::KeyPoint frame_keypoint(
        /*x = */ frame_keypoints(1, train_index),
        /*y = */ frame_keypoints(2, train_index), 8, -1,
        /*score = */ frame_keypoints(0, train_index));
    Eigen::Vector3d space_point =
        unprojected_result
            .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
            .cast<double>();

    // Local distance rejection
    if ((image_point - ubipose::CvKeyPointToColmapPoint2d(frame_keypoint))
            .norm() > 20) {
      continue;
    }

    outputs_2d->push_back(image_point);
    outputs_3d->push_back(space_point);
    output_matched_points->push_back(ubipose::MatchedPoint{
        space_point, query_keypoint, frame_keypoint,
        query_keypoints.block<256, 1>(3, query_index),
        frame_keypoints.block<256, 1>(3, train_index), nullptr});
  }
}

void AddNewSuperglueMatches(
    std::unordered_map<size_t, Eigen::Vector3d> &merged_index,
    const SuperPointOutput &query_keypoints,
    const std::vector<SuperPointOutput> &frame_keypoints_vec,
    const std::vector<std::vector<cv::DMatch>> &superglue_matches_vec,
    const std::vector<ubipose::UnprojectHelper> &unprojected_result_vec,
    cv::Mat query_image, std::vector<Eigen::Vector2d> *outputs_2d,
    std::vector<Eigen::Vector3d> *outputs_3d,
    std::vector<ubipose::MatchedPoint> *output_matched_points) {
  size_t num_matches_inter_frame = output_matched_points->size();
  size_t num_superglue_matches = 0;

  for (size_t i = 0; i < superglue_matches_vec.size(); i++) {
    const auto &superglue_matches = superglue_matches_vec[i];
    const auto &frame_keypoints = frame_keypoints_vec[i];
    const auto &unprojected_result = unprojected_result_vec[i];
    num_superglue_matches += superglue_matches.size();
    for (const auto &match : superglue_matches) {
      auto query_index = match.queryIdx;
      auto train_index = match.trainIdx;

      cv::KeyPoint query_keypoint =
          ubipose::SuperPointOutputToCvKeyPoint(query_keypoints, query_index);
      Eigen::Vector2d image_point =
          ubipose::CvKeyPointToColmapPoint2d(query_keypoint);

      cv::KeyPoint frame_keypoint =
          ubipose::SuperPointOutputToCvKeyPoint(frame_keypoints, train_index);
      Eigen::Vector3d space_point =
          unprojected_result
              .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
              .cast<double>();

      // If the camera image feature is found above and the pos is not too
      // different, we skip it here
      const auto it = merged_index.find(query_index);
      if (it != merged_index.end() && (space_point - it->second).norm() < 1.0) {
        continue;
      }

      merged_index[query_index] = space_point;

      outputs_2d->push_back(image_point);
      outputs_3d->push_back(space_point);
      output_matched_points->push_back(ubipose::MatchedPoint{
          space_point, query_keypoint, frame_keypoint,
          query_keypoints.block<256, 1>(3, query_index),
          frame_keypoints.block<256, 1>(3, train_index), nullptr});
    }
  }
  LOG(INFO) << "SuperGlue found "
            << (output_matched_points->size() - num_matches_inter_frame)
            << " new matches out of " << num_superglue_matches << " matches";
}

void MapPoint::AddObservation(
    const Eigen::Matrix<double, 256, 1> &cam_image_feature) {
  cam_image_features_.push_back(cam_image_feature);
  num_observations_++;
}

void MapPoint::AddVisible() { num_visible_++; }

double MapPoint::ObservedRatio() {
  return static_cast<double>(num_observations_) / num_visible_;
}

Eigen::Matrix<double, 256, 1>
MapPoint::ComputeDistinctiveCameraImageFeature() const {
  CHECK(!cam_image_features_.empty());
  if (cam_image_features_.size() == 1) {
    return cam_image_features_[0];
  }
  std::vector<std::vector<double>> distances(
      cam_image_features_.size(),
      std::vector<double>(cam_image_features_.size(), 0.0));
  for (size_t i = 0; i < cam_image_features_.size(); i++) {
    for (size_t j = i + 1; j < cam_image_features_.size(); j++) {
      double l2_norm = cam_image_features_[i].dot(cam_image_features_[j]);
      distances[i][j] = l2_norm;
      distances[j][i] = l2_norm;
    }
  }

  double best_median = 10000000;
  double best_index = 0;
  for (size_t i = 0; i < cam_image_features_.size(); i++) {
    std::sort(distances[i].begin(), distances[i].end());
    double median = distances[i][0.5 * (cam_image_features_.size() - 1)];
    if (median < best_median) {
      best_median = median;
      best_index = i;
    }
  }
  return cam_image_features_[best_index];
}

Eigen::Matrix<double, 256, 1>
MapPoint::ComputeDistinctiveMeshImageFeature() const {
  CHECK(!mesh_image_features_.empty());
  if (mesh_image_features_.size() == 1) {
    return mesh_image_features_[0];
  }
  std::vector<std::vector<double>> distances(
      mesh_image_features_.size(),
      std::vector<double>(mesh_image_features_.size(), 0.0));
  for (size_t i = 0; i < mesh_image_features_.size(); i++) {
    for (size_t j = i + 1; j < mesh_image_features_.size(); j++) {
      double l2_norm = mesh_image_features_[i].dot(mesh_image_features_[j]);
      distances[i][j] = l2_norm;
      distances[j][i] = l2_norm;
    }
  }

  double best_median = 10000000;
  double best_index = 0;
  for (size_t i = 0; i < mesh_image_features_.size(); i++) {
    std::sort(distances[i].begin(), distances[i].end());
    double median = distances[i][0.5 * (mesh_image_features_.size() - 1)];
    if (median < best_median) {
      best_median = median;
      best_index = i;
    }
  }
  return mesh_image_features_[best_index];
}

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>,
           std::unordered_map<size_t, Eigen::Vector3d>>
SuperpointFlow(
    std::vector<ubipose::MapPoint *> &projected_map_points,
    std::vector<cv::KeyPoint> &projected_cv_keypoints, cv::Mat query_image,
    const SuperPointOutput &query_keypoints,
    const std::vector<SuperPointOutput> &frame_keypoints_vec,
    const std::vector<std::vector<cv::DMatch>> &superglue_matches_vec,
    const std::vector<UnprojectHelper> &unprojected_result_vec) {
  std::vector<Eigen::Vector2d> outputs_2d;
  std::vector<Eigen::Vector3d> outputs_3d;
  std::vector<ubipose::MatchedPoint> output_matched_points;
  std::unordered_map<size_t, Eigen::Vector3d> merged_index;
  if (projected_map_points.empty()) {
    LOG(INFO) << "projection map point empty";
    AddNewSuperglueMatches(merged_index, query_keypoints, frame_keypoints_vec,
                           superglue_matches_vec, unprojected_result_vec,
                           query_image, &outputs_2d, &outputs_3d,
                           &output_matched_points);
    return {outputs_2d, outputs_3d, output_matched_points, merged_index};
  }

  Eigen::Matrix<float, 256, Eigen::Dynamic> projected_keypoint_features(
      256, projected_map_points.size());
  for (size_t i = 0; i < projected_map_points.size(); i++) {
    projected_keypoint_features.block<256, 1>(0, i) =
        projected_map_points[i]
            ->ComputeDistinctiveCameraImageFeature()
            .cast<float>();
  }
  Eigen::Matrix<float, 256, Eigen::Dynamic> query_keypoint_features =
      query_keypoints.bottomRows<256>().cast<float>();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> product =
      projected_keypoint_features.transpose() * query_keypoint_features;
  product = product.cwiseMin(1).cwiseMax(-1);
  product = (product.array() * (-2) + 2.0f).sqrt();

  std::vector<double> reproj_errors;

  for (Eigen::Index i = 0; i < product.rows(); i++) {
    Eigen::Index best_index;
    float best_dist = product.row(i).minCoeff(&best_index);
    if (best_dist > 0.7) {
      continue;
    }

    cv::KeyPoint query_keypoint(/*x = */ query_keypoints(1, best_index),
                                /*y = */ query_keypoints(2, best_index), 8, -1,
                                /*score = */ query_keypoints(0, best_index));
    Eigen::Vector2d image_point = CvKeyPointToColmapPoint2d(query_keypoint);

    double reproj_error =
        (image_point - CvKeyPointToColmapPoint2d(projected_cv_keypoints[i]))
            .norm();

    outputs_2d.push_back(image_point);
    outputs_3d.push_back(projected_map_points[i]->point_3d());
    output_matched_points.push_back(ubipose::MatchedPoint{
        projected_map_points[i]->point_3d(), query_keypoint,
        projected_cv_keypoints[i], query_keypoints.block<256, 1>(3, best_index),
        projected_map_points[i]->ComputeDistinctiveMeshImageFeature(),
        projected_map_points[i]});
    merged_index.insert(
        std::make_pair(best_index, projected_map_points[i]->point_3d()));
    reproj_errors.push_back(reproj_error);
  }
  double median_reproj_error = -1;
  if (!reproj_errors.empty()) {
    std::sort(reproj_errors.begin(), reproj_errors.end());
    median_reproj_error = reproj_errors[reproj_errors.size() / 2];
  }
  LOG(INFO) << "Inter frame found " << output_matched_points.size()
            << " matches. Med reproj error  " << median_reproj_error
            << " pixels";

  AddNewSuperglueMatches(merged_index, query_keypoints, frame_keypoints_vec,
                         superglue_matches_vec, unprojected_result_vec,
                         query_image, &outputs_2d, &outputs_3d,
                         &output_matched_points);

  return {outputs_2d, outputs_3d, output_matched_points, merged_index};
}

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>,
           std::unordered_map<size_t, Eigen::Vector3d>>
SuperpointFlow(std::vector<ubipose::MapPoint *> &projected_map_points,
               std::vector<cv::KeyPoint> &projected_cv_keypoints,
               cv::Mat query_image, const SuperPointOutput &query_keypoints,
               const SuperPointOutput &frame_keypoints,
               const std::vector<cv::DMatch> &superglue_matches,
               const UnprojectHelper &unprojected_result) {
  std::vector<SuperPointOutput> frame_keypoints_vec = {frame_keypoints};
  std::vector<std::vector<cv::DMatch>> superpoint_matches_vec = {
      superglue_matches};
  std::vector<UnprojectHelper> unproject_result_vec = {unprojected_result};
  return SuperpointFlow(projected_map_points, projected_cv_keypoints,
                        query_image, query_keypoints, frame_keypoints_vec,
                        superpoint_matches_vec, unproject_result_vec);
}

std::vector<ubipose::MapPoint>
FilterByInlier(std::vector<ubipose::MatchedPoint> &input,
               const std::vector<char> &inlier_mask) {
  if (input.size() != inlier_mask.size()) {
    LOG(FATAL) << "Size different between input and mask";
  }
  std::vector<ubipose::MapPoint> output;
  output.reserve(input.size());

  std::unordered_map<std::pair<float, float>,
                     std::vector<ubipose::MatchedPoint>, hash_pair>
      grouped_observations;
  for (size_t i = 0; i < input.size(); i++) {
    if (inlier_mask[i]) {
      grouped_observations[std::make_pair(input[i].keypoint.pt.x,
                                          input[i].keypoint.pt.y)]
          .push_back(input[i]);
    }
  }
  // For each inlied cam image key point, we have a group of points in 3D
  // Some of the points are fresh (just discover by superglue)
  // Some of the points are from projection
  for (auto &[kp, match_points] : grouped_observations) {
    Eigen::Vector3d mean_point_3d(0, 0, 0);
    std::vector<Eigen::Matrix<double, 256, 1>> mesh_image_features;

    for (const auto &match_point : match_points) {
      mean_point_3d += match_point.point_3d;
    }
    mean_point_3d = mean_point_3d / match_points.size();
    double diff = 0;
    for (const auto &match_point : match_points) {
      diff += (mean_point_3d - match_point.point_3d).norm();
      mesh_image_features.push_back(match_point.mesh_image_feature);
    }
    diff = diff / match_points.size();
    // If the mean diff is small enough, we merge the feature points
    if (diff < 1.0) {
      // If there are matched point from map point, we add observation (we don't
      // care if there are multiple map point)
      bool has_map_point = false;
      for (auto &match_point : match_points) {
        if (match_point.map_point_ != nullptr) {
          match_point.map_point_->AddObservation(match_point.cam_image_feature);
          has_map_point = true;
        }
      }
      // If all matched points are newly discovered, add to output
      if (!has_map_point) {
        output.push_back(ubipose::MapPoint(mean_point_3d,
                                           {match_points[0].cam_image_feature},
                                           mesh_image_features));
      }
    } else {
      // If the mean diff is too large, we simply treat them as individual
      // observation and output
      for (const auto &match_point : match_points) {
        if (match_point.map_point_ != nullptr) {
          match_point.map_point_->AddObservation(match_point.cam_image_feature);
        } else {
          output.push_back(ubipose::MapPoint(match_point.point_3d,
                                             {match_point.cam_image_feature},
                                             {match_point.mesh_image_feature}));
        }
      }
    }
  }
  return output;
}

std::tuple<std::vector<cv::Point2f>, std::vector<int>,
           std::unordered_map<int, int>>
FilterPointsByDistance(const std::vector<cv::Point2f> &points,
                       const std::vector<cv::Point2f> &existingPoints,
                       float minDist) {
  std::vector<cv::Point2f> filteredPoints;
  std::vector<int> keptIndices;
  std::unordered_map<int, int> merged_index;

  // Construct the KDTree on existing points
  cv::flann::KDTreeIndexParams indexParams;
  auto training_data =
      cv::Mat(existingPoints).reshape(1, existingPoints.size());
  cv::flann::Index kdtree(training_data, indexParams);

  // Filter out points that are too close to existing points
  for (size_t i = 0; i < points.size(); i++) {
    const cv::Point2f &p = points[i];

    // Find the distance to the nearest neighbor in the existing set
    std::vector<int> indices(1);
    std::vector<float> dists(1);
    auto query = cv::Mat(p).reshape(1, 1);
    kdtree.knnSearch(query, indices, dists, 1);

    // Add the point to the filtered set if it is not too close to any existing
    // point
    if (dists[0] > minDist) {
      filteredPoints.push_back(p);
      keptIndices.push_back(i);
    } else {
      merged_index[i] = indices[0];
    }
  }

  return std::make_tuple(filteredPoints, keptIndices, merged_index);
}

ubipose::SuperPointOutput ExtractAndConcateSuperPointsByIndices(
    const ubipose::SuperPointOutput &matrix, const std::vector<int> &indices,
    const ubipose::SuperPointOutput &other_matrix) {
  ubipose::SuperPointOutput result(259, indices.size() + other_matrix.cols());

  // Concatenate the other matrix to the result
  result.leftCols(other_matrix.cols()) = other_matrix;
  // Copy the columns indicated by the indices
  for (size_t i = 0; i < indices.size(); i++) {
    result.block(0, other_matrix.cols() + i, matrix.rows(), 1) =
        matrix.col(indices[i]);
  }

  return result;
}

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector3d>,
           std::vector<ubipose::MatchedPoint>>
ProjectedSuperGlue(
    cv::Mat query_image, cv::Mat rendered_img, const Eigen::Vector4d &qvec,
    const Eigen::Vector3d &tvec,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, ubipose::SuperPointOutput &query_keypoints,
    std::vector<ubipose::SuperPointOutput> &frame_keypoints_vec,
    const std::vector<ubipose::UnprojectHelper> &unprojected_result_vec,
    ubipose::SuperGlueFeatureMatcher *superglue) {
  CHECK(frame_keypoints_vec.size() == 3);
  colmap::Camera camera;
  camera.SetModelIdFromName("PINHOLE");
  camera.SetWidth(width);
  camera.SetHeight(height);
  camera.SetParams(
      {intrinsic(0, 0), intrinsic(1, 1), intrinsic(0, 2), intrinsic(1, 2)});
  auto projection_matrix = colmap::ComposeProjectionMatrix(qvec, tvec);

  const auto &initial_frame_keypoints = frame_keypoints_vec[1];

  std::vector<cv::Point2f> kept_kpts;

  // Index to 3d points map
  std::unordered_map<int, std::vector<Eigen::Vector3d>> points_3d;

  // We blindly take the first frame result
  // Loop through all the keypoints
  //   record the pixel location
  //   save the 3d position
  for (Eigen::Index i = 0; i < initial_frame_keypoints.cols(); i++) {
    cv::KeyPoint frame_keypoint =
        ubipose::SuperPointOutputToCvKeyPoint(initial_frame_keypoints, i);

    Eigen::Vector3d space_point =
        unprojected_result_vec[1]
            .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
            .cast<double>();

    kept_kpts.push_back(frame_keypoint.pt);
    points_3d[i] = {space_point};
  }

  ubipose::SuperPointOutput combined_feature_points = initial_frame_keypoints;
  LOG(INFO) << "Base frame num keypoints " << initial_frame_keypoints.cols();

  for (size_t i = 0; i < frame_keypoints_vec.size(); i++) {
    if (i == 1) {
      continue;
    }

    const auto &frame_keypoints = frame_keypoints_vec[i];

    // 2D feature point locations of all keypoints in this render image
    std::vector<cv::Point2f> new_keypoints;
    new_keypoints.reserve(frame_keypoints_vec[i].cols());

    // 3D position of all keypoints in this render image
    std::vector<Eigen::Vector3d> new_keypoints_3d;
    new_keypoints_3d.reserve(frame_keypoints_vec[i].cols());

    // Need to change the x and y to the projected pixel location
    ubipose::SuperPointOutput projected_feature_points;
    projected_feature_points.resize(259, frame_keypoints.cols());

    size_t num_in_pose = 0;
    for (Eigen::Index j = 0; j < frame_keypoints.cols(); j++) {
      cv::KeyPoint frame_keypoint =
          ubipose::SuperPointOutputToCvKeyPoint(frame_keypoints, j);
      Eigen::Vector3d space_point =
          unprojected_result_vec[i]
              .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
              .cast<double>();

      // Project to the virtual image plane
      auto eigen2d =
          colmap::ProjectPointToImage(space_point, projection_matrix, camera);
      cv::KeyPoint projected_frame_keypoint =
          ubipose::ColmapPoint2dToCvKeyPoint(eigen2d);

      if (projected_frame_keypoint.pt.x < 0 ||
          projected_frame_keypoint.pt.y < 0 ||
          projected_frame_keypoint.pt.x >= width ||
          projected_frame_keypoint.pt.y >= height) {
        continue;
      }
      projected_feature_points.col(num_in_pose) = frame_keypoints.col(j);
      projected_feature_points(1, num_in_pose) = projected_frame_keypoint.pt.x;
      projected_feature_points(2, num_in_pose) = projected_frame_keypoint.pt.y;

      new_keypoints.push_back(projected_frame_keypoint.pt);
      new_keypoints_3d.push_back(space_point);

      num_in_pose++;
    }

    LOG(INFO) << "before filtering new_keypoints size " << new_keypoints.size();

    auto [filtered, indices, merged_index] =
        FilterPointsByDistance(new_keypoints, kept_kpts, 25);
    kept_kpts.insert(kept_kpts.end(), filtered.begin(), filtered.end());
    LOG(INFO) << "After adding new kps #total = " << kept_kpts.size()
              << " keypoints";
    combined_feature_points = ExtractAndConcateSuperPointsByIndices(
        frame_keypoints, indices, combined_feature_points);

    // For new points, just add to the points_3d
    for (auto index : indices) {
      points_3d[points_3d.size()] = {new_keypoints_3d[index]};
    }

    // For merged points, we still add the 3D position so localizer can have
    // more point to localize
    for (const auto &[new_keypoints_index, existed_keypoints_index] :
         merged_index) {
      points_3d[existed_keypoints_index].push_back(
          new_keypoints_3d[new_keypoints_index]);
    }
  }

  std::vector<cv::KeyPoint> cv_keypoints;
  for (const auto &pt : kept_kpts) {
    cv_keypoints.push_back(cv::KeyPoint(pt, 8, 01, 1));
  }

  CHECK(kept_kpts.size() ==
        static_cast<size_t>(combined_feature_points.cols()));
  LOG(INFO) << "Number of features " << combined_feature_points.cols();

  // At this point, we have the combined feature points, and their corresponding
  // 3d points
  CHECK(static_cast<size_t>(combined_feature_points.cols()) ==
        points_3d.size());
  std::vector<cv::DMatch> superglue_matches;
  superglue->MatchPoints(combined_feature_points, query_keypoints,
                         superglue_matches);

  std::vector<Eigen::Vector2d> output_points_2d;
  std::vector<Eigen::Vector3d> output_points_3d;
  std::vector<ubipose::MatchedPoint> output_matched_points;

  for (const auto &match : superglue_matches) {
    auto train_index = match.queryIdx;
    auto query_index = match.trainIdx;
    cv::KeyPoint query_keypoint =
        ubipose::SuperPointOutputToCvKeyPoint(query_keypoints, query_index);
    Eigen::Vector2d image_point =
        ubipose::CvKeyPointToColmapPoint2d(query_keypoint);
    cv::KeyPoint frame_keypoint = ubipose::SuperPointOutputToCvKeyPoint(
        combined_feature_points, train_index);
    for (const auto &space_point : points_3d[train_index]) {

      output_points_2d.push_back(image_point);
      output_points_3d.push_back(space_point);
      output_matched_points.push_back(ubipose::MatchedPoint{
          space_point, query_keypoint, frame_keypoint,
          query_keypoints.block<256, 1>(3, query_index),
          combined_feature_points.block<256, 1>(3, train_index), nullptr});
    }
  }
  LOG(INFO) << "Num output point 3d " << output_points_3d.size();
  // PlotMatches(query_image, rendered_img, 0, output_matched_points);
  return {output_points_2d, output_points_3d, output_matched_points};
}

// Find additional 2D-3D matching by running a sg on 3x sp
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
    ubipose::SuperGlueFeatureMatcher *superglue) {
  CHECK(frame_keypoints_vec.size() == 3);
  colmap::Camera camera;
  camera.SetModelIdFromName("PINHOLE");
  camera.SetWidth(width);
  camera.SetHeight(height);
  camera.SetParams(
      {intrinsic(0, 0), intrinsic(1, 1), intrinsic(0, 2), intrinsic(1, 2)});
  auto projection_matrix = colmap::ComposeProjectionMatrix(qvec, tvec);

  const auto &initial_frame_keypoints = frame_keypoints_vec[1];

  std::vector<cv::Point2f> kept_kpts;

  // Index to 3d points map
  std::unordered_map<int, std::vector<Eigen::Vector3d>> points_3d;

  // We blindly take the first frame result
  // Loop through all the keypoints
  //   record the pixel location
  //   save the 3d position
  for (Eigen::Index i = 0; i < initial_frame_keypoints.cols(); i++) {
    cv::KeyPoint frame_keypoint =
        ubipose::SuperPointOutputToCvKeyPoint(initial_frame_keypoints, i);

    Eigen::Vector3d space_point =
        unprojected_result_vec[1]
            .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
            .cast<double>();

    kept_kpts.push_back(frame_keypoint.pt);
    points_3d[i] = {space_point};
  }

  ubipose::SuperPointOutput combined_feature_points = initial_frame_keypoints;
  LOG(INFO) << "Base frame num keypoints " << initial_frame_keypoints.cols();

  for (size_t i = 0; i < frame_keypoints_vec.size(); i++) {
    if (i == 1) {
      continue;
    }

    const auto &frame_keypoints = frame_keypoints_vec[i];

    // 2D feature point locations of all keypoints in this render image
    std::vector<cv::Point2f> new_keypoints;
    new_keypoints.reserve(frame_keypoints_vec[i].cols());

    // 3D position of all keypoints in this render image
    std::vector<Eigen::Vector3d> new_keypoints_3d;
    new_keypoints_3d.reserve(frame_keypoints_vec[i].cols());

    // Need to change the x and y to the projected pixel location
    ubipose::SuperPointOutput projected_feature_points;
    projected_feature_points.resize(259, frame_keypoints.cols());

    size_t num_in_pose = 0;
    for (Eigen::Index j = 0; j < frame_keypoints.cols(); j++) {
      cv::KeyPoint frame_keypoint =
          ubipose::SuperPointOutputToCvKeyPoint(frame_keypoints, j);
      Eigen::Vector3d space_point =
          unprojected_result_vec[i]
              .Get3dPointAtPixel(frame_keypoint.pt.x, frame_keypoint.pt.y)
              .cast<double>();

      // Project to the virtual image plane
      auto eigen2d =
          colmap::ProjectPointToImage(space_point, projection_matrix, camera);
      cv::KeyPoint projected_frame_keypoint =
          ubipose::ColmapPoint2dToCvKeyPoint(eigen2d);

      if (projected_frame_keypoint.pt.x < 0 ||
          projected_frame_keypoint.pt.y < 0 ||
          projected_frame_keypoint.pt.x >= width ||
          projected_frame_keypoint.pt.y >= height) {
        continue;
      }
      projected_feature_points.col(num_in_pose) = frame_keypoints.col(j);
      projected_feature_points(1, num_in_pose) = projected_frame_keypoint.pt.x;
      projected_feature_points(2, num_in_pose) = projected_frame_keypoint.pt.y;

      new_keypoints.push_back(projected_frame_keypoint.pt);
      new_keypoints_3d.push_back(space_point);

      num_in_pose++;
    }

    LOG(INFO) << "before filtering new_keypoints size " << new_keypoints.size();

    auto [filtered, indices, merged_index] =
        FilterPointsByDistance(new_keypoints, kept_kpts, 25);
    kept_kpts.insert(kept_kpts.end(), filtered.begin(), filtered.end());
    LOG(INFO) << "After adding new kps #total = " << kept_kpts.size()
              << " keypoints";
    combined_feature_points = ExtractAndConcateSuperPointsByIndices(
        frame_keypoints, indices, combined_feature_points);

    // For new points, just add to the points_3d
    for (auto index : indices) {
      points_3d[points_3d.size()] = {new_keypoints_3d[index]};
    }

    // For merged points, we still add the 3D position so localizer can have
    // more point to localize
    for (const auto &[new_keypoints_index, existed_keypoints_index] :
         merged_index) {
      points_3d[existed_keypoints_index].push_back(
          new_keypoints_3d[new_keypoints_index]);
    }
  }

  std::vector<cv::KeyPoint> cv_keypoints;
  for (const auto &pt : kept_kpts) {
    cv_keypoints.push_back(cv::KeyPoint(pt, 8, 01, 1));
  }

  CHECK(kept_kpts.size() ==
        static_cast<size_t>(combined_feature_points.cols()));
  LOG(INFO) << "Number of features " << combined_feature_points.cols();

  // At this point, we have the combined feature points, and their corresponding
  // 3d points
  CHECK(static_cast<size_t>(combined_feature_points.cols()) ==
        points_3d.size());
  std::vector<cv::DMatch> superglue_matches;
  auto matcher_start = std::chrono::steady_clock::now();
  superglue->MatchPoints(combined_feature_points, query_keypoints,
                         superglue_matches);
  auto matcher_end = std::chrono::steady_clock::now();
  LOG(INFO) << "Fused Superglue latency "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   matcher_end - matcher_start)
                   .count()
            << "ms";

  std::vector<Eigen::Vector2d> output_points_2d;
  std::vector<Eigen::Vector3d> output_points_3d;
  std::vector<ubipose::MatchedPoint> output_matched_points;

  for (const auto &match : superglue_matches) {
    auto train_index = match.queryIdx;
    auto query_index = match.trainIdx;

    // auto query_index = match.queryIdx;
    // auto train_index = match.trainIdx;
    cv::KeyPoint query_keypoint =
        ubipose::SuperPointOutputToCvKeyPoint(query_keypoints, query_index);
    Eigen::Vector2d image_point =
        ubipose::CvKeyPointToColmapPoint2d(query_keypoint);
    cv::KeyPoint frame_keypoint = ubipose::SuperPointOutputToCvKeyPoint(
        combined_feature_points, train_index);

    // If the camera image feature is found in the cache projection and the pos
    // is not too different, we skip it here
    const auto it = merged_index.find(query_index);

    for (const auto &space_point : points_3d[train_index]) {
      if (it != merged_index.end() && (space_point - it->second).norm() < 1.0) {
        continue;
      }

      output_points_2d.push_back(image_point);
      output_points_3d.push_back(space_point);
      output_matched_points.push_back(ubipose::MatchedPoint{
          space_point, query_keypoint, frame_keypoint,
          query_keypoints.block<256, 1>(3, query_index),
          combined_feature_points.block<256, 1>(3, train_index), nullptr});
    }
  }
  LOG(INFO) << "Num output point 3d " << output_points_3d.size();
  // PlotMatches(query_image, rendered_img, 0, output_matched_points);
  return {output_points_2d, output_points_3d, output_matched_points};
}
} // namespace ubipose
