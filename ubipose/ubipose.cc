#include "ubipose.h"

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "keypoints_utils.h"
#include "matching.h"
#include "modules/unprojector.h"
#include "plotting.h"
#include "pose_utils.h"
#include "types.h"

namespace {
bool AcceptResult(const ubipose::UbiposeConfigs &configs, double est_error_R,
                  double est_error_t, size_t num_matches, size_t num_inliers) {
  double inlier_ratio = (static_cast<double>(num_inliers) / num_matches);
  LOG(INFO) << "iniler ratio = " << inlier_ratio
            << " est_error_R = " << est_error_R
            << " est_error_t = " << est_error_t;

  if (inlier_ratio > configs.strong_inlier_ratio &&
      num_matches > static_cast<size_t>(configs.strong_superglue_matches) &&
      est_error_R < configs.strong_error_R &&
      est_error_t < configs.strong_error_t) {
    LOG(INFO) << "localize is strong";
    return true;
  }

  if ((inlier_ratio < configs.weak_inlier_ratio ||
       est_error_R > configs.weak_error_R ||
       est_error_t > configs.weak_error_t)) {
    LOG(INFO) << "localize seem to be failed";
    return false;
  }
  LOG(INFO) << "localize accepted";
  return true;
}
} // namespace

namespace ubipose {

std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunBaselineUbipose(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator,
    ubipose::UbiposeStats *stats) {
  auto localizer_start = std::chrono::steady_clock::now();

  auto projection_matrix = IntrinsicToProjectionMatrix(
      intrinsic, query_image.cols, query_image.rows);

  SuperPointOutput query_image_superpoints;
  cv::Mat gray_query_img;
  cv::cvtColor(query_image, gray_query_img, cv::COLOR_RGBA2GRAY);
  superpoint->Compute(gray_query_img, query_image_superpoints);

  std::vector<Eigen::Vector2d> outputs_2d;
  std::vector<Eigen::Vector3d> outputs_3d;
  std::vector<ubipose::MatchedPoint> output_matched_points;

  auto extrinsics = SampleCameraExtrinsicAround(input_extrinsic);

  int extrinsic_index = 0;
  int num_mesh_features = 0;
  for (const ubipose::EigenGl4f &extrinsic : extrinsics) {
    auto preprocess_start = std::chrono::steady_clock::now();
    auto [rendered_image, depth_image] =
        renderer->Render(extrinsic, projection_matrix);

    auto unprojected_result =
        unprojector->Unproject(depth_image, extrinsic, projection_matrix);

    cv::Mat gray_rendered_image;
    cv::cvtColor(rendered_image, gray_rendered_image, cv::COLOR_RGBA2GRAY);
    SuperPointOutput rendered_frame_superpoints;
    auto preprocess_end = std::chrono::steady_clock::now();
    stats->preprocess_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end -
                                                              preprocess_start)
            .count();

    auto keypoint_start = std::chrono::steady_clock::now();
    superpoint->Compute(gray_rendered_image, rendered_frame_superpoints);
    num_mesh_features += rendered_frame_superpoints.cols();
    auto keypoint_end = std::chrono::steady_clock::now();
    stats->superpoint_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(keypoint_end -
                                                              keypoint_start)
            .count();

    auto matcher_start = std::chrono::steady_clock::now();
    std::vector<cv::DMatch> superglue_matches;
    superglue->MatchPoints(query_image_superpoints, rendered_frame_superpoints,
                           superglue_matches);
    auto matcher_end = std::chrono::steady_clock::now();
    stats->superglue_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(matcher_end -
                                                              matcher_start)
            .count();

    auto postprocess_start = std::chrono::steady_clock::now();

    AddNewSuperglueMatches(query_image_superpoints, rendered_frame_superpoints,
                           superglue_matches, unprojected_result, query_image,
                           &outputs_2d, &outputs_3d, &output_matched_points);

    if (configs.debugging) {
      PlotSuperglueMatches(query_image, rendered_image, timestamp,
                           extrinsic_index, query_image_superpoints,
                           rendered_frame_superpoints, superglue_matches);
    }

    auto postprocess_end = std::chrono::steady_clock::now();
    stats->postprocess_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end -
                                                              postprocess_start)
            .count();

    extrinsic_index++;
  }

  auto register_start = std::chrono::steady_clock::now();

  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  size_t num_inliers = 0;

  std::vector<char> inlier_mask;

  stats->localized = image_registrator->Register(
      outputs_2d, outputs_3d, intrinsic, query_image.cols, query_image.rows,
      &qvec, &tvec, &num_inliers, &inlier_mask);

  if (stats->localized && configs.debugging) {
    RenderLocalizeOutput(renderer, ColmapVec2GlExtrinsic(qvec, tvec),
                         projection_matrix, timestamp, query_image,
                         "localized");
  }

  auto register_end = std::chrono::steady_clock::now();
  auto localizer_end = std::chrono::steady_clock::now();

  stats->num_query_features = query_image_superpoints.cols();
  stats->num_total_mesh_features = num_mesh_features;
  stats->num_matches = outputs_3d.size();
  stats->num_inliers = num_inliers;
  stats->register_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(register_end -
                                                            register_start)
          .count();
  stats->total_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(localizer_end -
                                                            localizer_start)
          .count();
  stats->localized_qvec = qvec;
  stats->localized_tvec = tvec;

  auto [est_error_R, est_error_t] =
      CalculateError(vio_est_qvec, vio_est_tvec, qvec, tvec);
  if (!stats->localized) {
    return {vio_est_qvec, vio_est_tvec};
  }

  // Reaching here means the localizer succeeds
  if (AcceptResult(configs, est_error_R, est_error_t, outputs_3d.size(),
                   num_inliers)) {
    stats->accepted = true;
    return {qvec, tvec};
  }
  return {vio_est_qvec, vio_est_tvec};

  return {qvec, tvec};
}

std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunBaselineUbiposeSingleSG(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator,
    ubipose::UbiposeStats *stats) {
  auto localizer_start = std::chrono::steady_clock::now();

  auto projection_matrix = IntrinsicToProjectionMatrix(
      intrinsic, query_image.cols, query_image.rows);

  SuperPointOutput query_image_superpoints;
  cv::Mat gray_query_img;
  cv::cvtColor(query_image, gray_query_img, cv::COLOR_RGBA2GRAY);
  superpoint->Compute(gray_query_img, query_image_superpoints);

  std::vector<Eigen::Vector2d> outputs_2d;
  std::vector<Eigen::Vector3d> outputs_3d;
  std::vector<ubipose::MatchedPoint> output_matched_points;

  int extrinsic_index = 0;
  int num_mesh_features = 0;
  auto preprocess_start = std::chrono::steady_clock::now();
  auto [rendered_image, depth_image] =
      renderer->Render(input_extrinsic, projection_matrix);

  auto unprojected_result =
      unprojector->Unproject(depth_image, input_extrinsic, projection_matrix);

  cv::Mat gray_rendered_image;
  cv::cvtColor(rendered_image, gray_rendered_image, cv::COLOR_RGBA2GRAY);
  SuperPointOutput rendered_frame_superpoints;
  auto preprocess_end = std::chrono::steady_clock::now();
  stats->preprocess_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end -
                                                            preprocess_start)
          .count();

  auto keypoint_start = std::chrono::steady_clock::now();
  superpoint->Compute(gray_rendered_image, rendered_frame_superpoints);
  num_mesh_features += rendered_frame_superpoints.cols();
  auto keypoint_end = std::chrono::steady_clock::now();
  stats->superpoint_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(keypoint_end -
                                                            keypoint_start)
          .count();

  auto matcher_start = std::chrono::steady_clock::now();
  std::vector<cv::DMatch> superglue_matches;
  superglue->MatchPoints(query_image_superpoints, rendered_frame_superpoints,
                         superglue_matches);
  auto matcher_end = std::chrono::steady_clock::now();
  stats->superglue_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(matcher_end -
                                                            matcher_start)
          .count();

  auto postprocess_start = std::chrono::steady_clock::now();

  AddNewSuperglueMatches(query_image_superpoints, rendered_frame_superpoints,
                         superglue_matches, unprojected_result, query_image,
                         &outputs_2d, &outputs_3d, &output_matched_points);

  if (configs.debugging) {
    PlotSuperglueMatches(query_image, rendered_image, timestamp,
                         extrinsic_index, query_image_superpoints,
                         rendered_frame_superpoints, superglue_matches);
  }

  auto postprocess_end = std::chrono::steady_clock::now();
  stats->postprocess_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end -
                                                            postprocess_start)
          .count();

  extrinsic_index++;

  auto register_start = std::chrono::steady_clock::now();

  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  size_t num_inliers = 0;

  std::vector<char> inlier_mask;

  stats->localized = image_registrator->Register(
      outputs_2d, outputs_3d, intrinsic, query_image.cols, query_image.rows,
      &qvec, &tvec, &num_inliers, &inlier_mask);

  if (stats->localized && configs.debugging) {
    RenderLocalizeOutput(renderer, ColmapVec2GlExtrinsic(qvec, tvec),
                         projection_matrix, timestamp, query_image,
                         "localized");
  }

  auto register_end = std::chrono::steady_clock::now();
  auto localizer_end = std::chrono::steady_clock::now();

  stats->num_query_features = query_image_superpoints.cols();
  stats->num_total_mesh_features = num_mesh_features;
  stats->num_matches = outputs_3d.size();
  stats->num_inliers = num_inliers;
  stats->register_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(register_end -
                                                            register_start)
          .count();
  stats->total_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(localizer_end -
                                                            localizer_start)
          .count();
  stats->localized_qvec = qvec;
  stats->localized_tvec = tvec;

  auto [est_error_R, est_error_t] =
      CalculateError(vio_est_qvec, vio_est_tvec, qvec, tvec);
  if (!stats->localized) {
    return {vio_est_qvec, vio_est_tvec};
  }

  // Reaching here means the localizer succeeds
  if (AcceptResult(configs, est_error_R, est_error_t, outputs_3d.size(),
                   num_inliers)) {
    stats->accepted = true;
    return {qvec, tvec};
  }
  return {vio_est_qvec, vio_est_tvec};

  return {qvec, tvec};
}

std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunFusedUbipose(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator,
    ubipose::UbiposeStats *stats) {
  auto localizer_start = std::chrono::steady_clock::now();

  auto projection_matrix = IntrinsicToProjectionMatrix(
      intrinsic, query_image.cols, query_image.rows);

  SuperPointOutput query_image_superpoints;
  cv::Mat gray_query_img;
  cv::cvtColor(query_image, gray_query_img, cv::COLOR_RGBA2GRAY);
  superpoint->Compute(gray_query_img, query_image_superpoints);

  std::vector<SuperPointOutput> frame_keypoints_vec;
  std::vector<UnprojectHelper> unproject_result_vec;

  auto extrinsics = SampleCameraExtrinsicAround(input_extrinsic);

  int num_mesh_features = 0;
  int extrinsic_index = 0;
  cv::Mat saved_rendered_img;
  for (const ubipose::EigenGl4f &extrinsic : extrinsics) {
    auto preprocess_start = std::chrono::steady_clock::now();
    auto [rendered_image, depth_img] =
        renderer->Render(extrinsic, projection_matrix);

    auto result =
        unprojector->Unproject(depth_img, extrinsic, projection_matrix);
    unproject_result_vec.push_back(result);

    cv::Mat gray_rendered_image;
    if (extrinsic_index == 1) {
      saved_rendered_img = rendered_image;
    }
    cv::cvtColor(rendered_image, gray_rendered_image, cv::COLOR_RGBA2GRAY);
    SuperPointOutput rendered_frame_superpoints;
    auto preprocess_end = std::chrono::steady_clock::now();
    stats->preprocess_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end -
                                                              preprocess_start)
            .count();

    auto keypoint_start = std::chrono::steady_clock::now();
    superpoint->Compute(gray_rendered_image, rendered_frame_superpoints);
    num_mesh_features += rendered_frame_superpoints.cols();
    frame_keypoints_vec.push_back(rendered_frame_superpoints);
    auto keypoint_end = std::chrono::steady_clock::now();
    stats->superpoint_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(keypoint_end -
                                                              keypoint_start)
            .count();

    extrinsic_index++;
  }

  LOG(INFO) << "Total num of features " << num_mesh_features;
  auto [tri_points2D, tri_points3D, matched_points] = ProjectedSuperGlue(
      query_image, saved_rendered_img, vio_est_qvec, vio_est_tvec, intrinsic,
      query_image.cols, query_image.rows, query_image_superpoints,
      frame_keypoints_vec, unproject_result_vec, superglue);

  auto register_start = std::chrono::steady_clock::now();

  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  size_t num_inliers = 0;

  std::vector<char> inlier_mask;

  stats->localized = image_registrator->Register(
      tri_points2D, tri_points3D, intrinsic, query_image.cols, query_image.rows,
      &qvec, &tvec, &num_inliers, &inlier_mask);

  if (configs.debugging && stats->localized) {
    RenderLocalizeOutput(renderer, ColmapVec2GlExtrinsic(qvec, tvec),
                         projection_matrix, timestamp, query_image,
                         "localized");
  }

  auto register_end = std::chrono::steady_clock::now();
  auto localizer_end = std::chrono::steady_clock::now();

  stats->num_query_features = query_image_superpoints.cols();
  stats->num_total_mesh_features = num_mesh_features;
  stats->num_matches = tri_points3D.size();
  stats->num_inliers = num_inliers;
  stats->register_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(register_end -
                                                            register_start)
          .count();
  stats->total_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(localizer_end -
                                                            localizer_start)
          .count();
  stats->localized_qvec = qvec;
  stats->localized_tvec = tvec;

  return {qvec, tvec};
}

struct FastPathOutput {
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  std::vector<ubipose::MapPoint> output_matched_points;
  std::vector<ubipose::MatchedPoint> output_superpoint_flow_matched_points;

  cv::Mat output_rendered_image;
  SuperPointOutput output_initial_frame_superpoints;
  UnprojectHelper output_initial_unproject_helper;
  std::vector<Eigen::Vector2d> output_points_2d;
  std::vector<Eigen::Vector3d> output_points_3d;
  std::unordered_map<size_t, Eigen::Vector3d> merged_index;
};

FastPathOutput RunMeshLocWithAllOptimizationFastPath(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    std::vector<MapPoint *> &projected_map_points,
    std::vector<cv::KeyPoint> &projected_cv_keypoints,
    const ubipose::EigenGl4f &projection_matrix,
    SuperPointOutput &query_image_superpoints,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator,
    ubipose::UbiposeStats *stats) {

  auto localizer_start = std::chrono::steady_clock::now();

  auto extrinsic = input_extrinsic;

  int num_mesh_features = 0;
  auto preprocess_start = std::chrono::steady_clock::now();
  auto [rendered_image, depth_img] =
      renderer->Render(extrinsic, projection_matrix);

  auto initial_unproject_helper =
      unprojector->Unproject(depth_img, extrinsic, projection_matrix);

  cv::Mat gray_rendered_image;
  cv::cvtColor(rendered_image, gray_rendered_image, cv::COLOR_RGBA2GRAY);
  SuperPointOutput initial_rendered_frame_superpoints;
  auto preprocess_end = std::chrono::steady_clock::now();
  stats->cache_localized_preprocess_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end -
                                                            preprocess_start)
          .count();

  auto keypoint_start = std::chrono::steady_clock::now();
  superpoint->Compute(gray_rendered_image, initial_rendered_frame_superpoints);
  num_mesh_features += initial_rendered_frame_superpoints.cols();
  auto keypoint_end = std::chrono::steady_clock::now();
  stats->cache_localized_superpoint_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(keypoint_end -
                                                            keypoint_start)
          .count();

  auto matcher_start = std::chrono::steady_clock::now();
  std::vector<cv::DMatch> superglue_matches;
  superglue->MatchPoints(query_image_superpoints,
                         initial_rendered_frame_superpoints, superglue_matches);
  auto matcher_end = std::chrono::steady_clock::now();
  stats->cache_localized_superglue_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(matcher_end -
                                                            matcher_start)
          .count();
  LOG(INFO) << "Num superglue match in cache localize "
            << superglue_matches.size();

  auto postprocess_start = std::chrono::steady_clock::now();
  if (configs.debugging) {
    std::vector<SuperPointOutput> frame_keypoints_vec;
    std::vector<std::vector<cv::DMatch>> superpoint_matches_vec;
    std::vector<UnprojectHelper> unproject_result_vec;
    auto [tri_points2D, tri_points3D, matched_points, merged_index] =
        SuperpointFlow(projected_map_points, projected_cv_keypoints,
                       query_image, query_image_superpoints,
                       frame_keypoints_vec, superpoint_matches_vec,
                       unproject_result_vec);

    PlotMatches(query_image, gray_rendered_image, timestamp, matched_points);
  }

  // Merge observations
  //  For each image feature point in the cache, try to see if it's still in
  //  this image frame (2 way NN) For each of these found feature points
  //     If there is match for the feature in this frame (from the 1x
  //     superglue), we check how far is the 3D point, if small enough, we
  //     merge, otherwise, we keep the old one If there is no matched mesh
  //     feature in this frame, we add it.
  auto [output_points_2d, output_points_3d, matched_points, merged_index] =
      SuperpointFlow(projected_map_points, projected_cv_keypoints, query_image,
                     query_image_superpoints,
                     initial_rendered_frame_superpoints, superglue_matches,
                     initial_unproject_helper);
  CHECK_EQ(output_points_3d.size(), matched_points.size());
  if (output_points_3d.size() <
      static_cast<size_t>(configs.early_exit_num_matches)) {
    stats->early_exited = true;
    stats->cache_localized_num_query_features = query_image_superpoints.cols();
    stats->cache_localized_num_total_mesh_features = num_mesh_features;
    stats->cache_localized_num_projected_matched_features =
        output_points_3d.size();
    stats->cache_localized_num_inliers = 0;

    stats->cache_localized_total_latency_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - localizer_start)
            .count();

    return FastPathOutput{
        .qvec = vio_est_qvec,
        .tvec = vio_est_tvec,
        .output_matched_points = {},
        .output_superpoint_flow_matched_points = matched_points,
        .output_rendered_image = rendered_image,
        .output_initial_frame_superpoints = initial_rendered_frame_superpoints,
        .output_initial_unproject_helper = initial_unproject_helper,
        .output_points_2d = output_points_2d,
        .output_points_3d = output_points_3d,
        .merged_index = merged_index};
  }

  auto postprocess_end = std::chrono::steady_clock::now();
  stats->cache_localized_match_projection_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end -
                                                            postprocess_start)
          .count();

  if (configs.debugging) {
    PlotSuperglueMatches(query_image, rendered_image, timestamp, 0,
                         query_image_superpoints,
                         initial_rendered_frame_superpoints, superglue_matches);
  }

  auto register_start = std::chrono::steady_clock::now();
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;

  stats->cache_localized = image_registrator->Register(
      output_points_2d, output_points_3d, intrinsic, query_image.cols,
      query_image.rows, &qvec, &tvec, &num_inliers, &inlier_mask);
  std::vector<ubipose::MapPoint> output_matched_points;
  if (stats->cache_localized) {
    output_matched_points = FilterByInlier(matched_points, inlier_mask);

    if (configs.debugging) {
      PlotMatches(query_image, gray_rendered_image, timestamp, matched_points);
      PlotInlierMatches(query_image, gray_rendered_image, timestamp,
                        matched_points, inlier_mask);
      RenderLocalizeOutput(renderer, ColmapVec2GlExtrinsic(qvec, tvec),
                           projection_matrix, timestamp, query_image,
                           "cache_localized");
    }
  }

  auto register_end = std::chrono::steady_clock::now();
  stats->cache_localized_register_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(register_end -
                                                            register_start)
          .count();

  auto localizer_end = std::chrono::steady_clock::now();

  stats->cache_localized_num_query_features = query_image_superpoints.cols();
  stats->cache_localized_num_total_mesh_features = num_mesh_features;
  stats->cache_localized_num_projected_matched_features =
      output_points_3d.size();
  stats->cache_localized_num_inliers = num_inliers;

  stats->cache_localized_total_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(localizer_end -
                                                            localizer_start)
          .count();
  stats->cache_localized_qvec = qvec;
  stats->cache_localized_tvec = tvec;

  // We early exit if we already failed here
  if (!stats->cache_localized) {
    stats->early_exited = true;
    stats->accepted_cache = false;
    return FastPathOutput{
        .qvec = vio_est_qvec,
        .tvec = vio_est_tvec,
        .output_matched_points = {},
        .output_superpoint_flow_matched_points = matched_points,
        .output_rendered_image = rendered_image,
        .output_initial_frame_superpoints = initial_rendered_frame_superpoints,
        .output_initial_unproject_helper = initial_unproject_helper,
        .output_points_2d = output_points_2d,
        .output_points_3d = output_points_3d,
        .merged_index = {}};
  }

  auto [est_error_R, est_error_t] =
      CalculateError(vio_est_qvec, vio_est_tvec, qvec, tvec);
  // If the result is acceptable, we take it
  if (AcceptResult(configs, est_error_R, est_error_t, output_points_3d.size(),
                   num_inliers)) {
    stats->accepted_cache = true;
  }

  return FastPathOutput{
      .qvec = qvec,
      .tvec = tvec,
      .output_matched_points = output_matched_points,
      .output_superpoint_flow_matched_points = matched_points,
      .output_rendered_image = rendered_image,
      .output_initial_frame_superpoints = initial_rendered_frame_superpoints,
      .output_initial_unproject_helper = initial_unproject_helper,
      .output_points_2d = output_points_2d,
      .output_points_3d = output_points_3d,
      .merged_index = merged_index};
  ;
}

std::tuple<Eigen::Vector4d, Eigen::Vector3d, std::vector<ubipose::MapPoint>>
RunUbiposeWithAllOptimization(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    std::vector<MapPoint *> &projected_map_points,
    std::vector<cv::KeyPoint> &projected_cv_keypoints,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator,
    ubipose::UbiposeStats *stats) {

  auto localizer_start = std::chrono::steady_clock::now();
  auto projection_matrix = IntrinsicToProjectionMatrix(
      intrinsic, query_image.cols, query_image.rows);

  SuperPointOutput query_image_superpoints;
  cv::Mat gray_query_img;
  cv::cvtColor(query_image, gray_query_img, cv::COLOR_RGBA2GRAY);
  superpoint->Compute(gray_query_img, query_image_superpoints);

  auto fast_path_output = RunMeshLocWithAllOptimizationFastPath(
      configs, timestamp, query_image, vio_est_qvec, vio_est_tvec,
      input_extrinsic, intrinsic, projected_map_points, projected_cv_keypoints,
      projection_matrix, query_image_superpoints, superpoint, superglue,
      renderer, unprojector, image_registrator, stats);

  if (stats->early_exited || stats->accepted_cache) {
    return {fast_path_output.qvec, fast_path_output.tvec,
            fast_path_output.output_matched_points};
  }

  // Take the longer path, do two more superglue
  std::vector<SuperPointOutput> frame_keypoints_vec;
  std::vector<UnprojectHelper> unproject_result_vec;

  std::vector<Eigen::Vector2d> tri_points2D =
      std::move(fast_path_output.output_points_2d);
  std::vector<Eigen::Vector3d> tri_points3D =
      std::move(fast_path_output.output_points_3d);
  std::vector<ubipose::MatchedPoint> matched_points =
      std::move(fast_path_output.output_superpoint_flow_matched_points);
  std::vector<ubipose::MapPoint> output_matched_points;

  auto extrinsics = SampleCameraExtrinsicAround(input_extrinsic);
  size_t num_mesh_features = 0;

  int extrinsic_index = 0;
  for (const ubipose::EigenGl4f &extrinsic : extrinsics) {
    if (extrinsic_index == 1) {
      frame_keypoints_vec.push_back(
          fast_path_output.output_initial_frame_superpoints);
      unproject_result_vec.push_back(
          fast_path_output.output_initial_unproject_helper);
      num_mesh_features +=
          fast_path_output.output_initial_frame_superpoints.cols();

      extrinsic_index++;
      continue;
    }
    auto preprocess_start = std::chrono::steady_clock::now();
    auto [rendered_image, depth_img] =
        renderer->Render(extrinsic, projection_matrix);

    auto result =
        unprojector->Unproject(depth_img, extrinsic, projection_matrix);
    unproject_result_vec.push_back(result);

    cv::Mat gray_rendered_image;

    cv::cvtColor(rendered_image, gray_rendered_image, cv::COLOR_RGBA2GRAY);
    SuperPointOutput rendered_frame_superpoints;
    auto preprocess_end = std::chrono::steady_clock::now();
    stats->preprocess_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end -
                                                              preprocess_start)
            .count();

    auto keypoint_start = std::chrono::steady_clock::now();
    superpoint->Compute(gray_rendered_image, rendered_frame_superpoints);
    num_mesh_features += rendered_frame_superpoints.cols();
    frame_keypoints_vec.push_back(rendered_frame_superpoints);
    auto keypoint_end = std::chrono::steady_clock::now();
    stats->superpoint_latency_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(keypoint_end -
                                                              keypoint_start)
            .count();

    extrinsic_index++;
  }
  LOG(INFO) << "Total num of features " << num_mesh_features;
  auto superglue_start = std::chrono::steady_clock::now();
  auto [additional_points2D, additional_points3D, additional_matched_points] =
      ProjectedSuperGlueWithStructuralPoints(
          query_image, fast_path_output.output_rendered_image, vio_est_qvec,
          vio_est_tvec, intrinsic, query_image.cols, query_image.rows,
          fast_path_output.merged_index, query_image_superpoints,
          frame_keypoints_vec, unproject_result_vec, superglue);

  // Combine the matched from the first round and the second round
  tri_points2D.insert(tri_points2D.end(), additional_points2D.begin(),
                      additional_points2D.end());
  tri_points3D.insert(tri_points3D.end(), additional_points3D.begin(),
                      additional_points3D.end());
  matched_points.insert(matched_points.end(), additional_matched_points.begin(),
                        additional_matched_points.end());
  auto superglue_end = std::chrono::steady_clock::now();
  stats->superglue_latency_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(superglue_end -
                                                            superglue_start)
          .count();

  auto register_start = std::chrono::steady_clock::now();
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  size_t num_inliers;
  std::vector<char> inlier_mask;

  stats->localized = image_registrator->Register(
      tri_points2D, tri_points3D, intrinsic, query_image.cols, query_image.rows,
      &qvec, &tvec, &num_inliers, &inlier_mask);

  if (stats->localized) {
    output_matched_points = FilterByInlier(matched_points, inlier_mask);
    if (configs.debugging) {
      RenderLocalizeOutput(renderer, ColmapVec2GlExtrinsic(qvec, tvec),
                           projection_matrix, timestamp, query_image,
                           "localized");
    }
  }

  auto register_end = std::chrono::steady_clock::now();
  auto localizer_end = std::chrono::steady_clock::now();

  stats->num_query_features = query_image_superpoints.cols();
  stats->num_total_mesh_features = num_mesh_features;
  stats->num_matches = tri_points3D.size();
  stats->num_inliers = num_inliers;
  stats->register_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(register_end -
                                                            register_start)
          .count();
  stats->total_latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(localizer_end -
                                                            localizer_start)
          .count();
  stats->localized_qvec = qvec;
  stats->localized_tvec = tvec;

  auto [est_error_R, est_error_t] =
      CalculateError(vio_est_qvec, vio_est_tvec, qvec, tvec);
  if (!stats->localized) {
    return {vio_est_qvec, vio_est_tvec, {}};
  }

  // Reaching here means the localizer succeeds
  if (AcceptResult(configs, est_error_R, est_error_t, tri_points3D.size(),
                   num_inliers)) {
    stats->accepted = true;
    return {qvec, tvec, output_matched_points};
  }
  return {vio_est_qvec, vio_est_tvec, {}};
}
} // namespace ubipose
