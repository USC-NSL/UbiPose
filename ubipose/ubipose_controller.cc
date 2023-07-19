#include "ubipose_controller.h"

#include <cstddef>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <boost/filesystem.hpp>
#include <colmap/base/camera.h>
#include <colmap/base/projection.h>
#include <colmap/util/types.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "colmap/colmap_reconstruction.h"
#include "colmap/colmap_utils.h"
#include "initial_pose_provider.h"
#include "matching.h"
#include "plotting.h"
#include "ubipose.h"

namespace {
bool ValidateUbiposeConfigs(const ubipose::UbiposeConfigs &configs) {
  if (configs.output_localization_path.empty()) {
    return false;
  }

  if (configs.output_images && configs.output_images_folder.empty()) {
    return false;
  }

  return true;
}

void OutputStats(const ubipose::UbiposeStats &stats,
                 const std::string &filename, std::ofstream &ofile) {
  // clang-format off
    ofile << filename << " " 
        << stats.vio_est_qvec[0] << " " 
        << stats.vio_est_qvec[1] << " " 
        << stats.vio_est_qvec[2] << " " 
        << stats.vio_est_qvec[3] << " "

        << stats.vio_est_tvec[0] << " " 
        << stats.vio_est_tvec[1] << " "
        << stats.vio_est_tvec[2] << " " 

        << stats.num_projected_in_pose << " "
        << stats.num_query_features << " " 
        << stats.num_total_mesh_features << " " 
        << stats.num_matches << " " 
        << stats.num_inliers << " "

        << stats.preprocess_latency_ms << " " 
        << stats.superpoint_latency_ms << " " 
        << stats.superglue_latency_ms << " "
        << stats.postprocess_latency_ms << " "
        << stats.match_projection_latency_ms << " " 
        << stats.register_latency_ms << " " 
        << stats.total_latency_ms << " " 

        << stats.localized_qvec[0] << " " 
        << stats.localized_qvec[1] << " " 
        << stats.localized_qvec[2] << " " 
        << stats.localized_qvec[3] << " " 

        << stats.localized_tvec[0] << " " 
        << stats.localized_tvec[1] << " " 
        << stats.localized_tvec[2] << " "

        << stats.cache_localized << " "
        << stats.cache_localized_num_query_features << " "
        << stats.cache_localized_num_total_mesh_features << " "
        << stats.cache_localized_num_projected_matched_features << " "
        << stats.cache_localized_num_inliers << " "

        << stats.cache_localized_preprocess_latency_ms << " "
        << stats.cache_localized_superpoint_latency_ms << " "
        << stats.cache_localized_superglue_latency_ms << " "
        << stats.cache_localized_match_projection_latency_ms << " "
        << stats.cache_localized_register_latency_ms << " "
        << stats.cache_localized_total_latency_ms << " "

        << stats.cache_localized_qvec[0] << " " 
        << stats.cache_localized_qvec[1] << " " 
        << stats.cache_localized_qvec[2] << " " 
        << stats.cache_localized_qvec[3] << " " 

        << stats.cache_localized_tvec[0] << " " 
        << stats.cache_localized_tvec[1] << " " 
        << stats.cache_localized_tvec[2] << " "

        << stats.early_exited << " "
        << stats.localized << " "
        << stats.accepted << " "
        << stats.accepted_cache << std::endl;

  // clang-format on
}

} // namespace

namespace ubipose {

UbiposeController::UbiposeController(const UbiposeConfigs &configs)
    : configs_(configs) {}

UbiposeController::~UbiposeController() {
  if (output_thread) {
    output_thread->join();
    delete output_thread;
  }
}

bool UbiposeController::Init() {
  if (!ValidateUbiposeConfigs(configs_)) {
    return false;
  }

  colmap_reconstruction_ = std::make_unique<ColmapReconstruction>();
  colmap_reconstruction_->Read(configs_.reconstruction_path);
  colmap_reconstruction_->ParseColmapImageTimestamps(
      configs_.colmap_image_prefix);
  auto transforms =
      ubipose::ReadColmapToMeshTransform(configs_.transform_to_mesh_path);
  if (transforms.size() != 1) {
    LOG(FATAL) << "Invalid transform read from "
               << configs_.transform_to_mesh_path;
  }
  colmap_reconstruction_->Transform(transforms[0]);
  colmap_reconstruction_->Write(configs_.output_transformed_colmap_path);

  renderer_ = std::make_unique<MeshRenderer>(
      configs_.image_width, configs_.image_height, configs_.vertex_file,
      configs_.fragment_file);
  renderer_->InitEGL();
  renderer_->LoadMesh(configs_.mesh_file);

  LOG(INFO) << "Finished loading";

  superpoint_ = std::make_unique<SuperPointFeatureExtractor>(
      configs_.model_config_file, configs_.model_weight_folder);
  if (!superpoint_->Build()) {
    return false;
  }
  superglue_ = std::make_unique<SuperGlueFeatureMatcher>(
      configs_.model_config_file, configs_.model_weight_folder);
  if (!superglue_->Build()) {
    return false;
  }

  LOG(INFO) << "Loaded mesh and model";
  std::this_thread::sleep_for(std::chrono::seconds(5));

  unprojector_ = std::make_unique<Unprojector>(configs_.image_height,
                                               configs_.image_width);

  image_registrator_ = std::make_unique<ImageRegistrator>();

  initial_pose_provider_ =
      CreateInitialPoseProvider(configs_, colmap_reconstruction_.get());

  output_stats_file = std::ofstream(configs_.output_stats_path);
  output_mesh_pose_file = std::ofstream(configs_.output_mesh_pose_path);

  output_thread = new std::thread(&UbiposeController::ProcessResultQueue, this);
  return true;
}

void UbiposeController::Localize(const UbiposeQuery &query) {
  if (query.image_timestamp - prev_query_image_timestamp_ < 1e6) {
    return;
  }

  if (!initialized_) {
    auto [qvec, tvec] = InitializeLocalize(query);
    if (initialized_) {
      AddImageToQueue(query.image, query.image_timestamp, qvec, tvec,
                      query.original_filename);
    }
    return;
  }

  auto [qvec, tvec] = LocalizeFrame(query);
  AddImageToQueue(query.image, query.image_timestamp, qvec, tvec,
                  query.original_filename);
}

void UbiposeController::Stop() {
  std::lock_guard<std::mutex> lock(mu_);
  stoping_ = true;
}

std::pair<Eigen::Vector4d, Eigen::Vector3d>
UbiposeController::InitializeLocalize(const UbiposeQuery &query) {
  LOG(INFO) << "Initialize Meshloc image at " << std::setprecision(20)
            << query.image_timestamp << " vo at " << query.vio_timestamp;
  auto sfm_result =
      initial_pose_provider_->GetInitialExtrinsicAt(query.image_timestamp);

  if (!sfm_result.has_value()) {
    LOG(INFO) << "No sfm data for image at " << query.image_timestamp;
    return {{}, {}};
  }
  auto [sfm_qvec, sfm_tvec] = *sfm_result;
  auto sfm_extrinsic = ColmapVec2GlExtrinsic(sfm_qvec, sfm_tvec);

  prev_sfm_qvec_ = sfm_qvec;
  prev_sfm_tvec_ = sfm_tvec;

  UbiposeStats stats;
  Eigen::Vector4d pred_qvec;
  Eigen::Vector3d pred_tvec;
  std::vector<MapPoint> matched_points;
  std::vector<MapPoint *> empty_project_map_points;
  std::vector<cv::KeyPoint> empty_project_cv_keypoints;
  if (!configs_.perform_localization) {
    pred_qvec = sfm_qvec;
    pred_tvec = sfm_tvec;
    stats.localized = true;
  } else {
    if (configs_.method == 0) {
      std::tie(pred_qvec, pred_tvec) = RunBaselineUbipose(
          configs_, query.image_timestamp, query.image, sfm_qvec, sfm_tvec,
          sfm_extrinsic, query.intrinsic, superpoint_.get(), superglue_.get(),
          renderer_.get(), unprojector_.get(), image_registrator_.get(),
          &stats);
    } else if (configs_.method == 1) {
      std::tie(pred_qvec, pred_tvec) = RunBaselineUbiposeSingleSG(
          configs_, query.image_timestamp, query.image, sfm_qvec, sfm_tvec,
          sfm_extrinsic, query.intrinsic, superpoint_.get(), superglue_.get(),
          renderer_.get(), unprojector_.get(), image_registrator_.get(),
          &stats);
    } else if (configs_.method == 2) {
      std::tie(pred_qvec, pred_tvec, matched_points) =
          RunUbiposeWithAllOptimization(
              configs_, query.image_timestamp, query.image, sfm_qvec, sfm_tvec,
              sfm_extrinsic, query.intrinsic, empty_project_map_points,
              empty_project_cv_keypoints, superpoint_.get(), superglue_.get(),
              renderer_.get(), unprojector_.get(), image_registrator_.get(),
              &stats);
    } else {
      LOG(FATAL) << "Invalid method";
    }
    stats.accepted = stats.localized;
  }

  prev_query_image_timestamp_ = query.image_timestamp;
  if (!stats.localized && !stats.cache_localized) {
    LOG(INFO) << "Initialize failed";
  }

  if (stats.localized || stats.cache_localized) {
    initialized_ = true;
    last_frame_use_vio_ = false;
    prev_localized_vio_timestamp_ = query.vio_timestamp;
    prev_localized_vio_qvec_ = query.vio_qvec;
    prev_localized_vio_tvec_ = query.vio_tvec;

    prev_localized_image_timestamp_ = query.image_timestamp;
    prev_localized_meshloc_qvec_ = pred_qvec;
    prev_localized_meshloc_tvec_ = pred_tvec;
    map_points_ = matched_points;
  }
  return {pred_qvec, pred_tvec};
}

std::pair<Eigen::Vector4d, Eigen::Vector3d>
UbiposeController::LocalizeFrame(const UbiposeQuery &query) {
  Eigen::Vector4d rel_qvec;
  Eigen::Vector3d rel_tvec;
  GetVioRelativePose(query.vio_qvec, query.vio_tvec, &rel_qvec, &rel_tvec);

  Eigen::Vector4d vio_est_qvec;
  Eigen::Vector3d vio_est_tvec;
  ColmapConcatenatePoses(prev_localized_meshloc_qvec_,
                         prev_localized_meshloc_tvec_, rel_qvec, rel_tvec,
                         &vio_est_qvec, &vio_est_tvec);

  auto sfm_result =
      colmap_reconstruction_->GetExtrinsicFromColmap(query.image_timestamp);

  Eigen::Vector4d pred_qvec;
  Eigen::Vector3d pred_tvec;
  bool accept_result = false;
  if (configs_.perform_localization) {
    auto input_extrinisc = ColmapVec2GlExtrinsic(vio_est_qvec, vio_est_tvec);
    auto [projected_map_points, projected_cv_keypoints] =
        MapPointsInImage(vio_est_qvec, vio_est_tvec, query.image.cols,
                         query.image.rows, query.intrinsic);
    std::vector<ubipose::MapPoint> matched_points;

    UbiposeStats stats;
    stats.vio_est_qvec = vio_est_qvec;
    stats.vio_est_tvec = vio_est_tvec;

    if (configs_.method == 0) {
      std::tie(pred_qvec, pred_tvec) = RunBaselineUbipose(
          configs_, query.image_timestamp, query.image, vio_est_qvec,
          vio_est_tvec, input_extrinisc, query.intrinsic, superpoint_.get(),
          superglue_.get(), renderer_.get(), unprojector_.get(),
          image_registrator_.get(), &stats);
    } else if (configs_.method == 1) {
      std::tie(pred_qvec, pred_tvec) = RunBaselineUbiposeSingleSG(
          configs_, query.image_timestamp, query.image, vio_est_qvec,
          vio_est_tvec, input_extrinisc, query.intrinsic, superpoint_.get(),
          superglue_.get(), renderer_.get(), unprojector_.get(),
          image_registrator_.get(), &stats);
    } else if (configs_.method == 2) {
      std::tie(pred_qvec, pred_tvec, matched_points) =
          RunUbiposeWithAllOptimization(
              configs_, query.image_timestamp, query.image, vio_est_qvec,
              vio_est_tvec, input_extrinisc, query.intrinsic,
              projected_map_points, projected_cv_keypoints, superpoint_.get(),
              superglue_.get(), renderer_.get(), unprojector_.get(),
              image_registrator_.get(), &stats);
    } else {
      LOG(FATAL) << "Invalid method";
    }
    accept_result = stats.accepted_cache || stats.accepted;

    if (query.vio_timestamp - prev_localized_vio_timestamp_ >
        configs_.num_vio_sec_before_loss * 1e6) {
      LOG(INFO) << "Too long since last localized pose, considering taking the "
                   "localize result. Current ts "
                << query.vio_timestamp << " prev ts "
                << prev_localized_vio_timestamp_;
      if ((stats.localized && stats.num_inliers > 100) ||
          (stats.cache_localized && stats.cache_localized_num_inliers > 100)) {
        LOG(INFO) << "Taking the result because it saids it's localized";
        accept_result = true;
        if (stats.localized) {
          pred_qvec = stats.localized_qvec;
          pred_tvec = stats.localized_tvec;
        } else {
          pred_qvec = stats.cache_localized_qvec;
          pred_tvec = stats.cache_localized_tvec;
        }
      }
    }

    if (accept_result) {
      map_points_.insert(map_points_.end(), matched_points.begin(),
                         matched_points.end());
    }
    PurgeBadMapPoints();

    OutputStats(stats, query.original_filename, output_stats_file);

    if (configs_.debugging && sfm_result.has_value()) {
      auto projection_matrix = IntrinsicToProjectionMatrix(
          query.intrinsic, query.image.cols, query.image.rows);
      output_mesh_pose_file
          << query.original_filename << "," << sfm_result->first[0] << ","
          << sfm_result->first[1] << "," << sfm_result->first[2] << ","
          << sfm_result->first[3] << "," << sfm_result->second[0] << ","
          << sfm_result->second[1] << "," << sfm_result->second[2] << std::endl;
      RenderLocalizeOutput(
          renderer_.get(),
          ColmapVec2GlExtrinsic(sfm_result->first, sfm_result->second),
          projection_matrix, query.image_timestamp, query.image, "gt");
      RenderLocalizeOutput(
          renderer_.get(), ColmapVec2GlExtrinsic(vio_est_qvec, vio_est_tvec),
          projection_matrix, query.image_timestamp, query.image, "vio");
    }
  }

  prev_query_image_timestamp_ = query.image_timestamp;

  Eigen::Vector4d output_qvec = vio_est_qvec;
  Eigen::Vector3d output_tvec = vio_est_tvec;
  if (!configs_.perform_localization) {
    CHECK(sfm_result.has_value());
    auto [sfm_qvec, sfm_tvec] = *sfm_result;
    LOG(INFO) << "gt estimation";
    LOG(INFO) << sfm_qvec;
    LOG(INFO) << vio_est_tvec;
    LOG(INFO) << "input estimation";
    LOG(INFO) << vio_est_qvec;
    LOG(INFO) << sfm_tvec;

    Eigen::Vector4d gt_rel_qvec;
    Eigen::Vector3d gt_rel_tvec;

    ColmapComputeRelativePose(prev_sfm_qvec_, prev_sfm_tvec_, sfm_qvec,
                              sfm_tvec, &gt_rel_qvec, &gt_rel_tvec);
    LOG(INFO) << "gt relative estimation ";
    LOG(INFO) << gt_rel_qvec;
    LOG(INFO) << gt_rel_tvec;
    LOG(INFO) << "gt tvec norm = " << gt_rel_tvec.norm();

    LOG(INFO) << "vio relative estimation";
    LOG(INFO) << rel_qvec;
    LOG(INFO) << rel_tvec;
    LOG(INFO) << "vio tvec norm = " << rel_tvec.norm();
    LOG(INFO) << "Use input pose";

    prev_sfm_qvec_ = sfm_qvec;
    prev_sfm_tvec_ = sfm_tvec;

    pred_qvec = vio_est_qvec;
    pred_tvec = vio_est_tvec;
    accept_result = true;
  }

  if (accept_result) {
    output_qvec = pred_qvec;
    output_tvec = pred_tvec;
    last_frame_use_vio_ = false;

    prev_localized_vio_timestamp_ = query.vio_timestamp;
    prev_localized_vio_qvec_ = query.vio_qvec;
    prev_localized_vio_tvec_ = query.vio_tvec;

    prev_localized_image_timestamp_ = query.image_timestamp;
    prev_localized_meshloc_qvec_ = output_qvec;
    prev_localized_meshloc_tvec_ = output_tvec;
  } else {
    last_frame_use_vio_ = true;
  }

  return {output_qvec, output_tvec};
}

void UbiposeController::GetVioRelativePose(const Eigen::Vector4d &vio_qvec,
                                           const Eigen::Vector3d &vio_tvec,
                                           Eigen::Vector4d *rel_qvec,
                                           Eigen::Vector3d *rel_tvec) {
  Eigen::Vector4d inv_vio_qvec;
  Eigen::Vector3d inv_vio_tvec;
  ColmapInvertPose(vio_qvec, vio_tvec, &inv_vio_qvec, &inv_vio_tvec);

  Eigen::Vector4d inv_base_vio_qvec;
  Eigen::Vector3d inv_base_vio_tvec;
  ColmapInvertPose(prev_localized_vio_qvec_, prev_localized_vio_tvec_,
                   &inv_base_vio_qvec, &inv_base_vio_tvec);

  ColmapComputeRelativePose(inv_base_vio_qvec, inv_base_vio_tvec, inv_vio_qvec,
                            inv_vio_tvec, rel_qvec, rel_tvec);
}

void UbiposeController::AddImageToQueue(cv::Mat image, size_t image_timestamp,
                                        const Eigen::Vector4d &qvec,
                                        const Eigen::Vector3d &tvec,
                                        const std::string &original_filename) {
  UbiposeResult result{
      .image = image.clone(),
      .image_timestamp = image_timestamp,
      .qvec = qvec,
      .tvec = tvec,
      .original_filename = original_filename,
  };
  {
    std::lock_guard<std::mutex> lock(mu_);
    result_queue_.push(std::move(result));
  }
}

void UbiposeController::ProcessResultQueue() {
  std::string output_image_folder = configs_.output_images_folder;
  std::string output_result_file = configs_.output_localization_path;
  std::ofstream output_fs(output_result_file);
  while (true) {
    UbiposeResult result;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (result_queue_.empty()) {
        if (!stoping_)
          continue;
        else
          break;
      }
      result = result_queue_.front();
      result_queue_.pop();
    }
    std::string output_filename =
        output_image_folder + std::to_string(result.image_timestamp) + ".jpg";
    LOG(INFO) << output_filename;
    if (configs_.output_images) {
      cv::imwrite(output_filename, result.image);
    }

    output_fs << (result.original_filename.empty() ? output_filename
                                                   : result.original_filename)
              << " " << result.qvec(0) << " " << result.qvec(1) << " "
              << result.qvec(2) << " " << result.qvec(3) << " "
              << result.tvec(0) << " " << result.tvec(1) << " "
              << result.tvec(2) << std::endl;
  }
}

bool UbiposeController::AcceptResult(double est_error_R, double est_error_t,
                                     const UbiposeStats &stats) {
  double inlier_ratio =
      (static_cast<double>(stats.num_inliers) / stats.num_matches);

  // Completely failed
  if (!stats.localized) {
    LOG(INFO) << "Mesh loc failed, use vio";
    return false;
  }

  if (inlier_ratio > configs_.strong_inlier_ratio &&
      stats.num_matches > configs_.strong_superglue_matches &&
      est_error_R < configs_.strong_error_R &&
      est_error_t < configs_.strong_error_t) {
    LOG(INFO) << "We think the result is good";
    return true;
  }

  if ((inlier_ratio < configs_.weak_inlier_ratio ||
       est_error_R > configs_.weak_error_R ||
       est_error_t > configs_.weak_error_t)) {
    LOG(INFO) << "We think the result is bad, use vio";
    return false;
  }
  return true;
}

bool UbiposeController::AcceptCacheLocalizeResult(double est_error_R,
                                                  double est_error_t,
                                                  const UbiposeStats &stats) {
  double inlier_ratio =
      (static_cast<double>(stats.num_inliers) / stats.num_matches);

  // Completely failed
  if (!stats.cache_localized) {
    LOG(INFO) << "Cache localize failed";
    return false;
  }

  if (inlier_ratio > configs_.strong_inlier_ratio &&
      stats.num_matches > configs_.strong_superglue_matches &&
      est_error_R < configs_.strong_error_R &&
      est_error_t < configs_.strong_error_t) {
    LOG(INFO) << "Cache localize is strong";
    return true;
  }

  if ((inlier_ratio < configs_.weak_inlier_ratio ||
       est_error_R > configs_.weak_error_R ||
       est_error_t > configs_.weak_error_t)) {
    LOG(INFO) << "Cache localize seem to be failed";
    return false;
  }
  LOG(INFO) << "Cache localize accepted";
  return true;
}

std::pair<std::vector<MapPoint *>, std::vector<cv::KeyPoint>>
UbiposeController::MapPointsInImage(
    const Eigen::Vector4d &qvec, const Eigen::Vector3d &tvec, size_t width,
    size_t height,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic) {
  colmap::Camera camera;
  camera.SetModelIdFromName("PINHOLE");
  camera.SetWidth(width);
  camera.SetHeight(height);
  camera.SetParams(
      {intrinsic(0, 0), intrinsic(1, 1), intrinsic(0, 2), intrinsic(1, 2)});

  std::vector<MapPoint *> projected_map_points;
  auto projection_matrix = ColmapComposeProjectionMatrix(qvec, tvec);
  LOG(INFO) << "Number of point in map " << map_points_.size();
  std::vector<cv::KeyPoint> projected_cv_keypoints;

  for (size_t i = 0; i < map_points_.size(); i++) {
    if (ColmapCalculateDepth(projection_matrix, map_points_[i].point_3d()) >
        0) {
      auto eigen2d = ColmapProjectPointToImage(map_points_[i].point_3d(),
                                               projection_matrix, camera);
      if (eigen2d.x() - 0.5 >= 0 && eigen2d.x() - 0.5 < width &&
          eigen2d.y() - 0.5 >= 0 && eigen2d.y() - 0.5 < height) {

        map_points_[i].AddVisible();
        projected_map_points.push_back(&map_points_[i]);
        projected_cv_keypoints.push_back(ColmapPoint2dToCvKeyPoint(eigen2d));
      }
    }
  }
  LOG(INFO) << "Number of point in curr image " << projected_map_points.size();
  return {projected_map_points, projected_cv_keypoints};
}

void UbiposeController::PurgeBadMapPoints() {

  std::vector<MapPoint> remained_map_points;
  for (auto &mp : map_points_) {
    if (mp.ObservedRatio() > 0.25) {
      remained_map_points.push_back(mp);
    }
  }

  map_points_ = remained_map_points;
}

} // namespace ubipose
