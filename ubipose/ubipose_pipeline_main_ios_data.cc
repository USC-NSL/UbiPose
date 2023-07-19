
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "colmap/colmap_reconstruction.h"
#include "colmap/colmap_utils.h"
#include "configs.h"
#include "ios_utils.h"
#include "pose_utils.h"
#include "ubipose_controller.h"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/initialize.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <colmap/base/pose.h>
#include <colmap/base/projection.h>
#include <colmap/util/math.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

ABSL_FLAG(std::string, arkit_directory, "", "The path to the arkit directory");
ABSL_FLAG(std::string, config_file, "", "The path to the config file");
ABSL_FLAG(bool, use_aranchor, false, "Use geoanchor extrinsic for camera pose");
ABSL_FLAG(double, start_timestamp, -1, "The start timestamp");
ABSL_FLAG(double, end_timestamp, -1, "The end timestamp");

Eigen::Matrix<float, 3, 3, Eigen::RowMajor> ResizedIntrinsic(
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &input_intrinsic,
    int downscale_factor) {
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> result = input_intrinsic;
  result(0, 0) = input_intrinsic(0, 0) / downscale_factor;
  result(1, 1) = input_intrinsic(1, 1) / downscale_factor;
  result(0, 2) = input_intrinsic(0, 2) / downscale_factor;
  result(1, 2) = input_intrinsic(1, 2) / downscale_factor;
  return result;
}

int main(int argc, char **argv) {

  absl::ParseCommandLine(argc, argv);
  // absl::InitializeLog();
  auto config_file = absl::GetFlag(FLAGS_config_file);
  if (config_file.empty()) {
    LOG(FATAL) << "Need to provide config file";
  }
  auto arkit_directory = absl::GetFlag(FLAGS_arkit_directory);
  if (arkit_directory.empty()) {
    LOG(FATAL) << "Need to provide the directory of arkit result";
  }

  auto meshloc_config = ubipose::ReadMeshlocConfigs(config_file);
  if (!meshloc_config.has_value()) {
    LOG(FATAL) << "Cannot find config";
  }

  bool use_aranchor = absl::GetFlag(FLAGS_use_aranchor);

  bool has_input_range = false;
  double start_timestamp = absl::GetFlag(FLAGS_start_timestamp);
  double end_timestamp = absl::GetFlag(FLAGS_end_timestamp);
  if (start_timestamp > 0 && end_timestamp > 0) {
    has_input_range = true;
  }

  constexpr int kImageWidth = 1920;
  constexpr int kImageHeight = 1440;

  auto meshloc_controller =
      std::make_unique<ubipose::UbiposeController>(*meshloc_config);
  if (!meshloc_controller->Init()) {
    LOG(FATAL) << "Error, cannot initialize";
  }

  boost::filesystem::path arkit_result_path_boost(arkit_directory);

  auto vio_result = ubipose::LoadARCameraExtrinsics(
      arkit_result_path_boost.string(), use_aranchor);
  int downscale_factor = kImageWidth / meshloc_config->image_width;
  if (kImageHeight / meshloc_config->image_height != downscale_factor) {
    LOG(FATAL) << "Need to keep the aspect ratio the same";
  }

  for (const auto &[input_timestamp, entry] : vio_result) {

    if (has_input_range) {
      if (input_timestamp < start_timestamp) {
        continue;
      }
      if (input_timestamp >= end_timestamp) {
        break;
      }
    }

    auto &[image_filename, vio_qvec, vio_tvec, intrinsic] = entry;
    auto image_path = arkit_result_path_boost / image_filename;
    cv::Mat image = cv::imread(image_path.string());

    cv::Mat resized_image;
    cv::Size resized_size(image.cols / downscale_factor,
                          image.rows / downscale_factor);
    cv::resize(image, resized_image, resized_size, cv::INTER_LINEAR);
    auto resized_intrinsic = ResizedIntrinsic(intrinsic, downscale_factor);

    size_t timestamp = static_cast<size_t>(input_timestamp * 1e6);
    ubipose::UbiposeController::UbiposeQuery query{
        .image = resized_image,
        .image_timestamp = timestamp,
        .vio_timestamp = timestamp,
        .vio_qvec = vio_qvec,
        .vio_tvec = vio_tvec,
        .intrinsic = resized_intrinsic,
        .original_filename = image_path.filename().string()};

    meshloc_controller->Localize(query);
  }

  meshloc_controller->Stop();

  return 0;
}
