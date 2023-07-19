#include "ios_utils.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <colmap/base/pose.h>
#include <colmap/base/similarity_transform.h>
#include <fstream>
#include <iostream>

#include "pose_utils.h"
#include "types.h"

namespace ubipose {

Eigen::Matrix<float, 3, 3, Eigen::RowMajor>
ReadCameraIntrinsic(const std::string &pose_file) {
  std::ifstream ifile(pose_file);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> result;
  std::string line;
  std::string token;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::getline(ifile, token, ',');

      float val = std::stof(token);
      result(j, i) = val;
    }
  }
  return result;
}

std::map<double, ARCameraFrame> LoadARCameraExtrinsics(const std::string &path,
                                                       bool use_aranchor) {
  std::map<double, ARCameraFrame> vio_result;
  boost::filesystem::directory_iterator dir_it(path);
  for (; dir_it != boost::filesystem::directory_iterator(); ++dir_it) {
    // Only lookat the extrinsic file right now
    auto filename = dir_it->path().filename().string();
    if (filename.find("output") != std::string::npos &&
        filename.find("anchor") == std::string::npos) {
      auto poses = ReadCameraExtrinsic(dir_it->path().string());
      if (poses.size() != 1) {
        LOG(ERROR) << "Invalid extrinsic file";
        continue;
      }

      ubipose::EigenGl4f slam_pose = poses[0];
      ubipose::EigenGl4f output_pose = slam_pose;
      if (use_aranchor) {
        auto anchor_pose_filename = "anchors_" + filename;
        auto anchor_pose_path =
            (dir_it->path().parent_path() / anchor_pose_filename);
        LOG(INFO) << "Anchor path " << anchor_pose_path.string();

        if (!boost::filesystem::exists(anchor_pose_path)) {
          LOG(ERROR) << "Anchor extrinsic file not exist";
          continue;
        }

        auto anchor_poses =
            ubipose::ReadCameraExtrinsic(anchor_pose_path.string());
        if (anchor_poses.size() != 1) {
          LOG(ERROR) << "Invalid extrinsic file";
          continue;
        }
        ubipose::EigenGl4f anchor_pose = anchor_poses[0];
        output_pose = (anchor_pose.inverse() * slam_pose);
      }

      auto rot_matrix = output_pose.block<3, 3>(0, 0);
      Eigen::Vector4d qvec =
          colmap::RotationMatrixToQuaternion(rot_matrix.cast<double>());
      Eigen::Vector3d tvec = output_pose.block<3, 1>(0, 3).cast<double>();
      std::string timestamp = filename.substr(0, filename.find('_'));

      auto intrinsic_filename = timestamp + "_intrinsics.txt";
      auto intrinsic_path = (dir_it->path().parent_path() / intrinsic_filename);
      if (!boost::filesystem::exists(intrinsic_path)) {
        LOG(ERROR) << "Intrinsic file not exist";
        continue;
      }

      auto intrinsic = ReadCameraIntrinsic(intrinsic_path.string());

      std::string image_filename = timestamp + ".jpeg";

      LOG(INFO) << "Insert vio for " << timestamp;
      VLOG(1) << "Extrinsic: " << output_pose << " Intrinsic " << intrinsic
              << " Qvec: " << qvec << " tvec: " << tvec;

      Eigen::Matrix3x4d change_coordinate_matrix;
      // x -> x
      // y -> -y
      // z -> -z
      change_coordinate_matrix << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0;

      colmap::SimilarityTransform3 change_coordinate_transform(
          change_coordinate_matrix);
      change_coordinate_transform.TransformPose(&qvec, &tvec);

      vio_result[std::stod(timestamp)] =
          ARCameraFrame{image_filename, qvec, tvec, intrinsic};
    }
  }
  return vio_result;
}
} // namespace ubipose
