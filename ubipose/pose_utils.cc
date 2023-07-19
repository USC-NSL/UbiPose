#include "pose_utils.h"

#include <fstream>

#include <colmap/base/pose.h>
#include <colmap/base/projection.h>
#include <colmap/util/math.h>

namespace ubipose {

std::vector<ubipose::EigenGl4f, Eigen::aligned_allocator<ubipose::EigenGl4f>>
ReadCameraExtrinsic(const std::string &pose_file, char delimiter) {
  std::cout << "Reading from " << pose_file << std::endl;
  std::ifstream ifile(pose_file);
  std::vector<
      Eigen::Matrix<float, 4, 4, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>>
      transforms;

  std::string line;
  std::string token;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> m;
  int col = 0;

  // std::fstream ss(line);

  for (int i = 0; i < 16; i++) {
    if (i > 0 && i % 4 == 0) {
      col++;
    }
    std::getline(ifile, token, delimiter);

    if (token.empty()) {
      return {};
    }

    double val = std::stod(token);
    m(i % 4, col) = val;
  }

  transforms.push_back(m);

  return transforms;
}

EigenGl4f IntrinsicToProjectionMatrix(
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height) {
  float cx = intrinsic(0, 2);
  float cy = intrinsic(1, 2);
  float fx = intrinsic(0, 0);
  float fy = intrinsic(1, 1);

  EigenGl4f result = EigenGl4f::Zero();
  result(0, 0) = 2.0 * fx / width;
  result(1, 1) = 2.0 * fy / height;
  result(0, 2) = 1.0 - 2.0 * cx / width;
  result(1, 2) = 2.0 * cy / height - 1.0;
  result(3, 2) = -1.0;

  const double Z_NEAR = 0.05;
  result(2, 2) = -1.0;
  result(2, 3) = -2 * Z_NEAR;
  return result;
}

std::vector<ubipose::EigenGl4f, Eigen::aligned_allocator<ubipose::EigenGl4f>>
SampleCameraExtrinsicAround(const ubipose::EigenGl4f &extrinsic) {
  auto forward = extrinsic;
  forward(2, 3) += 1;
  // forward(3, 3) = 1;
  auto backward = extrinsic;
  backward(2, 3) -= 1;
  // backward(3, 3) = 1;

  auto curr = extrinsic;
  // curr(3, 3) = 1;

  return {backward, curr, forward};
}

ubipose::EigenGl4f ColmapVec2GlPose(const Eigen::Vector4d &qvec,
                                    const Eigen::Vector3d &tvec) {
  Eigen::Matrix3d rotmat = colmap::QuaternionToRotationMatrix(qvec).transpose();

  ubipose::EigenGl4f T = ubipose::EigenGl4f::Identity();
  T.block<3, 3>(0, 0) = rotmat.cast<float>();
  T.block<3, 1>(0, 3) = (-rotmat * tvec).cast<float>();

  T.col(1) *= -1;
  T.col(2) *= -1;

  return T;
}

ubipose::EigenGl4f ColmapVec2GlExtrinsic(const Eigen::Vector4d &qvec,
                                         const Eigen::Vector3d &tvec) {
  return ColmapVec2GlPose(qvec, tvec).inverse();
}

std::pair<float, float> CalculateError(const Eigen::Vector4d &gt_qvec,
                                       const Eigen::Vector3d &gt_tvec,
                                       const Eigen::Vector4d &qvec,
                                       const Eigen::Vector3d &tvec) {
  auto gt_R = colmap::QuaternionToRotationMatrix(gt_qvec);
  auto R = colmap::QuaternionToRotationMatrix(qvec);
  auto error_tvec = (-gt_R.transpose() * gt_tvec + R.transpose() * tvec).norm();

  auto cos = ((gt_R.transpose() * R).trace() - 1) / 2;
  if (cos < -1) {
    cos = -1;
  } else if (cos > 1) {
    cos = 1;
  }
  auto error_qvec = colmap::RadToDeg(std::abs(acos(cos)));

  return {error_qvec, error_tvec};
}

void ColmapConcatenatePoses(const Eigen::Vector4d &qvec1,
                            const Eigen::Vector3d &tvec1,
                            const Eigen::Vector4d &qvec2,
                            const Eigen::Vector3d &tvec2,
                            Eigen::Vector4d *qvec12, Eigen::Vector3d *tvec12) {
  colmap::ConcatenatePoses(qvec1, tvec1, qvec2, tvec2, qvec12, tvec12);
}

void ColmapComputeRelativePose(const Eigen::Vector4d &qvec1,
                               const Eigen::Vector3d &tvec1,
                               const Eigen::Vector4d &qvec2,
                               const Eigen::Vector3d &tvec2,
                               Eigen::Vector4d *qvec12,
                               Eigen::Vector3d *tvec12) {
  colmap::ComputeRelativePose(qvec1, tvec1, qvec2, tvec2, qvec12, tvec12);
}

void ColmapInvertPose(const Eigen::Vector4d &qvec, const Eigen::Vector3d &tvec,
                      Eigen::Vector4d *inv_qvec, Eigen::Vector3d *inv_tvec) {
  colmap::InvertPose(qvec, tvec, inv_qvec, inv_tvec);
}

Eigen::Matrix3x4d ColmapComposeProjectionMatrix(const Eigen::Vector4d &qvec,
                                                const Eigen::Vector3d &tvec) {
  return colmap::ComposeProjectionMatrix(qvec, tvec);
}

double ColmapCalculateDepth(const Eigen::Matrix3x4d &proj_matrix,
                            const Eigen::Vector3d &point3D) {
  return colmap::CalculateDepth(proj_matrix, point3D);
}

Eigen::Vector2d ColmapProjectPointToImage(const Eigen::Vector3d &point3D,
                                          const Eigen::Matrix3x4d &proj_matrix,
                                          const colmap::Camera &camera) {
  return colmap::ProjectPointToImage(point3D, proj_matrix, camera);
}
} // namespace ubipose
