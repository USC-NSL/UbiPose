#ifndef UBIPOSE_POSE_UTILS_H
#define UBIPOSE_POSE_UTILS_H

#include "types.h"

#include <Eigen/Dense>
#include <colmap/base/camera.h>
#include <colmap/util/types.h>

namespace ubipose {

std::vector<ubipose::EigenGl4f, Eigen::aligned_allocator<ubipose::EigenGl4f>>
ReadCameraExtrinsic(const std::string &pose_file, char delimiter = ',');

EigenGl4f IntrinsicToProjectionMatrix(
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height);

std::vector<ubipose::EigenGl4f, Eigen::aligned_allocator<ubipose::EigenGl4f>>
SampleCameraExtrinsicAround(const ubipose::EigenGl4f &extrinsic);

ubipose::EigenGl4f ColmapVec2GlPose(const Eigen::Vector4d &qvec,
                                    const Eigen::Vector3d &tvec);

ubipose::EigenGl4f ColmapVec2GlExtrinsic(const Eigen::Vector4d &qvec,
                                         const Eigen::Vector3d &tvec);

std::pair<float, float> CalculateError(const Eigen::Vector4d &gt_qvec,
                                       const Eigen::Vector3d &gt_tvec,
                                       const Eigen::Vector4d &qvec,
                                       const Eigen::Vector3d &tvec);

void ColmapConcatenatePoses(const Eigen::Vector4d &qvec1,
                            const Eigen::Vector3d &tvec1,
                            const Eigen::Vector4d &qvec2,
                            const Eigen::Vector3d &tvec2,
                            Eigen::Vector4d *qvec12, Eigen::Vector3d *tvec12);

void ColmapComputeRelativePose(const Eigen::Vector4d &qvec1,
                               const Eigen::Vector3d &tvec1,
                               const Eigen::Vector4d &qvec2,
                               const Eigen::Vector3d &tvec2,
                               Eigen::Vector4d *qvec12,
                               Eigen::Vector3d *tvec12);

void ColmapInvertPose(const Eigen::Vector4d &qvec, const Eigen::Vector3d &tvec,
                      Eigen::Vector4d *inv_qvec, Eigen::Vector3d *inv_tvec);

Eigen::Matrix3x4d ColmapComposeProjectionMatrix(const Eigen::Vector4d &qvec,
                                                const Eigen::Vector3d &tvec);

double ColmapCalculateDepth(const Eigen::Matrix3x4d &proj_matrix,
                            const Eigen::Vector3d &point3D);

Eigen::Vector2d ColmapProjectPointToImage(const Eigen::Vector3d &point3D,
                                          const Eigen::Matrix3x4d &proj_matrix,
                                          const colmap::Camera &camera);
} // namespace ubipose

#endif // !DEBUG
