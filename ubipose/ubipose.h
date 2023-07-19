#ifndef UBIPOSE_MESHLOC_H
#define UBIPOSE_MESHLOC_H

#include <tuple>

#include <Eigen/Dense>

#include "configs.h"
#include "keypoints_utils.h"
#include "modules/image_registrator.h"
#include "modules/renderer.h"
#include "modules/superpointglue.h"
#include "modules/unprojector.h"
#include "pose_utils.h"
#include "types.h"

namespace ubipose {
std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunBaselineUbipose(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator, ubipose::UbiposeStats *stats);

std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunBaselineUbiposeSingleSG(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator, ubipose::UbiposeStats *stats);

std::tuple<Eigen::Vector4d, Eigen::Vector3d> RunFusedUbipose(
    const UbiposeConfigs &configs, size_t timestamp, cv::Mat query_image,
    const Eigen::Vector4d &vio_est_qvec, const Eigen::Vector3d &vio_est_tvec,
    const ubipose::EigenGl4f &input_extrinsic,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
    ubipose::SuperPointFeatureExtractor *superpoint,
    ubipose::SuperGlueFeatureMatcher *superglue,
    ubipose::MeshRenderer *renderer, ubipose::Unprojector *unprojector,
    ubipose::ImageRegistrator *image_registrator, ubipose::UbiposeStats *stats);

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
    ubipose::ImageRegistrator *image_registrator, ubipose::UbiposeStats *stats);
} // namespace ubipose

#endif
