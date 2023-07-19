#include "image_registrator.h"

namespace ubipose {

constexpr size_t kMinNumInliers = 30;

ImageRegistrator::ImageRegistrator()
    : abs_pose_estimation_(new ColmapAbsolutePoseEstimation()),
      abs_pose_refinement_(new ColmapAbsolutePoseRefinement()) {}

ImageRegistrator::ImageRegistrator(
    const ColmapAbsolutePoseEstimationOptions &abs_pose_est_options,
    const ColmapAbsolutePoseRefinementOptions abs_pose_refine_options)
    : abs_pose_estimation_(
          new ColmapAbsolutePoseEstimation(abs_pose_est_options)),
      abs_pose_refinement_(
          new ColmapAbsolutePoseRefinement(abs_pose_refine_options)) {}

bool ImageRegistrator::Register(
    const std::vector<Eigen::Vector2d> &tri_points2D,
    const std::vector<Eigen::Vector3d> &tri_points3D,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
    size_t *num_inliers, std::vector<char> *inlier_mask) {

  if (!abs_pose_estimation_->RunColmapAbsolutePoseEstimation(
          tri_points2D, tri_points3D, intrinsic, width, height, qvec, tvec,
          num_inliers, inlier_mask)) {
    return false;
  }

  if (*num_inliers < kMinNumInliers) {
    return false;
  }

  if (!abs_pose_refinement_->RunColmapAbsolutePoseRefinement(
          tri_points2D, tri_points3D, intrinsic, width, height, qvec, tvec,
          num_inliers, inlier_mask)) {
    return false;
  }

  return true;
}

} // namespace ubipose
