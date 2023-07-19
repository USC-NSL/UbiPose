#include "image_registration_impl.h"

#include <memory>

#include <Eigen/Dense>
#include <colmap/estimators/pose.h>

namespace ubipose {

colmap::AbsolutePoseEstimationOptions CopyAbsPoseOptions(
    const ubipose::ColmapAbsolutePoseEstimationOptions &abs_pose_options) {

  colmap::AbsolutePoseEstimationOptions colmap_abs_pose_options;
  colmap_abs_pose_options.num_threads = abs_pose_options.num_threads;
  colmap_abs_pose_options.num_focal_length_samples =
      abs_pose_options.num_focal_length_samples;
  colmap_abs_pose_options.min_focal_length_ratio =
      abs_pose_options.min_focal_length_ratio;
  colmap_abs_pose_options.max_focal_length_ratio =
      abs_pose_options.max_focal_length_ratio;
  colmap_abs_pose_options.estimate_focal_length =
      abs_pose_options.estimate_focal_length;

  colmap_abs_pose_options.ransac_options.max_error =
      abs_pose_options.ransac_options.max_error;
  colmap_abs_pose_options.ransac_options.min_inlier_ratio =
      abs_pose_options.ransac_options.min_inlier_ratio;
  colmap_abs_pose_options.ransac_options.min_num_trials =
      abs_pose_options.ransac_options.min_num_trials;
  colmap_abs_pose_options.ransac_options.max_num_trials =
      abs_pose_options.ransac_options.max_num_trials;
  colmap_abs_pose_options.ransac_options.confidence =
      abs_pose_options.ransac_options.confidence;

  return colmap_abs_pose_options;
}

colmap::AbsolutePoseRefinementOptions CopyAbsRefineOptions(
    const ubipose::ColmapAbsolutePoseRefinementOptions &refine_options) {
  colmap::AbsolutePoseRefinementOptions colmap_abs_refine_options;
  colmap_abs_refine_options.gradient_tolerance =
      refine_options.gradient_tolerance;
  colmap_abs_refine_options.max_num_iterations =
      refine_options.max_num_iterations;
  colmap_abs_refine_options.refine_focal_length =
      refine_options.refine_focal_length;
  colmap_abs_refine_options.refine_extra_params =
      refine_options.refine_extra_params;
  colmap_abs_refine_options.print_summary = refine_options.print_summary;
  return colmap_abs_refine_options;
}

class ColmapAbsolutePoseEstimationImpl {
public:
  ColmapAbsolutePoseEstimationImpl(
      const colmap::AbsolutePoseEstimationOptions &options)
      : colmap_abs_pose_options_(options) {}

  bool RunColmapAbsolutePoseEstimation(
      const std::vector<Eigen::Vector2d> &tri_points2D,
      const std::vector<Eigen::Vector3d> &tri_points3D,
      const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
      size_t width, size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
      size_t *num_inliers, std::vector<char> *inlier_mask);

private:
  const colmap::AbsolutePoseEstimationOptions colmap_abs_pose_options_;
};

bool ColmapAbsolutePoseEstimationImpl::RunColmapAbsolutePoseEstimation(
    const std::vector<Eigen::Vector2d> &tri_points2D,
    const std::vector<Eigen::Vector3d> &tri_points3D,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
    size_t *num_inliers, std::vector<char> *inlier_mask) {

  colmap::Camera camera;
  camera.SetModelIdFromName("PINHOLE");
  camera.SetWidth(width);
  camera.SetHeight(height);
  camera.SetParams(
      {intrinsic(0, 0), intrinsic(1, 1), intrinsic(0, 2), intrinsic(1, 2)});

  return EstimateAbsolutePose(colmap_abs_pose_options_, tri_points2D,
                              tri_points3D, qvec, tvec, &camera, num_inliers,
                              inlier_mask);
}

class ColmapAbsolutePoseRefinementImpl {
public:
  ColmapAbsolutePoseRefinementImpl(
      const colmap::AbsolutePoseRefinementOptions &options)
      : colmap_abs_refine_options_(options) {}

  bool RunColmapAbsolutePoseRefinement(
      const std::vector<Eigen::Vector2d> &tri_points2D,
      const std::vector<Eigen::Vector3d> &tri_points3D,
      const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
      size_t width, size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
      size_t *num_inliers, std::vector<char> *inlier_mask);

private:
  const colmap::AbsolutePoseRefinementOptions colmap_abs_refine_options_;
};

bool ColmapAbsolutePoseRefinementImpl::RunColmapAbsolutePoseRefinement(
    const std::vector<Eigen::Vector2d> &tri_points2D,
    const std::vector<Eigen::Vector3d> &tri_points3D,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
    size_t *num_inliers, std::vector<char> *inlier_mask) {

  colmap::Camera camera;
  camera.SetModelIdFromName("PINHOLE");
  camera.SetWidth(width);
  camera.SetHeight(height);
  camera.SetParams(
      {intrinsic(0, 0), intrinsic(1, 1), intrinsic(0, 2), intrinsic(1, 2)});
  return RefineAbsolutePose(colmap_abs_refine_options_, *inlier_mask,
                            tri_points2D, tri_points3D, qvec, tvec, &camera);
}

ColmapAbsolutePoseEstimation::ColmapAbsolutePoseEstimation() {
  ColmapAbsolutePoseEstimationOptions default_options;
  colmap::AbsolutePoseEstimationOptions colmap_abs_pose_options =
      CopyAbsPoseOptions(default_options);
  impl_ = new ColmapAbsolutePoseEstimationImpl(colmap_abs_pose_options);
}

ColmapAbsolutePoseEstimation::ColmapAbsolutePoseEstimation(
    const ColmapAbsolutePoseEstimationOptions &options) {
  colmap::AbsolutePoseEstimationOptions colmap_abs_pose_options =
      CopyAbsPoseOptions(options);
  impl_ = new ColmapAbsolutePoseEstimationImpl(colmap_abs_pose_options);
}

ColmapAbsolutePoseEstimation::~ColmapAbsolutePoseEstimation() { delete impl_; }

bool ColmapAbsolutePoseEstimation::RunColmapAbsolutePoseEstimation(
    const std::vector<Eigen::Vector2d> &tri_points2D,
    const std::vector<Eigen::Vector3d> &tri_points3D,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
    size_t *num_inliers, std::vector<char> *inlier_mask) {

  return impl_->RunColmapAbsolutePoseEstimation(tri_points2D, tri_points3D,
                                                intrinsic, width, height, qvec,
                                                tvec, num_inliers, inlier_mask);
}

ColmapAbsolutePoseRefinement::ColmapAbsolutePoseRefinement() {
  ColmapAbsolutePoseRefinementOptions default_options;
  colmap::AbsolutePoseRefinementOptions colmap_abs_refine_options =
      CopyAbsRefineOptions(default_options);
  impl_ = new ColmapAbsolutePoseRefinementImpl(colmap_abs_refine_options);
}

ColmapAbsolutePoseRefinement::ColmapAbsolutePoseRefinement(
    const ColmapAbsolutePoseRefinementOptions &options) {
  colmap::AbsolutePoseRefinementOptions colmap_abs_refine_options =
      CopyAbsRefineOptions(options);
  impl_ = new ColmapAbsolutePoseRefinementImpl(colmap_abs_refine_options);
}

ColmapAbsolutePoseRefinement::~ColmapAbsolutePoseRefinement() { delete impl_; }

bool ColmapAbsolutePoseRefinement::RunColmapAbsolutePoseRefinement(
    const std::vector<Eigen::Vector2d> &tri_points2D,
    const std::vector<Eigen::Vector3d> &tri_points3D,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic, size_t width,
    size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
    size_t *num_inliers, std::vector<char> *inlier_mask) {

  return impl_->RunColmapAbsolutePoseRefinement(tri_points2D, tri_points3D,
                                                intrinsic, width, height, qvec,
                                                tvec, num_inliers, inlier_mask);
}
} // namespace ubipose
