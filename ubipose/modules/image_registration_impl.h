#ifndef UBIPOSE_MODULES_IMAGE_REGISTRATION_IMPL_H
#define UBIPOSE_MODULES_IMAGE_REGISTRATION_IMPL_H

#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace ubipose {

class ColmapAbsolutePoseEstimationImpl;
class ColmapAbsolutePoseRefinementImpl;

struct ColmapRansacOptions {
  size_t max_error = 12;
  double min_inlier_ratio = 0.1;
  size_t min_num_trials = 1000;
  size_t max_num_trials = 10000;
  double confidence = 0.9999;
};

struct ColmapAbsolutePoseEstimationOptions {
  size_t num_threads = 4;
  size_t num_focal_length_samples = 30;
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10;
  bool estimate_focal_length = false;

  ColmapRansacOptions ransac_options;
};

struct ColmapAbsolutePoseRefinementOptions {
  double gradient_tolerance = 1e-9;
  int max_num_iterations = 100;
  bool refine_focal_length = false;
  bool refine_extra_params = false;
  bool print_summary = false;
};

class ColmapAbsolutePoseEstimation {
public:
  ColmapAbsolutePoseEstimation();
  ColmapAbsolutePoseEstimation(
      const ColmapAbsolutePoseEstimationOptions &options);
  ~ColmapAbsolutePoseEstimation();

  bool RunColmapAbsolutePoseEstimation(
      const std::vector<Eigen::Vector2d> &tri_points2D,
      const std::vector<Eigen::Vector3d> &tri_points3D,
      const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
      size_t width, size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
      size_t *num_inliers, std::vector<char> *inlier_mask);

private:
  ColmapAbsolutePoseEstimationImpl *impl_;
};

class ColmapAbsolutePoseRefinement {
public:
  ColmapAbsolutePoseRefinement();
  ColmapAbsolutePoseRefinement(
      const ColmapAbsolutePoseRefinementOptions &options);
  ~ColmapAbsolutePoseRefinement();

  bool RunColmapAbsolutePoseRefinement(
      const std::vector<Eigen::Vector2d> &tri_points2D,
      const std::vector<Eigen::Vector3d> &tri_points3D,
      const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
      size_t width, size_t height, Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
      size_t *num_inliers, std::vector<char> *inlier_mask);

private:
  ColmapAbsolutePoseRefinementImpl *impl_;
};

} // namespace ubipose
#endif
