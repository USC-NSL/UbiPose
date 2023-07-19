#ifndef UBIPOSE_MODULE_IMAGE_REGISTRATOR_H
#define UBIPOSE_MODULE_IMAGE_REGISTRATOR_H

#include <vector>

#include <Eigen/Dense>

#include "modules/image_registration_impl.h"

namespace ubipose {

class ImageRegistrator {
public:
  ImageRegistrator();
  ImageRegistrator(
      const ColmapAbsolutePoseEstimationOptions &abs_pose_est_options,
      const ColmapAbsolutePoseRefinementOptions abs_pose_refine_options);

  bool Register(const std::vector<Eigen::Vector2d> &tri_points2D,
                const std::vector<Eigen::Vector3d> &tri_points3D,
                const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
                size_t width, size_t height, Eigen::Vector4d *qvec,
                Eigen::Vector3d *tvec, size_t *num_inliers,
                std::vector<char> *inlier_mask);

private:
  std::unique_ptr<ColmapAbsolutePoseEstimation> abs_pose_estimation_;
  std::unique_ptr<ColmapAbsolutePoseRefinement> abs_pose_refinement_;
};

} // namespace ubipose

#endif
