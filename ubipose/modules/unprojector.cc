#include "unprojector.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

namespace ubipose {

UnprojectHelper::UnprojectHelper(
    const EigenGl4f &inv_mvp,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        &xyzw,
    size_t width, size_t height)
    : inv_mvp_(inv_mvp), xyzw_(xyzw), width_(width), height_(height) {}

Eigen::Vector3f UnprojectHelper::Get3dPointAtPixel(size_t x, size_t y) const {
  auto xyzw = inv_mvp_ * xyzw_.col(x * height_ + y);
  return Eigen::Vector3f{xyzw[0] / xyzw[3], xyzw[1] / xyzw[3],
                         xyzw[2] / xyzw[3]};
}

Unprojector::Unprojector(int viewport_height, int viewport_width)
    : height_(viewport_height), width_(viewport_width),
      v_lin_matrix(
          Eigen::VectorXf::LinSpaced(viewport_height, 0, viewport_height - 1)
              .rowwise()
              .replicate(viewport_width)),
      h_lin_matrix(
          Eigen::RowVectorXf::LinSpaced(viewport_width, 0, viewport_width - 1)
              .colwise()
              .replicate(viewport_height)),
      x_nd(2.0 * (h_lin_matrix.reshaped().array() / viewport_width) - 1.0),
      y_nd(2.0 * ((viewport_height - 1 - v_lin_matrix.reshaped().array()) /
                  viewport_height) -
           1.0),
      w_nd(Eigen::RowVectorXf::Ones(viewport_height * viewport_width)) {}

UnprojectHelper
Unprojector::Unproject(cv::Mat depth_img, const EigenGl4f &camera_extrinsic,
                       const EigenGl4f &projection_matrix) const {
  auto mvp = projection_matrix * camera_extrinsic;
  auto inv_mvp = mvp.inverse();

  size_t num_pixels = height_ * width_;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> z(
      height_, width_);

  cv::cv2eigen(depth_img, z);

  z = (z.array() * 2.0) - 1;

  // 4 x N
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xyzw(
      4, num_pixels);
  xyzw.row(0) = x_nd;
  xyzw.row(1) = y_nd;
  xyzw.row(2) = z.reshaped();
  xyzw.row(3) = w_nd;

  UnprojectHelper helper(inv_mvp, xyzw, depth_img.cols, depth_img.rows);

  return helper;
}
} // namespace ubipose
