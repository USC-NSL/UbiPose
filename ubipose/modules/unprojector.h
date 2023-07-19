#ifndef UBIPOSE_MODULES_UNPROJECTOR_H
#define UBIPOSE_MODULES_UNPROJECTOR_H

#include <cstddef>

#include <Eigen/Dense>

#include "renderer.h"

namespace ubipose {

class UnprojectHelper {
public:
  UnprojectHelper(const ubipose::EigenGl4f &inv_mvp,
                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor> &xyzw,
                  size_t width, size_t height);
  // UnprojectHelper(const UnprojectHelper &other);
  // void operator=(const UnprojectHelper &other);
  Eigen::Vector3f Get3dPointAtPixel(size_t x, size_t y) const;

private:
  ubipose::EigenGl4f inv_mvp_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xyzw_;
  size_t width_;
  size_t height_;
};

class Unprojector {
public:
  Unprojector(int viewport_height, int viewport_width);

  UnprojectHelper Unproject(cv::Mat depth_img,
                            const ubipose::EigenGl4f &camera_extrinsic,
                            const ubipose::EigenGl4f &projection_matrix) const;

private:
  const int height_;
  const int width_;
  const Eigen::MatrixXf v_lin_matrix;
  const Eigen::MatrixXf h_lin_matrix;
  const Eigen::RowVectorXf x_nd;
  const Eigen::RowVectorXf y_nd;
  const Eigen::RowVectorXf w_nd;
};
} // namespace ubipose

#endif
