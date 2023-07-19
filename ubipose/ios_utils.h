#ifndef UBIPOSE_IOS_UTILS_H
#define UBIPOSE_IOS_UTILS_H

#include <map>
#include <string>

#include <eigen3/Eigen/Dense>

namespace ubipose {

struct ARCameraFrame {
  std::string filename;
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> intrinsic;
};

std::map<double, ARCameraFrame> LoadARCameraExtrinsics(const std::string &path,
                                                       bool use_aranchor);

} // namespace ubipose

#endif // !DEBUG
