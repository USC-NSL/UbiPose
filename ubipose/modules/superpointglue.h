
#ifndef UBIPOSE_MODULES_SUPERPOINTGLUE_H
#define UBIPOSE_MODULES_SUPERPOINTGLUE_H

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace ubipose {
class SuperPointImpl;
class SuperGlueImpl;

class SuperPointFeatureExtractor {
public:
  SuperPointFeatureExtractor(const std::string &config_path,
                             const std::string &model_path);
  ~SuperPointFeatureExtractor();
  bool Build();
  void Compute(const cv::Mat &image,
               Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

private:
  SuperPointImpl *superpoint_impl_;
};

class SuperGlueFeatureMatcher {
public:
  SuperGlueFeatureMatcher(const std::string &config_path,
                          const std::string &model_path);
  ~SuperGlueFeatureMatcher();
  bool Build();
  void MatchPoints(Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                   Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                   std::vector<cv::DMatch> &matches);

private:
  SuperGlueImpl *superglue_impl_;
};

} // namespace ubipose

#endif
