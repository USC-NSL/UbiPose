#include "superpointglue.h"

// Superpointglue
#include "super_glue.h"
#include "super_point.h"
#include "utils.h"

namespace ubipose {

class SuperPointImpl {
public:
  SuperPointImpl(const std::string &config_path,
                 const std::string &model_path) {
    Configs configs(config_path, model_path);
    superpoint_ = std::make_unique<SuperPoint>(configs.superpoint_config);
  }

  bool build() { return superpoint_->build(); }

  bool infer(const cv::Mat &image,
             Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    return superpoint_->infer(image, features);
  }

private:
  std::unique_ptr<SuperPoint> superpoint_;
};

class SuperGlueImpl {
public:
  SuperGlueImpl(const std::string &config_path, const std::string &model_path) {
    Configs configs(config_path, model_path);
    superglue_ = std::make_unique<SuperGlue>(configs.superglue_config);
  }

  bool build() { return superglue_->build(); }

  void match_points(Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                    Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                    std::vector<cv::DMatch> &matches) {
    superglue_->matching_points(features0, features1, matches);
  }

private:
  std::unique_ptr<SuperGlue> superglue_;
};
SuperPointFeatureExtractor::SuperPointFeatureExtractor(
    const std::string &config_path, const std::string &model_path) {
  superpoint_impl_ = new SuperPointImpl(config_path, model_path);
}
SuperPointFeatureExtractor::~SuperPointFeatureExtractor() {
  delete superpoint_impl_;
}

bool SuperPointFeatureExtractor::Build() {
  if (!superpoint_impl_->build()) {
    std::cerr << "Error in SuperPoint building engine. Please check your onnx "
                 "model path."
              << std::endl;
    return false;
  }
  return true;
}

void SuperPointFeatureExtractor::Compute(
    const cv::Mat &image,
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
  if (!superpoint_impl_->infer(image, features)) {
    std::cerr << "Failed when extracting features from first image."
              << std::endl;
  }
}

SuperGlueFeatureMatcher::SuperGlueFeatureMatcher(
    const std::string &config_path, const std::string &model_path) {
  superglue_impl_ = new SuperGlueImpl(config_path, model_path);
}

SuperGlueFeatureMatcher::~SuperGlueFeatureMatcher() { delete superglue_impl_; }

bool SuperGlueFeatureMatcher::Build() {
  if (!superglue_impl_->build()) {
    std::cerr << "Error in SuperGlue building engine. Please check your onnx "
                 "model path."
              << std::endl;
    return false;
  }
  return true;
}

void SuperGlueFeatureMatcher::MatchPoints(
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
    std::vector<cv::DMatch> &matches) {
  superglue_impl_->match_points(features0, features1, matches);
}

} // namespace ubipose
