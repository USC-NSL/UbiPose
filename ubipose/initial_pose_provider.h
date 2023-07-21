#ifndef UBIPOSE_INITIAL_POSE_PROVIDER_H
#define UBIPOSE_INITIAL_POSE_PROVIDER_H

#include <cstddef>
#include <memory>
#include <optional>

#include <Eigen/Dense>

#include "colmap/colmap_reconstruction.h"
#include "configs.h"
#include "types.h"

namespace ubipose {

class InitialPoseProvider {
public:
  InitialPoseProvider() {}
  virtual std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetInitialExtrinsicAt(size_t image_timestamp) = 0;
};

class ConfigInitialPoseProvider : public InitialPoseProvider {
public:
  ConfigInitialPoseProvider(size_t image_timestamp, Eigen::Vector4d qvec,
                            Eigen::Vector3d tvec)
      : image_timestamp_(image_timestamp), qvec_(qvec), tvec_(tvec) {}
  std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetInitialExtrinsicAt(size_t image_timestamp) override;

private:
  size_t image_timestamp_;
  Eigen::Vector4d qvec_;
  Eigen::Vector3d tvec_;
};

std::unique_ptr<InitialPoseProvider>
CreateInitialPoseProvider(const UbiposeConfigs &config,
                          ColmapReconstruction *reconstruction);
} // namespace ubipose

#endif
