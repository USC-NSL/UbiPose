#ifndef UBIPOSE_INITIAL_POSE_PROVIDER_H
#define UBIPOSE_INITIAL_POSE_PROVIDER_H

#include <memory>
#include <optional>

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

std::unique_ptr<InitialPoseProvider>
CreateInitialPoseProvider(const UbiposeConfigs &config,
                          ColmapReconstruction *reconstruction);
} // namespace ubipose

#endif // !INITIAL_POSE_PROVIDER_H
