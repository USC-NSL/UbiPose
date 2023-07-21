#include "initial_pose_provider.h"

#include <memory>
#include <optional>

#include <absl/log/check.h>

#include "colmap/colmap_initial_pose_provider.h"
#include "colmap/colmap_reconstruction.h"

namespace ubipose {

std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
ConfigInitialPoseProvider::GetInitialExtrinsicAt(size_t image_timestamp) {
  if (image_timestamp != image_timestamp_) {
    return std::nullopt;
  }
  return std::pair{qvec_, tvec_};
}

std::unique_ptr<InitialPoseProvider>
CreateInitialPoseProvider(const UbiposeConfigs &config,
                          ColmapReconstruction *reconstruction) {
  if (config.initial_pose.has_value()) {
    return std::make_unique<ConfigInitialPoseProvider>(
        config.initial_pose->image_timestamp, config.initial_pose->qvec,
        config.initial_pose->tvec);
  }

  return std::make_unique<ColmapInitialPoseProvider>(reconstruction);
}
} // namespace ubipose
