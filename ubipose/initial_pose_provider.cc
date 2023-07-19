#include "initial_pose_provider.h"

#include <memory>

#include <absl/log/check.h>

#include "colmap/colmap_initial_pose_provider.h"
#include "colmap/colmap_reconstruction.h"

namespace ubipose {

std::unique_ptr<InitialPoseProvider>
CreateInitialPoseProvider(const UbiposeConfigs &config,
                          ColmapReconstruction *reconstruction) {

  auto provider = std::make_unique<ColmapInitialPoseProvider>(reconstruction);

  return provider;
}
} // namespace ubipose
