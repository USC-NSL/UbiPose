#ifndef UBIPOSE_COLMAP_INITIAL_POSE_PROVIDER_H
#define UBIPOSE_COLMAP_INITIAL_POSE_PROVIDER_H

#include <memory>

#include "colmap/colmap_reconstruction.h"
#include "initial_pose_provider.h"

namespace ubipose {

class ColmapInitialPoseProviderImpl;

class ColmapInitialPoseProvider : public InitialPoseProvider {
public:
  ColmapInitialPoseProvider(ColmapReconstruction *reconstruction);

  std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetInitialExtrinsicAt(size_t image_timestamp) override;

private:
  ColmapReconstruction *reconstruction_;
};
} // namespace ubipose

#endif // !COLMAP_INITIAL_POSE_PROVIDER_H
