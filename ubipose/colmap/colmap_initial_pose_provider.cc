#include "colmap_initial_pose_provider.h"

#include <map>
#include <memory>

#include <boost/filesystem/operations.hpp>
#include <colmap/base/reconstruction.h>
#include <glog/logging.h>

#include "colmap/colmap_reconstruction.h"
#include "colmap_utils.h"

namespace ubipose {

class ColmapInitialPoseProviderImpl {
public:
  ColmapInitialPoseProviderImpl() {}

  bool Init(const std::string &reconstruction_path,
            const std::string &colmap_image_prefix,
            const std::string &transform_to_mesh_path,
            const std::string &output_transformed_colmap_path);

  std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetExtrinsicFromColmap(size_t image_timestamp);

private:
  colmap::Reconstruction reconstruction_;
  std::map<size_t, colmap::image_t> reconstruction_images_;
};

bool ColmapInitialPoseProviderImpl::Init(
    const std::string &reconstruction_path,
    const std::string &colmap_image_prefix,
    const std::string &transform_to_mesh_path,
    const std::string &output_transformed_colmap_path) {

  reconstruction_.Read(reconstruction_path);
  reconstruction_images_ =
      ColmapTimestamps(reconstruction_, colmap_image_prefix);
  if (reconstruction_images_.empty()) {
    LOG(FATAL)
        << "Cannot find images with expected timestamp format and prefix: "
        << colmap_image_prefix;
    return false;
  }

  auto transforms = ReadColmapToMeshTransform(transform_to_mesh_path);
  if (transforms.size() != 1) {
    LOG(FATAL) << "Invalid transform read from " << transform_to_mesh_path;
    return false;
  }
  auto transform = transforms[0];
  Eigen::Matrix3x4d transform_matrix =
      transform.block<3, 4>(0, 0).cast<double>();
  colmap::SimilarityTransform3 sim_transform(transform_matrix);
  reconstruction_.Transform(sim_transform);

  if (!output_transformed_colmap_path.empty()) {
    if (!boost::filesystem::exists(output_transformed_colmap_path)) {
      boost::filesystem::create_directory(output_transformed_colmap_path);
    }
    reconstruction_.WriteText(output_transformed_colmap_path);
  }

  return true;
}

std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
ColmapInitialPoseProviderImpl::GetExtrinsicFromColmap(size_t image_timestamp) {
  auto image_it = reconstruction_images_.lower_bound(image_timestamp);
  if (image_it == reconstruction_images_.end()) {
    return std::nullopt;
  }
  if (image_it->first - image_timestamp > 1e5) {
    return std::nullopt;
  }

  auto image_id = image_it->second;
  std::cout << "Found image " << reconstruction_.Image(image_id).Name()
            << std::endl;
  auto sfm_qvec = reconstruction_.Image(image_id).Qvec();
  auto sfm_tvec = reconstruction_.Image(image_id).Tvec();

  return std::pair{sfm_qvec, sfm_tvec};
}

ColmapInitialPoseProvider::ColmapInitialPoseProvider(
    ColmapReconstruction *reconstruction)
    : reconstruction_(reconstruction) {}

std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
ColmapInitialPoseProvider::GetInitialExtrinsicAt(size_t image_timestamp) {
  return reconstruction_->GetExtrinsicFromColmap(image_timestamp);
}

} // namespace ubipose
