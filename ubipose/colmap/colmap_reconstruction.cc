#include "colmap_reconstruction.h"
#include "colmap_utils.h"
#include "types.h"

#include <boost/filesystem/operations.hpp>
#include <colmap/base/reconstruction.h>
#include <map>

namespace ubipose {

class ColmapReconstructionImpl {
public:
  ColmapReconstructionImpl() {}
  const colmap::Reconstruction &GetReconstruction();

  void Read(const std::string &path);
  void ParseColmapImageTimestamps(const std::string &prefix);
  void Transform(const EigenGl4f &transform);
  void Write(const std::string &path);

  std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetExtrinsicFromColmap(size_t image_timestamp);

private:
  colmap::Reconstruction reconstruction_;
  std::map<size_t, colmap::image_t> reconstruction_images_;
};

void ColmapReconstructionImpl::Read(const std::string &path) {
  reconstruction_.Read(path);
}

void ColmapReconstructionImpl::ParseColmapImageTimestamps(
    const std::string &prefix) {
  reconstruction_images_ = ColmapTimestamps(reconstruction_, prefix);
}

void ColmapReconstructionImpl::Transform(const EigenGl4f &transform) {
  Eigen::Matrix3x4d transform_matrix =
      transform.block<3, 4>(0, 0).cast<double>();
  colmap::SimilarityTransform3 sim_transform(transform_matrix);
  reconstruction_.Transform(sim_transform);
}

void ColmapReconstructionImpl::Write(const std::string &path) {
  if (!path.empty()) {
    if (!boost::filesystem::exists(path)) {
      boost::filesystem::create_directory(path);
    }
    reconstruction_.WriteText(path);
  }
}
std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
ColmapReconstructionImpl::GetExtrinsicFromColmap(size_t image_timestamp) {
  auto image_it = reconstruction_images_.lower_bound(image_timestamp);
  if (image_it == reconstruction_images_.end()) {
    return std::nullopt;
  }
  if (image_it->first - image_timestamp > 1e5) {
    return std::nullopt;
  }

  auto image_id = image_it->second;
  auto sfm_qvec = reconstruction_.Image(image_id).Qvec();
  auto sfm_tvec = reconstruction_.Image(image_id).Tvec();

  return std::pair{sfm_qvec, sfm_tvec};
}

ColmapReconstruction::ColmapReconstruction()
    : impl_(new ColmapReconstructionImpl) {}

ColmapReconstruction::~ColmapReconstruction() { delete impl_; }

void ColmapReconstruction::Read(const std::string &path) { impl_->Read(path); }

void ColmapReconstruction::ParseColmapImageTimestamps(const std::string &path) {
  impl_->ParseColmapImageTimestamps(path);
}

void ColmapReconstruction::Transform(const EigenGl4f &transform) {
  impl_->Transform(transform);
}
void ColmapReconstruction::Write(const std::string &path) {
  impl_->Write(path);
}

std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
ColmapReconstruction::GetExtrinsicFromColmap(size_t image_timestamp) {
  return impl_->GetExtrinsicFromColmap(image_timestamp);
}

} // namespace ubipose
