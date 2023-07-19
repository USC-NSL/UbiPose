#ifndef UBIPOSE_COLMAP_RECONSTRUCTION_H
#define UBIPOSE_COLMAP_RECONSTRUCTION_H

#include "types.h"
#include <string>
namespace colmap {
class Reconstruction;
}

namespace ubipose {

class ColmapReconstructionImpl;
// Provide an interface for our project to interact with the Colmap library

class ColmapReconstruction {
public:
  ColmapReconstruction();
  ~ColmapReconstruction();

  void Read(const std::string &path);
  void ParseColmapImageTimestamps(const std::string &prefix);
  void Transform(const EigenGl4f &transform);
  void Write(const std::string &path);
  std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  GetExtrinsicFromColmap(size_t image_timestamp);

private:
  ColmapReconstructionImpl *impl_;
};
} // namespace ubipose

#endif
