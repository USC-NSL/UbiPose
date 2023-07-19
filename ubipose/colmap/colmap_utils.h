#ifndef UBIPOSE_COLMAP_UTILS_H
#define UBIPOSE_COLMAP_UTILS_H

#include <map>

#include <colmap/util/types.h>

#include "types.h"

namespace colmap {

class Reconstruction;
}

namespace ubipose {

std::vector<EigenGl4f, Eigen::aligned_allocator<EigenGl4f>>
ReadColmapToMeshTransform(const std::string &pose_file);

std::map<size_t, colmap::image_t>
ColmapTimestamps(const colmap::Reconstruction &reconstruction,
                 const std::string &prefix);

} // namespace ubipose

#endif
