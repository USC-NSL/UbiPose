#include "colmap_utils.h"

#include <boost/filesystem.hpp>
#include <colmap/base/reconstruction.h>

namespace ubipose {

std::vector<ubipose::EigenGl4f, Eigen::aligned_allocator<ubipose::EigenGl4f>>
ReadColmapToMeshTransform(const std::string &pose_file) {
  std::ifstream ifile(pose_file);
  std::vector<
      Eigen::Matrix<float, 4, 4, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>>
      transforms;

  std::string line;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> m;
  int row = 0;

  while (std::getline(ifile, line)) {
    std::stringstream ss;
    ss << line;

    for (int i = 0; i < 4; i++) {
      double val;
      ss >> val;
      m(row, i) = val;
    }
    row++;

    if (row == 4) {
      transforms.push_back(m);
    }
    row = row % 4;
  }

  return transforms;
}
std::map<size_t, colmap::image_t>
ColmapTimestamps(const colmap::Reconstruction &reconstruction,
                 const std::string &prefix) {
  std::map<size_t, colmap::image_t> images_list;

  const auto &images = reconstruction.Images();
  for (const auto &image : images) {
    std::string img_rel_path = image.second.Name();
    if (img_rel_path.find(prefix) != 0) {
      continue;
    }
    boost::filesystem::path p(img_rel_path.substr(prefix.size()));
    auto stem = p.stem();

    std::string timestamp_string = stem.string();
    size_t timestamp = static_cast<size_t>(std::stod(timestamp_string) * 1e6);
    images_list.insert(std::make_pair(timestamp, image.first));
  }
  return images_list;
}
} // namespace ubipose
