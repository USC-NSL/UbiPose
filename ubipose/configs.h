#ifndef PIPELINE_CONFIGS_H
#define PIPELINE_CONFIGS_H

#include <optional>
#include <string>

#include <Eigen/Dense>
namespace ubipose {

struct ConfigInitialPose {
  size_t image_timestamp;
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
};

struct UbiposeConfigs {
  int image_height;
  int image_width;

  std::optional<ConfigInitialPose> initial_pose;

  // Renderer
  std::string vertex_file;
  std::string fragment_file;

  std::string mesh_file;

  // Colmap
  std::string reconstruction_path;
  std::string colmap_image_prefix;
  std::string transform_to_mesh_path;
  std::string output_transformed_colmap_path;

  // superpoint/superglue
  std::string model_config_file;
  std::string model_weight_folder;

  int method;

  // Early exit
  bool do_early_exit;
  int early_exit_num_matches;

  // Thresholds

  // For strong result
  double strong_inlier_ratio;
  int strong_superglue_matches;
  double strong_error_R;
  double strong_error_t;

  // For weak result
  double weak_inlier_ratio;
  double weak_error_R;
  double weak_error_t;

  int num_vio_sec_before_loss;

  // debug / evaluate related
  bool perform_localization = true;
  bool output_images = true;
  std::string output_localization_path;
  std::string output_images_folder;
  std::string output_stats_path;
  std::string output_mesh_pose_path;
  bool debugging;
};

std::optional<UbiposeConfigs>
ReadMeshlocConfigs(const std::string &config_path);

} // namespace ubipose

#endif // !PIPELINE_CONFIGS_H
