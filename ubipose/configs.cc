#include "configs.h"

#include <Eigen/Dense>
#include <iostream>

#include <absl/log/log.h>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

namespace ubipose {

std::optional<UbiposeConfigs>
ReadMeshlocConfigs(const std::string &config_path) {
  if (!boost::filesystem::exists(config_path)) {
    LOG(ERROR) << "Config file doesn't exists";
    return std::nullopt;
  }
  YAML::Node file_node = YAML::LoadFile(config_path);

  YAML::Node controller_node = file_node["controller"];

  ubipose::UbiposeConfigs meshloc_config;
  meshloc_config.image_height = controller_node["image_height"].as<int>();
  meshloc_config.image_width = controller_node["image_width"].as<int>();

  YAML::Node initial_pose_node = controller_node["initial_pose"];
  if (initial_pose_node) {
    meshloc_config.initial_pose = ConfigInitialPose{
        .image_timestamp = initial_pose_node["image_timestamp"].as<size_t>(),
        .qvec =
            Eigen::Vector4d{
                initial_pose_node["qvec"]["qw"].as<double>(),
                initial_pose_node["qvec"]["qx"].as<double>(),
                initial_pose_node["qvec"]["qy"].as<double>(),
                initial_pose_node["qvec"]["qz"].as<double>(),
            },
        .tvec = Eigen::Vector3d{
            initial_pose_node["tvec"]["tx"].as<double>(),
            initial_pose_node["tvec"]["ty"].as<double>(),
            initial_pose_node["tvec"]["tz"].as<double>(),
        }};
  }

  meshloc_config.vertex_file = controller_node["vertex_file"].as<std::string>();
  meshloc_config.fragment_file =
      controller_node["fragment_file"].as<std::string>();
  meshloc_config.mesh_file = controller_node["mesh_file"].as<std::string>();

  meshloc_config.reconstruction_path =
      controller_node["reconstruction_path"].as<std::string>();
  meshloc_config.transform_to_mesh_path =
      controller_node["transform_to_mesh_path"].as<std::string>();
  meshloc_config.colmap_image_prefix =
      controller_node["colmap_image_prefix"].as<std::string>();
  meshloc_config.output_transformed_colmap_path =
      controller_node["output_transformed_colmap_path"].as<std::string>();

  meshloc_config.model_config_file =
      controller_node["model_config_file"].as<std::string>();
  meshloc_config.model_weight_folder =
      controller_node["model_weight_folder"].as<std::string>();

  meshloc_config.method = controller_node["method"].as<int>();

  meshloc_config.do_early_exit = controller_node["do_early_exit"].as<bool>();
  meshloc_config.early_exit_num_matches =
      controller_node["early_exit_num_matches"].as<int>();

  meshloc_config.strong_inlier_ratio =
      controller_node["strong_inlier_ratio"].as<double>();
  meshloc_config.strong_superglue_matches =
      controller_node["strong_superglue_matches"].as<int>();
  meshloc_config.strong_error_R =
      controller_node["strong_error_R"].as<double>();
  meshloc_config.strong_error_t =
      controller_node["strong_error_t"].as<double>();

  meshloc_config.weak_inlier_ratio =
      controller_node["weak_inlier_ratio"].as<double>();
  meshloc_config.weak_error_R = controller_node["weak_error_R"].as<double>();
  meshloc_config.weak_error_t = controller_node["weak_error_t"].as<double>();

  meshloc_config.num_vio_sec_before_loss =
      controller_node["num_vio_sec_before_loss"].as<double>();

  meshloc_config.perform_localization =
      controller_node["perform_localization"].as<bool>();
  meshloc_config.output_images = controller_node["output_images"].as<bool>();
  meshloc_config.output_images_folder =
      controller_node["output_images_folder"].as<std::string>();

  meshloc_config.output_localization_path =
      controller_node["output_localization_path"].as<std::string>();
  meshloc_config.output_stats_path =
      controller_node["output_stats_path"].as<std::string>();
  meshloc_config.output_mesh_pose_path =
      controller_node["output_mesh_pose_file"].as<std::string>();
  meshloc_config.debugging = controller_node["debugging"].as<bool>();

  return meshloc_config;
}

} // namespace ubipose
