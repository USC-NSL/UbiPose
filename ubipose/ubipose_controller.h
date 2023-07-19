#ifndef UBIPOSE_CONTROLLER_H
#define UBIPOSE_CONTROLLER_H

#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

#include <colmap/util/types.h>

#include "colmap/colmap_reconstruction.h"
#include "configs.h"
#include "initial_pose_provider.h"
#include "modules/image_registrator.h"
#include "modules/renderer.h"
#include "modules/superpointglue.h"
#include "modules/unprojector.h"
#include "types.h"

namespace ubipose {

class UbiposeController {
public:
  struct UbiposeQuery {
    cv::Mat image;
    size_t image_timestamp;
    size_t vio_timestamp;
    Eigen::Vector4d vio_qvec;
    Eigen::Vector3d vio_tvec;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> intrinsic;
    std::string original_filename;
  };

  UbiposeController(const UbiposeConfigs &configs);
  ~UbiposeController();

  bool Init();

  void Localize(const UbiposeQuery &query);

  void Stop();

private:
  struct UbiposeResult {
    cv::Mat image;
    size_t image_timestamp;
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    std::string original_filename;
  };

  // std::optional<std::pair<Eigen::Vector4d, Eigen::Vector3d>>
  // GetExtrinsicFromColmap(size_t image_timestamp);

  std::pair<Eigen::Vector4d, Eigen::Vector3d>
  InitializeLocalize(const UbiposeQuery &query);
  std::pair<Eigen::Vector4d, Eigen::Vector3d>
  LocalizeFrame(const UbiposeQuery &query);

  void GetVioRelativePose(const Eigen::Vector4d &vio_qvec,
                          const Eigen::Vector3d &vio_tvec,
                          Eigen::Vector4d *rel_qvec, Eigen::Vector3d *rel_tvec);

  void AddImageToQueue(cv::Mat image, size_t image_timestamp,
                       const Eigen::Vector4d &qvec, const Eigen::Vector3d &tvec,
                       const std::string &original_filename);
  void ProcessResultQueue();
  bool AcceptResult(double est_error_R, double est_error_t,
                    const UbiposeStats &stats);
  bool AcceptCacheLocalizeResult(double est_error_R, double est_error_t,
                                 const UbiposeStats &stats);
  ;
  bool
  ShouldEarlyExit(const std::vector<Eigen::Vector2d> &tri_points2D,
                  const std::vector<Eigen::Vector3d> &tri_points3D,
                  const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic,
                  cv::Mat query_image);

  std::pair<std::vector<MapPoint *>, std::vector<cv::KeyPoint>>
  MapPointsInImage(
      const Eigen::Vector4d &qvec, const Eigen::Vector3d &tvec, size_t width,
      size_t height,
      const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &intrinsic);

  void PurgeBadMapPoints();

  UbiposeConfigs configs_;
  std::unique_ptr<ColmapReconstruction> colmap_reconstruction_;

  std::unique_ptr<MeshRenderer> renderer_;
  std::unique_ptr<Unprojector> unprojector_;

  std::unique_ptr<ImageRegistrator> image_registrator_;
  std::unique_ptr<InitialPoseProvider> initial_pose_provider_;

  std::unique_ptr<SuperPointFeatureExtractor> superpoint_;
  std::unique_ptr<SuperGlueFeatureMatcher> superglue_;

  std::map<size_t, colmap::image_t> reconstruction_images_;

  bool initialized_ = false;
  size_t prev_query_image_timestamp_ = 0;

  size_t prev_localized_vio_timestamp_;
  Eigen::Vector4d prev_localized_vio_qvec_;
  Eigen::Vector3d prev_localized_vio_tvec_;

  size_t prev_localized_image_timestamp_;
  Eigen::Vector4d prev_localized_meshloc_qvec_;
  Eigen::Vector3d prev_localized_meshloc_tvec_;

  Eigen::Vector4d prev_sfm_qvec_;
  Eigen::Vector3d prev_sfm_tvec_;
  bool last_frame_use_vio_;

  std::vector<MapPoint> map_points_;

  std::thread *output_thread = nullptr;
  std::mutex mu_;
  std::queue<UbiposeResult> result_queue_;
  bool stoping_ = false;

  std::ofstream output_stats_file;
  std::ofstream output_mesh_pose_file;
};
} // namespace ubipose
#endif
