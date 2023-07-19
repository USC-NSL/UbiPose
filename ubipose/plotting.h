
#ifndef UBIPOSE_PLOTTING_H
#define UBIPOSE_PLOTTING_H

#include "opencv2/core.hpp"

#include "modules/renderer.h"
#include "types.h"

namespace ubipose {

void PlotMatches(cv::Mat query_image, cv::Mat mesh_image, size_t timestamp,
                 const std::vector<ubipose::MatchedPoint> &matched_points);

void PlotInlierMatches(cv::Mat query_image, cv::Mat mesh_image,
                       size_t timestamp,
                       const std::vector<ubipose::MatchedPoint> &matched_points,
                       const std::vector<char> &inlier_mask);

void PlotSuperglueMatches(
    cv::Mat query_image, cv::Mat mesh_image, size_t timestamp,
    size_t extrinsic_index,
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &query_keypoints,
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &frame_keypoints,
    const std::vector<cv::DMatch> &superglue_matches);

void RenderLocalizeOutput(ubipose::MeshRenderer *renderer,
                          const ubipose::EigenGl4f &extrinsic,
                          const ubipose::EigenGl4f &projection_matrix,
                          size_t timestamp, cv::Mat query_image,
                          const std::string &prefix);

} // namespace ubipose
#endif
