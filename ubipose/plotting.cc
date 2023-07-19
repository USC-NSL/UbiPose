#include "plotting.h"

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ubipose {

void PlotMatches(cv::Mat query_image, cv::Mat mesh_image, size_t timestamp,
                 const std::vector<ubipose::MatchedPoint> &matched_points) {
  std::vector<cv::KeyPoint> keypoints0;
  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::DMatch> matches;
  for (size_t i = 0; i < matched_points.size(); ++i) {
    keypoints0.push_back(matched_points[i].keypoint);
    keypoints1.push_back(matched_points[i].mesh_keypoint);
    matches.push_back(cv::DMatch(i, i, 0));
  }
  cv::Mat matchesimage;
  cv::drawMatches(query_image, keypoints0, mesh_image, keypoints1, matches,
                  matchesimage);
  cv::imwrite(std::to_string(timestamp) + "_used_matches.jpg", matchesimage);
}

void PlotInlierMatches(cv::Mat query_image, cv::Mat mesh_image,
                       size_t timestamp,
                       const std::vector<ubipose::MatchedPoint> &matched_points,
                       const std::vector<char> &inlier_mask) {
  std::vector<cv::KeyPoint> keypoints0;
  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::DMatch> matches;
  for (size_t i = 0; i < matched_points.size(); ++i) {
    keypoints0.push_back(matched_points[i].keypoint);
    keypoints1.push_back(matched_points[i].mesh_keypoint);
    matches.push_back(cv::DMatch(i, i, 0));
  }
  cv::Mat matchesimage;
  cv::drawMatches(query_image, keypoints0, mesh_image, keypoints1, matches,
                  matchesimage, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  inlier_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite(std::to_string(timestamp) + "_inlied_matches.jpg", matchesimage);
}

void PlotSuperglueMatches(
    cv::Mat query_image, cv::Mat mesh_image, size_t timestamp,
    size_t extrinsic_index,
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &query_keypoints,
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &frame_keypoints,
    const std::vector<cv::DMatch> &superglue_matches) {
  std::vector<cv::KeyPoint> keypoints0;
  for (Eigen::Index i = 0; i < query_keypoints.cols(); ++i) {
    double score = query_keypoints(0, i);
    double x = query_keypoints(1, i);
    double y = query_keypoints(2, i);
    keypoints0.emplace_back(x, y, 8, -1, score);
  }
  cv::Mat query_img_kps;
  cv::drawKeypoints(query_image, keypoints0, query_img_kps);
  cv::imwrite(std::to_string(timestamp) + "keypoints_query.png", query_img_kps);

  std::vector<cv::KeyPoint> keypoints1;
  for (Eigen::Index i = 0; i < frame_keypoints.cols(); ++i) {
    double score = frame_keypoints(0, i);
    double x = frame_keypoints(1, i);
    double y = frame_keypoints(2, i);
    keypoints1.emplace_back(x, y, 8, -1, score);
  }
  std::cout << "Number of keypoints " << keypoints1.size() << std::endl;

  cv::Mat match_image;

  cv::imwrite(std::to_string(timestamp) + "_images_" +
                  std::to_string(extrinsic_index) + ".png",
              mesh_image);
  cv::Mat img_kps;
  cv::drawKeypoints(mesh_image, keypoints1, img_kps);
  cv::imwrite(std::to_string(timestamp) + "_keypoints_" +
                  std::to_string(extrinsic_index) + ".png",
              img_kps);

  cv::Mat matchesimage;
  cv::drawMatches(query_image, keypoints0, mesh_image, keypoints1,
                  superglue_matches, matchesimage);
  cv::imwrite(std::to_string(timestamp) + "_matches_" +
                  std::to_string(extrinsic_index) + ".jpg",
              matchesimage);

  // std::cout << "Preprocess time in milliseconds: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  preprocess_end - preprocess_start)
  //                  .count()
  //           << " ms" << std::endl;

  // std::cout << "Superpoint time in milliseconds: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  keypoint_end - keypoint_start)
  //                  .count()
  //           << " ms" << std::endl;
  // std::cout << "Superglue time in milliseconds: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  matcher_end - matcher_start)
  //                  .count()
  //           << " ms" << std::endl;
  // std::cout << "Postprocess time in milliseconds: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  postprocess_end - postprocess_start)
  //                  .count()
  //           << " ms" << std::endl;
  // std::cout << "Total time in milliseconds: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  pipeline_end - pipeline_start)
  //                  .count()
  //           << " ms" << std::endl;
}

void RenderLocalizeOutput(ubipose::MeshRenderer *renderer,
                          const ubipose::EigenGl4f &extrinsic,
                          const ubipose::EigenGl4f &projection_matrix,
                          size_t timestamp, cv::Mat query_image,
                          const std::string &prefix) {
  auto [render_img, depth_img] = renderer->Render(extrinsic, projection_matrix);
  cv::Mat output_image;

  cv::Mat gray_query_img;
  cv::cvtColor(query_image, gray_query_img, cv::COLOR_RGBA2GRAY);
  cv::Mat gray_render_img;
  cv::cvtColor(render_img, gray_render_img, cv::COLOR_RGBA2GRAY);
  cv::hconcat(gray_query_img, gray_render_img, output_image);

  int dist = 20;
  for (int i = 0; i < output_image.rows; i += dist)
    cv::line(output_image, cv::Point(0, i), cv::Point(output_image.cols, i),
             cv::Scalar(100, 100, 100));

  for (int i = 0; i < output_image.cols; i += dist)
    cv::line(output_image, cv::Point(i, 0), cv::Point(i, output_image.rows),
             cv::Scalar(100, 100, 100));

  cv::imwrite(std::to_string(timestamp) + "_" + prefix + "_localized_.jpg",
              output_image);
  cv::imwrite(std::to_string(timestamp) + "_" + prefix + "_rendered.jpg",
              render_img);
}
} // namespace ubipose
