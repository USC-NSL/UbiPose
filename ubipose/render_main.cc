
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <string>

#include "opencv2/imgcodecs.hpp"
#include "renderer.h"

std::vector<pipeline::EigenGl4f, Eigen::aligned_allocator<pipeline::EigenGl4f>>
ReadCameraExtrinsic(const std::string& pose_file)
{
    std::ifstream ifile(pose_file);
    std::vector<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>,
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

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    pipeline::MeshRenderer renderer("../data/mesh.vert", "../data/mesh.frag");

    renderer.InitEGL();

    renderer.LoadMesh("../data/sal_scaled_transformed/Mesh.obj");
    auto extrinsics = ReadCameraExtrinsic("../data/camera_extrinsic.txt");

    int i = 0;
    for (const auto& extrinsic : extrinsics) {
        std::cout << "Extrinsic:\n" << extrinsic << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        auto [output_img, depth_img] = renderer.Render(extrinsic);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Rendering takes " << duration.count() << " ms"<< std::endl;
        cv::imwrite("output_img_" + std::to_string(i) + ".png", output_img);
        i++;
    }
}
