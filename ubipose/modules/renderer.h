#ifndef UBIPOSE_MODULES_RENDERER_H
#define UBIPOSE_MODULES_RENDERER_H

#include <filesystem>
#include <unordered_map>

#include <Eigen/Dense>
#include <assimp/Importer.hpp>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <opencv2/core/mat.hpp>

#define EGL_NO_X11

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "types.h"
namespace ubipose {

struct Vertex {
  float x, y, z;    // Vertex
  float nx, ny, nz; // Normal
  float s0, t0;     // Texcoord0
};

class Mesh {
public:
  Mesh(aiMesh *mesh);
  void AddToContext();
  void Bind();
  void UnBind();
  size_t numFaces() { return indices_.size(); }

  EigenGl4f pose_ = EigenGl4f::Identity();

private:
  aiMesh *mesh_;

  std::vector<Vertex> vertices_array_buf_;

  GLuint vaid_;
  GLuint vertex_buffer_;
  GLuint model_buffer_;

  std::vector<uint32_t> indices_;
  GLuint element_buffer_;
};

class Texture {
public:
  Texture(unsigned char *data, int width, int height, int channels);

  void AddToContext();
  void Bind();
  void UnBind();

private:
  unsigned char *data_;
  int width_;
  int height_;
  int channels_;

  GLuint texid_;

public:
  const Eigen::Vector3f emisive_factor_;
  const Eigen::Vector4f base_color_factor_;
  const float metallic_factor_;
  const float roughness_factor_;
  const int tex_alloc_idx_ = 0;
};

class Program {
public:
  Program(const std::string &vertex_shader, const std::string &fragment_shader);

  void AddToContext();
  void Bind();
  void UnBind();
  GLuint program() { return program_; }

private:
  const std::filesystem::path vertex_shader_path_;
  const std::filesystem::path fragment_shader_path_;
  GLuint program_;
};

class MeshRenderer {
public:
  MeshRenderer(int width, int height, const std::string &vertex_shader,
               const std::string &fragment_shader);

  void InitEGL();
  void LoadMesh(const std::string &mesh_file);
  void UpdateContext();

  std::pair<cv::Mat, cv::Mat> Render(const EigenGl4f &camera_extrinsic,
                                     const EigenGl4f &camera_projection_matrix);

private:
  void ConfigureMainFramebuffer();
  void ConfigureForwardPassViewport();

  void ForwardPass(const EigenGl4f &camera_extrinsic,
                   const EigenGl4f &camera_projection_matrix);
  void RenderAMesh(Mesh &mesh, Texture &texture, const EigenGl4f &v_matrix,
                   const EigenGl4f &p_matrix,
                   const Eigen::Vector3f &camera_position,
                   const Eigen::Vector3f &ambient_light);
  void BindAndDrawPrimitive(Mesh &mesh, Texture &texture);

  std::pair<cv::Mat, cv::Mat> ReadMainBuffers();

  const int VIEWPORT_WIDTH;
  const int VIEWPORT_HEIGHT;

  Program program_;
  Assimp::Importer importer_;
  const aiScene *scene_ = NULL;
  std::unordered_map<aiMesh *, Mesh> mesh_map_;
  std::unordered_map<aiMesh *, Texture> texture_map_;

  // Color buffer and depth buffer
  GLuint main_cb = 0;
  GLuint main_db = 0;

  GLuint main_fb = 0;

  GLuint main_cb_ms = 0, main_db_ms = 0;
  GLuint main_fb_ms = 0;
};
} // namespace ubipose

#endif
