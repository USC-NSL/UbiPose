#include "renderer.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <absl/log/log.h>
#include <assimp/Importer.hpp>
#include <assimp/cimport.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace {
void CheckEGLErrorAt(const std::string &msg) {
  EGLint error = eglGetError();
  if (error != EGL_SUCCESS) {
    LOG(FATAL) << "EGL error 0x" << std::hex << error << " at " << msg;
  }
}

void CheckOpenGLError(const std::string &msg) {
  GLenum error = glGetError();

  if (error != GL_NO_ERROR) {
    LOG(FATAL) << "OpenGL error 0x" << std::hex << error << " at " << msg;
  }
}

std::unordered_map<aiMesh *, ubipose::Texture>
LoadTexture(const std::string &modelpath, const aiScene *scene) {
  stbi_set_flip_vertically_on_load(true);

  std::vector<std::string> texture_files; // map image filenames to textureIds
  std::unordered_map<aiMesh *, ubipose::Texture> textures;

  std::filesystem::path model_path(modelpath);

  std::filesystem::path basepath = model_path.parent_path().string();
  for (size_t i = 0; i < scene->mNumMeshes; i++) {
    auto *mesh = scene->mMeshes[i];
    auto material_index = mesh->mMaterialIndex;

    int tex_index = 0;
    aiString path;

    aiReturn texFound = AI_SUCCESS;
    while (texFound == AI_SUCCESS) {
      texFound = scene->mMaterials[material_index]->GetTexture(
          aiTextureType_DIFFUSE, tex_index, &path);
      if (texFound != AI_SUCCESS) {
        break;
      }
      std::string filename = path.data;          // get filename
      std::string fileloc = basepath / filename; /* Loading of image */
      int x, y, n;
      unsigned char *data =
          stbi_load(fileloc.c_str(), &x, &y, &n, STBI_rgb_alpha);
      if (data != nullptr) {
        textures.insert({mesh, {data, x, y, n}});
      } else {
        LOG(FATAL) << "Texture not found " << fileloc;
      }

      tex_index++;
    }
    if (tex_index > 1) {
      LOG(FATAL) << "More than 1 texture for the mesh";
    }
  }

  return textures;
}

std::string ReadShaderSource(const std::filesystem::path &path) {
  std::ifstream ifile(path);
  std::stringstream buffer;
  buffer << ifile.rdbuf();
  return buffer.str();
}

bool CheckShaderCompiled(GLuint shader) {
  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> errorLog(maxLength);
    glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

    // Provide the infolog in whatever manor you deem best.
    // Exit with failure.
    glDeleteShader(shader); // Don't leak the shader.
    return false;
  }
  return true;
}

bool CheckProgramLinked(GLuint program) {
  // Note the different functions here: glGetProgram* instead of glGetShader*.
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
  if (isLinked == GL_FALSE) {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> infoLog(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
    std::string error_msg(infoLog.begin(), infoLog.end());
    LOG(ERROR) << error_msg << std::endl;
    // We don't need the program anymore.
    glDeleteProgram(program);

    return false;
  }
  return true;
}

bool CheckValidateProgram(GLuint program) {
  GLint validated = 1;

  glValidateProgram(program);
  glGetProgramiv(program, GL_VALIDATE_STATUS, &validated);
  if (validated == GL_FALSE) {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> errorLog(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]);

    std::string error_msg(errorLog.begin(), errorLog.end());
    LOG(ERROR) << error_msg << std::endl;
    return false;
  }
  return true;
}

// pipeline::EigenGl4f ConstructPMatrix()
// {
//     pipeline::EigenGl4f p_matrix;

//     p_matrix << 1.00176887, 0., 0., 0., 0., 1.76979167, 0., 0., 0., 0.,
//     -1.0010005, -0.20010005, 0., 0., -1., 0.;
//     // std::cout << "P matrix: \n" << p_matrix << std::endl;
//     return p_matrix;
// }

} // namespace

namespace ubipose {

#define BUFFER_OFFSET(i) ((char *)NULL + (i))
constexpr int kMaxEglDevice = 32;
constexpr int kDeviceIndex = 1;
constexpr int kNumAttrib = 3;
const int kBorderColor[4] = {255, 255, 255, 255};
// constexpr int VIEWPORT_WIDTH = 640;
// constexpr int VIEWPORT_HEIGHT = 480;
const int n_directional_lights = 0;
const int n_spot_lights = 0;
const int n_point_lights = 0;

static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                       EGL_PBUFFER_BIT,
                                       EGL_BLUE_SIZE,
                                       8,
                                       EGL_GREEN_SIZE,
                                       8,
                                       EGL_RED_SIZE,
                                       8,
                                       EGL_DEPTH_SIZE,
                                       24,
                                       EGL_COLOR_BUFFER_TYPE,
                                       EGL_RGB_BUFFER,
                                       EGL_RENDERABLE_TYPE,
                                       EGL_OPENGL_BIT,
                                       EGL_CONFORMANT,
                                       EGL_OPENGL_BIT,
                                       EGL_NONE};
static const EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION,
                                        4,
                                        EGL_CONTEXT_MINOR_VERSION,
                                        1,
                                        EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                        EGL_NONE};

MeshRenderer::MeshRenderer(int width, int height,
                           const std::string &vertex_shader,
                           const std::string &fragment_shader)
    : VIEWPORT_WIDTH(width), VIEWPORT_HEIGHT(height),
      program_(vertex_shader, fragment_shader) {}

void MeshRenderer::InitEGL() {
  EGLDeviceEXT devices[kMaxEglDevice];

  // load the function pointers for the device,platform extensions
  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
  if (!eglQueryDevicesEXT) {
    LOG(FATAL) << "Cannot load the PFN for query device extention";
  }

  // load the function pointers for the device,platform extensions
  PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress(
          "eglQueryDeviceStringEXT");
  if (!eglQueryDeviceStringEXT) {
    LOG(FATAL) << "Cannot load the PFN for query device string";
  }

  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
          "eglGetPlatformDisplayEXT");
  if (!eglGetPlatformDisplayEXT) {
    LOG(FATAL) << "Cannot load the PFN for query platform display ext";
  }

  EGLint numDevices;
  eglQueryDevicesEXT(kMaxEglDevice, devices, &numDevices);

  for (int i = 0; i < numDevices; i++) {
    const char *device_str =
        eglQueryDeviceStringEXT(devices[i], EGL_EXTENSIONS);
    std::cerr << "found device " << device_str << std::endl;
  }
  EGLDisplay eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                               devices[kDeviceIndex], 0);
  CheckEGLErrorAt("eglGetPlatformDisplayEXT");

  EGLint major, minor;
  eglInitialize(eglDpy, &major, &minor);
  CheckEGLErrorAt("eglInitialize");

  // Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;
  eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
  CheckEGLErrorAt("eglChooseConfig");

  // Bind the API
  eglBindAPI(EGL_OPENGL_API);
  CheckEGLErrorAt("eglBindAPI");

  // Create a context and make it current
  EGLContext eglCtx =
      eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, contextAttribs);
  CheckEGLErrorAt("eglCreateContext");
  eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
  CheckEGLErrorAt("eglMakeCurrent");
}

void MeshRenderer::LoadMesh(const std::string &mesh_file) {
  if (scene_ != nullptr) {
    LOG(FATAL) << "Already has mesh";
  }

  aiLogStream stream;
  stream = aiGetPredefinedLogStream(aiDefaultLogStream_STDOUT, NULL);
  aiAttachLogStream(&stream);

  // stream = aiGetPredefinedLogStream(aiDefaultLogStream_FILE,
  // "assimp_log.txt"); aiAttachLogStream(&stream);

  scene_ =
      importer_.ReadFile(mesh_file, aiProcessPreset_TargetRealtime_MaxQuality);
  if (scene_ == nullptr) {
    LOG(FATAL) << "Failed to import mesh " << mesh_file;
  }

  // Load meshes in our data structures
  aiMesh **meshes = scene_->mMeshes;
  for (size_t i = 0; i < scene_->mNumMeshes; i++) {
    mesh_map_.insert({meshes[i], meshes[i]});

    for (size_t j = 0; j < meshes[i]->mNumFaces; j++) {
      auto &face = meshes[i]->mFaces[j];
      if (face.mNumIndices != 3) {
        LOG(ERROR) << "Unexpected: number of indices = " << face.mNumIndices
                   << " num faces = " << meshes[i]->mNumFaces
                   << " face index = " << j << " mesh id = " << i;
      }
    }
  }

  // Load texture in our data structures
  texture_map_ = LoadTexture(mesh_file, scene_);
}

void MeshRenderer::UpdateContext() {
  for (auto &it : mesh_map_) {
    it.second.AddToContext();
  }
  for (auto &it : texture_map_) {
    it.second.AddToContext();
  }

  program_.AddToContext();
}

std::pair<cv::Mat, cv::Mat> MeshRenderer::ReadMainBuffers() {
  glBindFramebuffer(GL_READ_FRAMEBUFFER, main_fb_ms);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, main_fb);
  glBlitFramebuffer(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 0, 0, VIEWPORT_WIDTH,
                    VIEWPORT_HEIGHT, GL_COLOR_BUFFER_BIT, GL_LINEAR);
  glBlitFramebuffer(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, 0, 0, VIEWPORT_WIDTH,
                    VIEWPORT_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, main_fb);

  cv::Mat depth_mat(VIEWPORT_HEIGHT, VIEWPORT_WIDTH, CV_32FC1, cv::Scalar(0));
  glReadPixels(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, GL_DEPTH_COMPONENT,
               GL_FLOAT, depth_mat.data);
  CheckOpenGLError("After read depth pixels");

  cv::Mat color_mat(VIEWPORT_HEIGHT, VIEWPORT_WIDTH, CV_8UC3,
                    cv::Scalar(255, 255, 255, 0));
  glReadPixels(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE,
               color_mat.data);
  CheckOpenGLError("After read color pixels");

  cv::Mat flipped_depth_mat;
  cv::flip(depth_mat, flipped_depth_mat, 0);

  cv::Mat flipped_color_mat;
  cv::flip(color_mat, flipped_color_mat, 0);

  cv::Mat cvt_color_mat;
  cv::cvtColor(flipped_color_mat, cvt_color_mat, cv::COLOR_RGB2BGRA);
  return {cvt_color_mat, flipped_depth_mat};
}

std::pair<cv::Mat, cv::Mat>
MeshRenderer::Render(const EigenGl4f &camera_extrinsic,
                     const EigenGl4f &camera_projection_matrix) {
  UpdateContext();
  CheckOpenGLError("Update Context");

  ForwardPass(camera_extrinsic, camera_projection_matrix);

  return ReadMainBuffers();
}

void MeshRenderer::ForwardPass(const EigenGl4f &camera_extrinsic,
                               const EigenGl4f &camera_projection_matrix) {
  ConfigureForwardPassViewport();
  CheckOpenGLError("Configure forward pass view port");

  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  CheckOpenGLError("Clear Color");

  glEnable(GL_MULTISAMPLE);

  EigenGl4f v_matrix = camera_extrinsic;
  EigenGl4f p_matrix = camera_projection_matrix;
  Eigen::Vector3f camera_position =
      -camera_extrinsic.block<3, 3>(0, 0).transpose() *
      camera_extrinsic.block<3, 1>(0, 3);
  Eigen::Vector3f ambient_light(1, 1, 1);
  program_.Bind();

  // 4. For each mesh in the scene, render
  for (auto &it : mesh_map_) {
    auto textures_it = texture_map_.find(it.first);
    if (textures_it == texture_map_.end()) {
      LOG(FATAL) << "Cannot find texture for mesh";
    }
    RenderAMesh(it.second, textures_it->second, v_matrix, p_matrix,
                camera_position, ambient_light);
    CheckOpenGLError("After render a mesh");
  }

  program_.UnBind();

  glFlush();
}

void MeshRenderer::RenderAMesh(Mesh &mesh, Texture &texture,
                               const EigenGl4f &v_matrix,
                               const EigenGl4f &p_matrix,
                               const Eigen::Vector3f &camera_position,
                               const Eigen::Vector3f &ambient_light) {
  if (mesh.numFaces() == 0) {
    return;
  }
  // Set camera uniforms
  GLint v_loc = glGetUniformLocation(program_.program(), "V");
  glUniformMatrix4fv(v_loc, 1, GL_TRUE, v_matrix.data());

  GLint p_loc = glGetUniformLocation(program_.program(), "P");
  glUniformMatrix4fv(p_loc, 1, GL_TRUE, p_matrix.data());

  GLint cam_pos_loc = glGetUniformLocation(program_.program(), "cam_pos");
  glUniform3fv(cam_pos_loc, 1, camera_position.data());

  CheckOpenGLError("Set camera uniforms");

  // Set lightings
  GLint ambient_light_loc =
      glGetUniformLocation(program_.program(), "ambient_light");
  glUniform3fv(ambient_light_loc, 1, ambient_light.data());

  GLint n_directional_lights_loc =
      glGetUniformLocation(program_.program(), "n_directional_lights");
  glUniform1iv(n_directional_lights_loc, 1, &n_directional_lights);

  GLint n_spot_lights_loc =
      glGetUniformLocation(program_.program(), "n_spot_lights");
  glUniform1iv(n_spot_lights_loc, 1, &n_spot_lights);

  GLint n_point_lights_loc =
      glGetUniformLocation(program_.program(), "n_point_lights");
  glUniform1iv(n_point_lights_loc, 1, &n_point_lights);

  CheckOpenGLError("Set lighting");

  BindAndDrawPrimitive(mesh, texture);
  CheckOpenGLError("Draw primitive");
}

void MeshRenderer::BindAndDrawPrimitive(Mesh &mesh, Texture &texture) {
  // Set model pose matrix
  GLint m_loc = glGetUniformLocation(program_.program(), "M");
  glUniformMatrix4fv(m_loc, 1, GL_TRUE, mesh.pose_.data());
  CheckOpenGLError("Set model pose matrix");

  // Bind mesh buffers
  mesh.Bind();
  CheckOpenGLError("Bind mesh");

  // Bind texture
  glActiveTexture(GL_TEXTURE0);
  CheckOpenGLError("glActiveTexture");

  texture.Bind();
  GLint base_color_texture_loc =
      glGetUniformLocation(program_.program(), "material.base_color_texture");
  glUniform1iv(base_color_texture_loc, 1, &texture.tex_alloc_idx_);
  CheckOpenGLError("Bind texture");

  GLint emissive_factor_loc =
      glGetUniformLocation(program_.program(), "material.emissive_factor");
  glUniform3fv(emissive_factor_loc, 1, texture.emisive_factor_.data());

  GLint base_color_factor_loc =
      glGetUniformLocation(program_.program(), "material.base_color_factor");
  glUniform4fv(base_color_factor_loc, 1, texture.base_color_factor_.data());

  GLint metallic_factor_loc =
      glGetUniformLocation(program_.program(), "material.metallic_factor");
  glUniform1fv(metallic_factor_loc, 1, &texture.metallic_factor_);

  GLint roughness_factor_loc =
      glGetUniformLocation(program_.program(), "material.roughness_factor");
  glUniform1fv(roughness_factor_loc, 1, &texture.roughness_factor_);
  CheckOpenGLError("Set uniform for materials");

  //  Set blending options
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Set wireframe mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // Set culling mode
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glDisable(GL_PROGRAM_POINT_SIZE);

  glDrawElementsInstanced(GL_TRIANGLES, mesh.numFaces(), GL_UNSIGNED_INT,
                          BUFFER_OFFSET(0), 1);

  mesh.UnBind();
}

void MeshRenderer::ConfigureMainFramebuffer() {
  if (main_fb != 0) {
    return;
  }
  // Standard buffer
  glGenRenderbuffers(1, &main_cb);
  glGenRenderbuffers(1, &main_db);

  glBindRenderbuffer(GL_RENDERBUFFER, main_cb);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, VIEWPORT_WIDTH,
                        VIEWPORT_HEIGHT);

  glBindRenderbuffer(GL_RENDERBUFFER, main_db);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, VIEWPORT_WIDTH,
                        VIEWPORT_HEIGHT);

  glGenFramebuffers(1, &main_fb);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, main_fb);
  glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, main_cb);
  glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, main_db);

  // Multisample buffer
  glGenRenderbuffers(1, &main_cb_ms);
  glGenRenderbuffers(1, &main_db_ms);

  glBindRenderbuffer(GL_RENDERBUFFER, main_cb_ms);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA, VIEWPORT_WIDTH,
                                   VIEWPORT_HEIGHT);

  glBindRenderbuffer(GL_RENDERBUFFER, main_db_ms);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24,
                                   VIEWPORT_WIDTH, VIEWPORT_HEIGHT);

  glGenFramebuffers(1, &main_fb_ms);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, main_fb_ms);
  glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, main_cb_ms);
  glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, main_db_ms);
}

void MeshRenderer::ConfigureForwardPassViewport() {
  ConfigureMainFramebuffer();

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, main_fb_ms);

  glViewport(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);
  glDepthRange(0.0, 1.0);
}

Mesh::Mesh(aiMesh *mesh) : mesh_(mesh), vaid_(0) {}

void Mesh::AddToContext() {
  if (vaid_ != 0) {
    return;
  }

  auto *positions = mesh_->mVertices;
  auto *normals = mesh_->mNormals;
  auto *texcoord = mesh_->mTextureCoords[0];
  if (texcoord == nullptr) {
    return;
  }
  // CHECK(texcoord != nullptr);
  // LOG(WARNING) << "Texcoord is ok";

  // Generate and bind VAO
  glGenVertexArrays(1, &vaid_);
  glBindVertexArray(vaid_);

  // Fill vertex buffer
  size_t num_vertex = mesh_->mNumVertices;
  vertices_array_buf_.resize(num_vertex);

  for (size_t j = 0; j < num_vertex; j++) {
    Vertex &vertex = vertices_array_buf_[j];
    vertex.x = positions[j].x;
    vertex.y = positions[j].y;
    vertex.z = positions[j].z;

    vertex.nx = normals[j].x;
    vertex.ny = normals[j].y;
    vertex.nz = normals[j].z;

    vertex.s0 = texcoord[j].x;
    vertex.t0 = texcoord[j].y;
  }
  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);

  glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * num_vertex,
               vertices_array_buf_.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        BUFFER_OFFSET(0));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        BUFFER_OFFSET(12));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        BUFFER_OFFSET(24));
  glEnableVertexAttribArray(2);

  // Fill model matrix buffer
  glGenBuffers(1, &model_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, model_buffer_);
  glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), pose_.data(),
               GL_STATIC_DRAW);
  for (int i = 0; i < 4; i++) {
    size_t idx = i + kNumAttrib;
    glEnableVertexAttribArray(idx);
    glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 16,
                          BUFFER_OFFSET(4 * sizeof(float) * i));
    glVertexAttribDivisor(idx, 1);
  }

  // Fill element buffer
  size_t num_faces = mesh_->mNumFaces;

  indices_.clear();
  indices_.reserve(num_faces * 3);
  for (size_t j = 0; j < mesh_->mNumFaces; j++) {
    auto &face = mesh_->mFaces[j];
    if (face.mNumIndices != 3) {
      LOG(ERROR) << "Unexpected: number of indices = " << face.mNumIndices
                 << " num faces = " << num_faces << " face index = " << j;
    }
    indices_.push_back(face.mIndices[0]);
    indices_.push_back(face.mIndices[1]);
    indices_.push_back(face.mIndices[2]);
  }

  glGenBuffers(1, &element_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices_.size(),
               indices_.data(), GL_STATIC_DRAW);

  // Unbind
  glBindVertexArray(0);
  CheckOpenGLError("Mesh::AddToContext");
}

void Mesh::Bind() {
  if (vaid_ == 0) {
    // LOG(FATAL) << "Haven't add to context";
    return;
  }
  glBindVertexArray(vaid_);

  CheckOpenGLError("Mesh::Bind()");
}

void Mesh::UnBind() {
  if (vaid_ == 0) {
    // LOG(FATAL) << "Haven't add to context";
    return;
  }
  glBindVertexArray(0);
  CheckOpenGLError("Mesh::UnBind()");
}

Texture::Texture(unsigned char *data, int width, int height, int channels)
    : data_(data), width_(width), height_(height), channels_(channels),
      texid_(0), emisive_factor_(0, 0, 0), base_color_factor_(1, 1, 1, 1),
      metallic_factor_(1.0), roughness_factor_(0.9036020036098448) {}

void Texture::AddToContext() {
  if (texid_ != 0) {
    return;
  }

  // Generate the opengl texture
  glGenTextures(1, &texid_);
  glBindTexture(GL_TEXTURE_2D, texid_);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, data_);

  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, kBorderColor);

  // Unbind texture
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("Texture::AddToContext");
}

void Texture::Bind() {
  if (texid_ == 0) {
    LOG(FATAL) << "Haven't add to context";
  }
  glBindTexture(GL_TEXTURE_2D, texid_);
  CheckOpenGLError("Texture::Bind");
}

void Texture::UnBind() {
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("Texture::UnBind");
}

Program::Program(const std::string &vertex_shader,
                 const std::string &fragment_shader)
    : vertex_shader_path_(vertex_shader),
      fragment_shader_path_(fragment_shader), program_(0) {}

void Program::AddToContext() {
  if (program_ != 0) {
    return;
  }

  if (!std::filesystem::exists(vertex_shader_path_) ||
      !std::filesystem::exists(fragment_shader_path_)) {
    LOG(FATAL) << "One of the shader doesn't exist";
  }

  std::string vertex_shader_source = ReadShaderSource(vertex_shader_path_);
  std::string fragment_shader_source = ReadShaderSource(fragment_shader_path_);
  if (vertex_shader_source.empty()) {
    LOG(FATAL) << "Empty vertex shader source";
  }
  if (fragment_shader_source.empty()) {
    LOG(FATAL) << "Empty fragment shader source";
  }

  GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  const GLchar *vertex_shader_source_c_str =
      (const GLchar *)vertex_shader_source.c_str();
  glShaderSource(vertex_shader, 1, &vertex_shader_source_c_str, nullptr);
  glCompileShader(vertex_shader);
  if (!CheckShaderCompiled(vertex_shader)) {
    LOG(FATAL) << "Error in compiling " << vertex_shader_path_;
  }

  GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  const GLchar *fragment_shader_source_c_str =
      (const GLchar *)fragment_shader_source.c_str();
  glShaderSource(fragment_shader, 1, &fragment_shader_source_c_str, nullptr);
  glCompileShader(fragment_shader);
  if (!CheckShaderCompiled(fragment_shader)) {
    LOG(FATAL) << "Error in compiling " << fragment_shader_path_;
  }

  LOG(INFO) << "Shader compiled successfully";

  program_ = glCreateProgram();

  glAttachShader(program_, vertex_shader);
  glAttachShader(program_, fragment_shader);
  glLinkProgram(program_);
  if (!CheckValidateProgram(program_)) {
    LOG(FATAL) << "Error in validating program" << std::endl;
  }
  if (!CheckProgramLinked(program_)) {
    LOG(FATAL) << "Error in linking program" << std::endl;
  }
  CheckOpenGLError("Compile Program");
}

void Program::Bind() {
  if (program_ == 0) {
    LOG(FATAL) << "Program hasn't add to context";
  }
  glUseProgram(program_);
}

void Program::UnBind() { glUseProgram(0); }

} // namespace ubipose
