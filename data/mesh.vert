#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
#if 1
layout(location = 1) in vec3 normal;
#endif
#if 0
layout(location = TANGENT_LOC) in vec4 tangent;
#endif
#if 1
layout(location = 2) in vec2 texcoord_0;
#endif
#if 0
layout(location = TEXCOORD_1_LOC) in vec2 texcoord_1;
#endif
#if 0
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif
#if 0
layout(location = JOINTS_0_LOC) in vec4 joints_0;
#endif
#if 0
layout(location = WEIGHTS_0_LOC) in vec4 weights_0;
#endif
layout(location = 3) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
out vec3 frag_position;
#if 1
out vec3 frag_normal;
#endif
#if 0
#if 0
#if 1
out mat3 tbn;
#endif
#endif
#endif
#if 1
out vec2 uv_0;
#endif
#if 0
out vec2 uv_1;
#endif
#if 0
out vec4 color_multiplier;
#endif


void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat4 N = transpose(inverse(M * inst_m));

#if 1
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
#endif

#if 0
#if 0
#if 1
    vec3 normal_w = normalize(vec3(N * vec4(normal, 0.0)));
    vec3 tangent_w = normalize(vec3(N * vec4(tangent.xyz, 0.0)));
    vec3 bitangent_w = cross(normal_w, tangent_w) * tangent.w;
    tbn = mat3(tangent_w, bitangent_w, normal_w);
#endif
#endif
#endif
#if 1
    uv_0 = texcoord_0;
#endif
#if 0
    uv_1 = texcoord_1;
#endif
#if 0
    color_multiplier = color_0;
#endif
}