#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(binding = 0) uniform UniformBlock {
    mat4 mvp;
    vec4 color;
} ubo;

layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
    outColor = inColor * ubo.color.rgb;
}
