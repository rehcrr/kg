#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 sunDirection;
    vec4 sunColor;
    ivec4 counts;
} scene;
layout(set = 1, binding = 0, std140) uniform Model {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} material;
layout(push_constant) uniform Draw {
    mat4 markerModel;
    vec4 emission;
    int unlit;
} drawParams;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    mat4 m = (drawParams.unlit != 0) ? drawParams.markerModel : material.model;
    vec4 world = m * vec4(inPosition, 1.0);
    gl_Position = scene.viewProjection * world;
    fragWorldPosition = world.xyz;
    fragWorldNormal = (material.normalMatrix * vec4(inNormal, 0.0)).xyz;
    fragColor = inColor;
}
