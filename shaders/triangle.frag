#version 450
layout(location = 0) in vec3 fragWorldPosition;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

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

struct PointLight { vec4 position; vec4 colorIntensity; };
layout(set = 0, binding = 1, std430) readonly buffer PointBuffer {
    PointLight lights[];
} pointBuffer;

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 cutoffs;
};
layout(set = 0, binding = 2, std430) readonly buffer SpotBuffer {
    SpotLight lights[];
} spotBuffer;
layout(push_constant) uniform Draw {
    mat4 markerModel;
    vec4 emission;
    int unlit;
} drawParams;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 baseColor) {
    float diffuseFactor = max(dot(N, L), 0.0);
    vec3 diffuse = baseColor * diffuseFactor;
    vec3 specular = vec3(0.0);
    if (diffuseFactor > 0.0) {
        vec3 H = normalize(L + V);
        float highlight = pow(max(dot(N, H), 0.0), max(1.0, material.specular.w));
        specular = material.specular.rgb * highlight;
    }
    return (diffuse + specular) * radiance;
}

void main() {
    // Emissive markers are never lit by Blinn-Phong or other sources.
    if (drawParams.unlit != 0) {
        outColor = vec4(drawParams.emission.rgb, 1.0);
        return;
    }

    vec3 N = normalize(fragWorldNormal);
    vec3 V = normalize(scene.cameraPosition.xyz - fragWorldPosition);
    vec3 baseColor = material.albedo.rgb * fragColor;
    vec3 lit = baseColor * scene.ambient.rgb * scene.ambient.w;

    vec3 sunL = normalize(-scene.sunDirection.xyz);
    lit += blinnPhong(N, V, sunL, scene.sunColor.rgb * scene.sunColor.w, baseColor);

    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 toLight = pointBuffer.lights[i].position.xyz - fragWorldPosition;
        float distanceSquared = dot(toLight, toLight);
        vec3 L = toLight * inversesqrt(max(distanceSquared, 0.0001));
        // Physically motivated inverse-square attenuation with a small near-field clamp.
        float attenuation = 1.0 / max(distanceSquared, 0.05);
        vec3 radiance = pointBuffer.lights[i].colorIntensity.rgb *
                        pointBuffer.lights[i].colorIntensity.w * attenuation;
        lit += blinnPhong(N, V, L, radiance, baseColor);
    }
    for (int i = 0; i < scene.counts.y; ++i) {
        vec3 toLight = spotBuffer.lights[i].position.xyz - fragWorldPosition;
        float distanceSquared = dot(toLight, toLight);
        vec3 L = toLight * inversesqrt(max(distanceSquared, 0.0001));
        float attenuation = 1.0 / max(distanceSquared, 0.05);
        float cosTheta = dot(normalize(spotBuffer.lights[i].direction.xyz), -L);
        float innerCos = spotBuffer.lights[i].cutoffs.x;
        float outerCos = spotBuffer.lights[i].cutoffs.y;
        float edge = (innerCos - outerCos < 0.00001)
                     ? step(innerCos, cosTheta)
                     : smoothstep(outerCos, innerCos, cosTheta);
        vec3 radiance = spotBuffer.lights[i].colorIntensity.rgb *
                        spotBuffer.lights[i].colorIntensity.w * attenuation * edge;
        lit += blinnPhong(N, V, L, radiance, baseColor);
    }
    // Tone mapping prevents bright lights from turning the entire torus white.
    lit = lit / (lit + vec3(1.0));
    outColor = vec4(lit, 1.0);
}
