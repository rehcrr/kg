#include "application.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {
namespace {

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 2.0f * pi;

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

struct Vertex {
    Vec3 position;
    Vec3 color;
};

// GLSL mat4 и массив float[16] хранятся в column-major порядке.
struct Mat4 {
    float v[16]{};
};

struct UniformBlock {
    float mvp[16];
    float color[4]; // vec4: std140 alignment
};

struct GpuResources {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_allocation = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_allocation = VK_NULL_HANDLE;
    VkBuffer uniform_buffer = VK_NULL_HANDLE;
    VmaAllocation uniform_allocation = VK_NULL_HANDLE;
    void* uniform_mapped = nullptr;
    uint32_t index_count = 0;
};

struct UiSettings {
    int projection = 0; // 0 = perspective, 1 = orthographic
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotation[3] = {0.0f, 0.0f, 0.0f}; // градусы
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float color[3] = {1.0f, 1.0f, 1.0f};

    bool playing = true;
    float speed = 1.0f;
    float trajectory_radius = 0.8f;
    float trajectory_height = 0.45f;
    float spin_speed = 65.0f; // градусов в секунду
    double animation_time = 0.0;
    double previous_time = -1.0;
};

GpuResources gpu;
UiSettings ui;

Mat4 identity() {
    Mat4 r{};
    r.v[0] = r.v[5] = r.v[10] = r.v[15] = 1.0f;
    return r;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result.v[col * 4 + row] +=
                    a.v[k * 4 + row] * b.v[col * 4 + k];
            }
        }
    }
    return result;
}

Mat4 translation(float x, float y, float z) {
    Mat4 result = identity();
    result.v[12] = x;
    result.v[13] = y;
    result.v[14] = z;
    return result;
}

Mat4 scaling(float x, float y, float z) {
    Mat4 result = identity();
    result.v[0] = x;
    result.v[5] = y;
    result.v[10] = z;
    return result;
}

Mat4 rotationX(float degrees) {
    const float angle = degrees * pi / 180.0f;
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 result = identity();
    result.v[5] = c;
    result.v[6] = s;
    result.v[9] = -s;
    result.v[10] = c;
    return result;
}

Mat4 rotationY(float degrees) {
    const float angle = degrees * pi / 180.0f;
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 result = identity();
    result.v[0] = c;
    result.v[2] = -s;
    result.v[8] = s;
    result.v[10] = c;
    return result;
}

Mat4 rotationZ(float degrees) {
    const float angle = degrees * pi / 180.0f;
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 result = identity();
    result.v[0] = c;
    result.v[1] = s;
    result.v[4] = -s;
    result.v[5] = c;
    return result;
}

Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalized(Vec3 a) {
    const float len = std::sqrt(dot(a, a));
    return {a.x / len, a.y / len, a.z / len};
}

Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalized(subtract(center, eye));
    const Vec3 s = normalized(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r = identity();
    r.v[0] = s.x;  r.v[4] = s.y;  r.v[8]  = s.z;
    r.v[1] = u.x;  r.v[5] = u.y;  r.v[9]  = u.z;
    r.v[2] = -f.x; r.v[6] = -f.y; r.v[10] = -f.z;
    r.v[12] = -dot(s, eye);
    r.v[13] = -dot(u, eye);
    r.v[14] = dot(f, eye);
    return r;
}

// Здесь проекции уже используют Vulkan NDC (ось Y инвертирована, Z = [0,1]).
Mat4 perspective(float field_of_view, float aspect, float z_near, float z_far) {
    const float f = 1.0f / std::tan(field_of_view * pi / 360.0f);
    Mat4 r{};
    r.v[0] = f / aspect;
    r.v[5] = -f;
    r.v[10] = z_far / (z_near - z_far);
    r.v[11] = -1.0f;
    r.v[14] = (z_far * z_near) / (z_near - z_far);
    return r;
}

Mat4 orthographic(float half_height, float aspect, float z_near, float z_far) {
    Mat4 r = identity();
    r.v[0] = 1.0f / (half_height * aspect);
    r.v[5] = -1.0f / half_height;
    r.v[10] = 1.0f / (z_near - z_far);
    r.v[14] = z_near / (z_near - z_far);
    return r;
}

// Генерация тора как индексированной сетки; ось тора совпадает с локальной Z.
// x=(R+r*cos(v))*cos(u), y=(R+r*cos(v))*sin(u), z=r*sin(v).
void makeTorus(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    constexpr auto n = torus_major_segments;
    constexpr auto m = torus_tube_segments;
    constexpr float R = torus_major_radius;
    constexpr float r = torus_tube_radius;

    static_assert(n * m <= 65535, "Use uint32_t indices for a larger torus");
    vertices.reserve(n * m);
    indices.reserve(n * m * 6);

    for (uint32_t i = 0; i < n; ++i) {
        const float u = two_pi * float(i) / float(n);
        const float cu = std::cos(u), su = std::sin(u);
        for (uint32_t j = 0; j < m; ++j) {
            const float v = two_pi * float(j) / float(m);
            const float cv = std::cos(v), sv = std::sin(v);
            const float ring = R + r * cv;
            const Vec3 p = {ring * cu, ring * su, r * sv};

            // Задание 5: процедурный цвет зависит от локальных координат.
            // Интервал каждого канала: [0.35, 1.0].
            const Vec3 c = {
                0.675f + 0.325f * (p.x / (R + r)),
                0.675f + 0.325f * (p.y / (R + r)),
                0.675f + 0.325f * (p.z / r)
            };
            vertices.push_back({p, c});
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < m; ++j) {
            const uint16_t a = uint16_t(i * m + j);
            const uint16_t b = uint16_t(((i + 1) % n) * m + j);
            const uint16_t c = uint16_t(i * m + (j + 1) % m);
            const uint16_t d = uint16_t(((i + 1) % n) * m + (j + 1) % m);
            // CCW при взгляде снаружи тора, замыкание по обоим параметрам.
            indices.insert(indices.end(), {a, b, c, b, d, c});
        }
    }
}

bool loadShader(const char* path, std::vector<uint32_t>& words) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open shader: " << path << '\n';
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        std::cerr << "Invalid SPIR-V shader: " << path << '\n';
        return false;
    }
    words.resize(static_cast<size_t>(size) / 4);
    file.seekg(0);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(words.data()), size));
}

bool allocateBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage,
                    VkBuffer& buffer, VmaAllocation& allocation, void*& mapped) {
    auto& ctx = graphics::internal::context;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo memory{};
    memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
    memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

    VmaAllocationInfo out{};
    if (vmaCreateBuffer(ctx.allocator, &info, &memory, &buffer, &allocation, &out) != VK_SUCCESS) {
        std::cerr << "Cannot allocate Vulkan buffer\n";
        return false;
    }
    mapped = out.pMappedData;
    return true;
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo ds_info{};
    ds_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_info.bindingCount = 1;
    ds_info.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(ctx.device, &ds_info, nullptr, &gpu.descriptor_layout) != VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &gpu.descriptor_layout;
    if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &gpu.pipeline_layout) != VK_SUCCESS)
        return false;

    std::vector<uint32_t> vert_words, frag_words;
    if (!loadShader("shaders/triangle.vert.spv", vert_words) ||
        !loadShader("shaders/triangle.frag.spv", frag_words))
        return false;

    VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo sm_info{};
    sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_info.codeSize = vert_words.size() * sizeof(uint32_t);
    sm_info.pCode = vert_words.data();
    if (vkCreateShaderModule(ctx.device, &sm_info, nullptr, &vert) != VK_SUCCESS)
        return false;
    sm_info.codeSize = frag_words.size() * sizeof(uint32_t);
    sm_info.pCode = frag_words.data();
    if (vkCreateShaderModule(ctx.device, &sm_info, nullptr, &frag) != VK_SUCCESS) {
        vkDestroyShaderModule(ctx.device, vert, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding = 0;
    vertex_binding.stride = sizeof(Vertex);
    vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(Vertex, position);
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &vertex_binding;
    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo topology{};
    topology.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    topology.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE; // корректно видно даже при отражающем scale
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    depth.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending{};
    blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blending.attachmentCount = 1;
    blending.pAttachments = &attachment;

    VkGraphicsPipelineCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    create.stageCount = 2;
    create.pStages = stages;
    create.pVertexInputState = &vertex_input;
    create.pInputAssemblyState = &topology;
    create.pViewportState = &viewport;
    create.pRasterizationState = &raster;
    create.pMultisampleState = &ms;
    create.pDepthStencilState = &depth;
    create.pColorBlendState = &blending;
    create.pDynamicState = &dynamic;
    create.layout = gpu.pipeline_layout;
    create.renderPass = ctx.render_pass;
    create.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE,
                                                     1, &create, nullptr, &gpu.pipeline);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create graphics pipeline\n";
        return false;
    }
    return true;
}

bool createMeshBuffers() {
    auto& ctx = graphics::internal::context;
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    makeTorus(vertices, indices);
    gpu.index_count = static_cast<uint32_t>(indices.size());

    void* mapped = nullptr;
    if (!allocateBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        gpu.vertex_buffer, gpu.vertex_allocation, mapped))
        return false;
    std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(Vertex));
    if (vmaFlushAllocation(ctx.allocator, gpu.vertex_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS)
        return false;

    if (!allocateBuffer(sizeof(uint16_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        gpu.index_buffer, gpu.index_allocation, mapped))
        return false;
    std::memcpy(mapped, indices.data(), indices.size() * sizeof(uint16_t));
    if (vmaFlushAllocation(ctx.allocator, gpu.index_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS)
        return false;

    if (!allocateBuffer(sizeof(UniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        gpu.uniform_buffer, gpu.uniform_allocation, gpu.uniform_mapped))
        return false;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &gpu.descriptor_pool) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = gpu.descriptor_pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &gpu.descriptor_layout;
    if (vkAllocateDescriptorSets(ctx.device, &alloc, &gpu.descriptor_set) != VK_SUCCESS)
        return false;

    VkDescriptorBufferInfo ubo_buffer{};
    ubo_buffer.buffer = gpu.uniform_buffer;
    ubo_buffer.offset = 0;
    ubo_buffer.range = sizeof(UniformBlock);
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = gpu.descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &ubo_buffer;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return true;
}

void updateUniform(float aspect) {
    const Mat4 projection = ui.projection == 0
        ? perspective(50.0f, aspect, 0.1f, 100.0f)
        : orthographic(3.6f, aspect, 0.1f, 100.0f);
    const Mat4 view = lookAt({0.0f, 2.8f, 9.0f},
                             {0.0f, 0.0f, 0.0f},
                             {0.0f, 1.0f, 0.0f});

    const float t = static_cast<float>(ui.animation_time);
    // Сложная периодическая траектория: компоненты с частотами t и 2t.
    const float x = ui.position[0] + ui.trajectory_radius * std::cos(t);
    const float y = ui.position[1] + ui.trajectory_height * std::sin(2.0f * t);
    const float z = ui.position[2] + ui.trajectory_radius * std::sin(t);
    const Mat4 model = multiply(translation(x, y, z),
        multiply(rotationY(ui.rotation[1] + t * ui.spin_speed),
        multiply(rotationX(ui.rotation[0] + t * 0.35f * ui.spin_speed),
        multiply(rotationZ(ui.rotation[2]),
                 scaling(ui.scale[0], ui.scale[1], ui.scale[2])))));

    const Mat4 mvp = multiply(projection, multiply(view, model));
    UniformBlock block{};
    std::memcpy(block.mvp, mvp.v, sizeof(block.mvp));
    block.color[0] = ui.color[0];
    block.color[1] = ui.color[1];
    block.color[2] = ui.color[2];
    block.color[3] = 1.0f;
    std::memcpy(gpu.uniform_mapped, &block, sizeof(block));
    vmaFlushAllocation(graphics::internal::context.allocator,
                       gpu.uniform_allocation, 0, VK_WHOLE_SIZE);
}

} // namespace

bool initialize() {
    if (!createPipeline() || !createMeshBuffers()) {
        std::cerr << "Torus initialization failed\n";
        shutdown();
        return false;
    }
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    if (gpu.vertex_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx.allocator, gpu.vertex_buffer, gpu.vertex_allocation);
    if (gpu.index_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx.allocator, gpu.index_buffer, gpu.index_allocation);
    if (gpu.uniform_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx.allocator, gpu.uniform_buffer, gpu.uniform_allocation);

    if (gpu.descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx.device, gpu.descriptor_pool, nullptr);
    if (gpu.pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, gpu.pipeline, nullptr);
    if (gpu.pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx.device, gpu.pipeline_layout, nullptr);
    if (gpu.descriptor_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx.device, gpu.descriptor_layout, nullptr);
    gpu = GpuResources{};
}

void update(double time) {
    if (ui.previous_time >= 0.0 && ui.playing) {
        // Ограничение защищает от большого скачка после перетаскивания окна.
        ui.animation_time += std::clamp(time - ui.previous_time, 0.0, 0.1)
                             * ui.speed;
    }
    ui.previous_time = time;

    ImGui::SetNextWindowSize(ImVec2(365.0f, 490.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 1 - Torus (variant 7)");

    // Доп. задание 1: переключение проекций в реальном времени.
    const char* modes[] = {"Perspective", "Orthographic"};
    ImGui::Combo("Projection", &ui.projection, modes, IM_ARRAYSIZE(modes));

    ImGui::SeparatorText("Transform");
    // Доп. задание 2: аффинные преобразования.
    ImGui::DragFloat3("Position", ui.position, 0.025f, -10.0f, 10.0f, "%.2f");
    ImGui::DragFloat3("Rotation (degrees)", ui.rotation, 0.5f, -360.0f, 360.0f, "%.1f");
    ImGui::DragFloat3("Scale", ui.scale, 0.01f, 0.05f, 5.0f, "%.2f");
    if (ImGui::Button("Reset transform")) {
        for (int i = 0; i < 3; ++i) {
            ui.position[i] = ui.rotation[i] = 0.0f;
            ui.scale[i] = 1.0f;
        }
    }

    ImGui::SeparatorText("Animation");
    // Доп. задание 3: сложная траектория и вращение.
    if (ImGui::Button(ui.playing ? "Pause" : "Play")) ui.playing = !ui.playing;
    ImGui::SameLine();
    if (ImGui::Button("Restart")) ui.animation_time = 0.0;
    ImGui::SliderFloat("Speed", &ui.speed, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Path radius", &ui.trajectory_radius, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Path height", &ui.trajectory_height, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Spin (deg/sec)", &ui.spin_speed, -240.0f, 240.0f, "%.0f");

    ImGui::SeparatorText("Color");
    // Доп. задания 4 и 5: базовый цвет умножается на цвет каждой вершины в GLSL.
    ImGui::ColorEdit3("Torus tint", ui.color);
    ImGui::TextUnformatted("Tint is multiplied by procedural vertex RGB.");
    ImGui::Text("Vertices: %u, triangles: %u",
                torus_major_segments * torus_tube_segments,
                2 * torus_major_segments * torus_tube_segments);
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE ||
        ctx.swapchain_extent.height == 0 || ctx.swapchain_extent.width == 0) return;

    // graphics::internal::prepare() уже дождался Vulkan fence предыдущего кадра.
    const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                         static_cast<float>(ctx.swapchain_extent.height);
    updateUniform(aspect);

    vkResetCommandBuffer(fd.command_buffer, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(fd.command_buffer, &begin) != VK_SUCCESS) return;

    VkClearValue clear[2]{};
    clear[0].color.float32[0] = 0.08f;
    clear[0].color.float32[1] = 0.09f;
    clear[0].color.float32[2] = 0.13f;
    clear[0].color.float32[3] = 1.0f;
    clear[1].depthStencil.depth = 1.0f;
    clear[1].depthStencil.stencil = 0;

    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = ctx.render_pass;
    pass.framebuffer = fd.framebuffer;
    pass.renderArea.offset = {0, 0};
    pass.renderArea.extent = ctx.swapchain_extent;
    pass.clearValueCount = 2;
    pass.pClearValues = clear;

    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.pipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(ctx.swapchain_extent.width);
    viewport.height = static_cast<float>(ctx.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &gpu.vertex_buffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, gpu.index_buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpu.pipeline_layout, 0, 1, &gpu.descriptor_set, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, gpu.index_count, 1, 0, 0, 0);
    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS)
        std::cerr << "Failed to record torus command buffer\n";
}

} // namespace application
