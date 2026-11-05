#include "application.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace application {
namespace {

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 2.0f * pi;
constexpr uint32_t max_lights = 128; 
constexpr int cone_sides = 16;

struct Vec3 { float x = 0.0f, y = 0.0f, z = 0.0f; };
struct alignas(16) Vec4 { float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f; };
struct Mat4 { float v[16]{}; };
struct Vertex { Vec3 position, normal, color; };


struct alignas(16) SceneUniform {
    Mat4 view_projection;
    Vec4 camera_position;
    Vec4 ambient;             
    Vec4 sun_direction;       
    Vec4 sun_color;           
    alignas(16) int counts[4]{}; 
};
struct alignas(16) ModelUniform {
    Mat4 model;
    Mat4 normal_matrix;
    Vec4 albedo;              // умножается на процедурный цвет вершины
    Vec4 specular;            // rgb + степень Блинна-Фонга
};
struct alignas(16) PointGpu { Vec4 position; Vec4 color_intensity; };
struct alignas(16) SpotGpu {
    Vec4 position;
    Vec4 direction;
    Vec4 color_intensity;
    Vec4 cutoffs;             
};
struct alignas(16) DrawPush {
    Mat4 model;
    Vec4 emission;
    int unlit = 0;
    int padding[3]{};
};
// Проверки на компиляции
static_assert(sizeof(Vec4) == 16);
static_assert(sizeof(PointGpu) == 32 && sizeof(SpotGpu) == 64);
static_assert(offsetof(SceneUniform, counts) % 16 == 0);
static_assert(sizeof(DrawPush) == 96);

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
};
struct Mesh {
    Buffer vertices, indices;
    uint32_t index_count = 0;
};
struct GpuResources {
    VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> sets{};
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline solid_pipeline = VK_NULL_HANDLE;
    VkPipeline line_pipeline = VK_NULL_HANDLE;
    Mesh torus, cube;
    Buffer cone_lines;
    Buffer scene_ubo, model_ubo, points_ssbo, spots_ssbo;
};
struct PointLight {
    float position[3] = {2.4f, 1.6f, 2.0f};
    float color[3] = {1.0f, 0.42f, 0.18f};
    float intensity = 12.0f;
};
struct SpotLight {
    float position[3] = {-2.5f, 3.0f, 2.0f};
    float direction[3] = {0.6f, -1.0f, -0.5f};
    float color[3] = {0.30f, 0.65f, 1.0f};
    float intensity = 30.0f;
    float inner_angle = 15.0f;
    float outer_angle = 27.0f;
};
struct Settings {
    int projection = 0;
    float camera_position[3] = {0.0f, 2.8f, 8.0f};
    float yaw = -90.0f, pitch = -19.3f, fov = 55.0f;
    float camera_speed = 4.0f;
    float ortho_size = 4.2f;
    bool last_mouse_valid = false;
    double mouse_x = 0.0, mouse_y = 0.0;

    float position[3]{};
    float rotation[3]{};
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float albedo[3] = {1.0f, 0.82f, 0.72f};
    float specular[3] = {1.0f, 1.0f, 1.0f};
    float shininess = 48.0f;

    float ambient[3] = {0.18f, 0.20f, 0.28f};
    float ambient_intensity = 0.65f;
    float sun_direction[3] = {-0.5f, -1.0f, -0.3f};
    float sun_color[3] = {1.0f, 0.96f, 0.86f};
    float sun_intensity = 1.8f;

    std::vector<PointLight> points = {PointLight{}};
    std::vector<SpotLight> spots = {SpotLight{}};
    bool show_light_markers = true;
    bool show_spot_cones = true;

    bool playing = false;
    double previous_time = -1.0, animation_time = 0.0;
    float speed = 1.0f, trajectory_radius = 0.8f;
    float trajectory_height = 0.45f, spin_speed = 65.0f;
};
GpuResources gpu;
Settings ui;

Vec3 v3(const float a[3]) { return {a[0], a[1], a[2]}; }
Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 mul(Vec3 a, float t) { return {a.x * t, a.y * t, a.z * t}; }
float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
// Нормирование вектора при нулевой длине возвращаем запасной вектор
Vec3 normal(Vec3 a) {
    const float len = std::sqrt(dot(a,a));
    return len > 1e-6f ? mul(a,1.0f/len) : Vec3{0.0f,-1.0f,0.0f};
}
Vec4 vec4(Vec3 a, float w = 0.0f) { return {a.x, a.y, a.z, w}; }
float radians(float degrees) { return degrees * pi / 180.0f; }

Mat4 identity() {
    Mat4 a{};
    a.v[0] = a.v[5] = a.v[10] = a.v[15] = 1.0f;
    return a;
}
// Умножение матриц
Mat4 mmul(const Mat4& a, const Mat4& b) {
    Mat4 c{};
    for (int col=0; col<4; ++col)
        for (int row=0; row<4; ++row)
            for (int k=0; k<4; ++k)
                c.v[col*4+row] += a.v[k*4+row]*b.v[col*4+k];
    return c;
}
Mat4 translation(Vec3 p) {
    Mat4 a=identity();
    a.v[12]=p.x; a.v[13]=p.y; a.v[14]=p.z;
    return a;
}
Mat4 scale(Vec3 p) {
    Mat4 a=identity();
    a.v[0]=p.x; a.v[5]=p.y; a.v[10]=p.z;
    return a;
}
Mat4 rx(float deg) {
    const float c=std::cos(radians(deg)), s=std::sin(radians(deg));
    Mat4 a=identity(); a.v[5]=c; a.v[6]=s; a.v[9]=-s; a.v[10]=c;
    return a;
}
Mat4 ry(float deg) {
    const float c=std::cos(radians(deg)), s=std::sin(radians(deg));
    Mat4 a=identity(); a.v[0]=c; a.v[2]=-s; a.v[8]=s; a.v[10]=c;
    return a;
}
Mat4 rz(float deg) {
    const float c=std::cos(radians(deg)), s=std::sin(radians(deg));
    Mat4 a=identity(); a.v[0]=c; a.v[1]=s; a.v[4]=-s; a.v[5]=c;
    return a;
}
// переводит мировые координаты в координаты камеры
Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f=normal(sub(center,eye)), s=normal(cross(f,up)), u=cross(s,f);
    Mat4 a=identity();
    a.v[0]=s.x; a.v[4]=s.y; a.v[8]=s.z;
    a.v[1]=u.x; a.v[5]=u.y; a.v[9]=u.z;
    a.v[2]=-f.x; a.v[6]=-f.y; a.v[10]=-f.z;
    a.v[12]=-dot(s,eye); a.v[13]=-dot(u,eye); a.v[14]=dot(f,eye);
    return a;
}
Mat4 perspective(float fov, float aspect, float near_z, float far_z) {
    const float f=1.0f/std::tan(radians(fov)*0.5f);
    Mat4 a{}; a.v[0]=f/aspect; a.v[5]=-f;
    a.v[10]=far_z/(near_z-far_z); a.v[11]=-1.0f;
    a.v[14]=near_z*far_z/(near_z-far_z);
    return a;
}
Mat4 ortho(float half_h, float aspect, float near_z, float far_z) {
    Mat4 a=identity(); a.v[0]=1.0f/(half_h*aspect); a.v[5]=-1.0f/half_h;
    a.v[10]=1.0f/(near_z-far_z); a.v[14]=near_z/(near_z-far_z);
    return a;
}
// Направление взгляда из углов 
Vec3 cameraForward() {
    const float yaw=radians(ui.yaw), pitch=radians(ui.pitch);
    return normal({std::cos(pitch)*std::cos(yaw),std::sin(pitch),std::cos(pitch)*std::sin(yaw)});
}

// Генерация тора 
void makeTorus(std::vector<Vertex>& v, std::vector<uint16_t>& ix) {
    constexpr auto n=torus_major_segments, m=torus_tube_segments;
    constexpr float R=torus_major_radius, r=torus_tube_radius;
    static_assert(n*m < 65536);
    v.reserve(n*m); ix.reserve(n*m*6);
    for (uint32_t i=0;i<n;++i) {
        const float u=two_pi*float(i)/float(n), cu=std::cos(u), su=std::sin(u);
        for (uint32_t j=0;j<m;++j) {
            const float a=two_pi*float(j)/float(m), ca=std::cos(a), sa=std::sin(a);
            const Vec3 pos={(R+r*ca)*cu,(R+r*ca)*su,r*sa};
            // Аналитическая наружная единичная нормаль.
            const Vec3 norm={ca*cu,ca*su,sa};
            // Процедурный цвет в диапазоне [0.35, 1.0].
            const Vec3 col={0.675f+0.325f*pos.x/(R+r),
                            0.675f+0.325f*pos.y/(R+r),
                            0.675f+0.325f*pos.z/r};
            v.push_back({pos,norm,col});
        }
    }
    for (uint32_t i=0;i<n;++i) for(uint32_t j=0;j<m;++j) {
        const auto a=static_cast<uint16_t>(i*m+j);
        const auto b=static_cast<uint16_t>(((i+1)%n)*m+j);
        const auto c=static_cast<uint16_t>(i*m+(j+1)%m);
        const auto d=static_cast<uint16_t>(((i+1)%n)*m+(j+1)%m);
        // CCW при взгляде снаружи, замыкание по обоим параметрам.
        ix.insert(ix.end(), {a,b,c,b,d,c});
    }
}
// Куб единичного размера 
void makeCube(std::vector<Vertex>& v, std::vector<uint16_t>& ix) {
    constexpr float p=0.5f;
    const Vec3 xyz[] = {{-p,-p,-p},{p,-p,-p},{p,p,-p},{-p,p,-p},
                        {-p,-p,p},{p,-p,p},{p,p,p},{-p,p,p}};
    // Угловые нормали
    for (Vec3 a:xyz) v.push_back({a,normal(a),{1,1,1}});
    const uint16_t triangles[] = {0,2,1,0,3,2,4,5,6,4,6,7,
                                  0,1,5,0,5,4,3,7,6,3,6,2,
                                  1,2,6,1,6,5,0,4,7,0,7,3};
    ix.assign(std::begin(triangles),std::end(triangles));
}

// Загрузка SPIR-V
bool readSpirv(const char* path, std::vector<uint32_t>& words) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file) {std::cerr << "Cannot open shader: " << path << "\n";return false;}
    const std::streamsize bytes=static_cast<std::streamsize>(file.tellg());
    if(bytes<=0 || bytes%4!=0) {std::cerr<<"Invalid SPIR-V: "<<path<<"\n";return false;}
    words.resize(static_cast<size_t>(bytes)/4);
    file.seekg(0);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(words.data()),bytes));
}
// Создание буфера через VMA: host-visible + сразу замаплен
bool createBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, Buffer& b) {
    auto& ctx=graphics::internal::context;
    VkBufferCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size=bytes; ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage=VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    ai.flags=VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    if(vmaCreateBuffer(ctx.allocator,&ci,&ai,&b.handle,&b.allocation,&info)!=VK_SUCCESS) {
        std::cerr<<"Failed to allocate buffer of size "<<bytes<<"\n";return false;
    }
    b.mapped=info.pMappedData;
    return b.mapped!=nullptr;
}
// Запись в mapped-память + flush, чтобы GPU увидел изменения
void upload(Buffer& b, const void* data, size_t bytes) {
    std::memcpy(b.mapped,data,bytes);
    vmaFlushAllocation(graphics::internal::context.allocator,b.allocation,0,VK_WHOLE_SIZE);
}
void destroyBuffer(Buffer& b) {
    if(b.handle) vmaDestroyBuffer(graphics::internal::context.allocator,b.handle,b.allocation);
    b=Buffer{};
}
bool makeMesh(Mesh& dst, const std::vector<Vertex>& verts, const std::vector<uint16_t>& indices) {
    dst.index_count=static_cast<uint32_t>(indices.size());
    if(!createBuffer(verts.size()*sizeof(Vertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,dst.vertices) ||
       !createBuffer(indices.size()*sizeof(uint16_t),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,dst.indices)) return false;
    upload(dst.vertices,verts.data(),verts.size()*sizeof(Vertex));
    upload(dst.indices,indices.data(),indices.size()*sizeof(uint16_t));
    return true;
}
void destroyMesh(Mesh& mesh) {destroyBuffer(mesh.vertices);destroyBuffer(mesh.indices);mesh.index_count=0;}

bool createPipelines() {
    auto& ctx=graphics::internal::context;
    // Set 0 — данные всей сцены: UBO + два SSBO со списками источников.
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}
    };
    // Set 1 — данные конкретной модели: один UBO.
    const VkDescriptorSetLayoutBinding model_bindings[] = {
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dsi{}; dsi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsi.bindingCount=3;dsi.pBindings=scene_bindings;
    if(vkCreateDescriptorSetLayout(ctx.device,&dsi,nullptr,&gpu.scene_layout)!=VK_SUCCESS) return false;
    dsi.bindingCount=1;dsi.pBindings=model_bindings;
    if(vkCreateDescriptorSetLayout(ctx.device,&dsi,nullptr,&gpu.model_layout)!=VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[]={gpu.scene_layout,gpu.model_layout};
    // Push constants: модельная матрица, emission, флаг unlit.
    VkPushConstantRange push_range{};
    push_range.stageFlags=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset=0;push_range.size=sizeof(DrawPush);
    VkPipelineLayoutCreateInfo li{};li.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount=2;li.pSetLayouts=layouts;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push_range;
    if(vkCreatePipelineLayout(ctx.device,&li,nullptr,&gpu.pipeline_layout)!=VK_SUCCESS)return false;

    std::vector<uint32_t> vw,fw;
    if(!readSpirv("shaders/triangle.vert.spv",vw)||!readSpirv("shaders/triangle.frag.spv",fw))return false;
    VkShaderModule vs=VK_NULL_HANDLE, fs=VK_NULL_HANDLE;
    VkShaderModuleCreateInfo sm{};sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize=vw.size()*4;sm.pCode=vw.data();
    if(vkCreateShaderModule(ctx.device,&sm,nullptr,&vs)!=VK_SUCCESS)return false;
    sm.codeSize=fw.size()*4;sm.pCode=fw.data();
    if(vkCreateShaderModule(ctx.device,&sm,nullptr,&fs)!=VK_SUCCESS) {
        vkDestroyShaderModule(ctx.device,vs,nullptr);return false;
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType=stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[0].module=vs;stages[1].module=fs;
    stages[0].pName="main";stages[1].pName="main";

    // вершины: position, normal, color — три vec3
    VkVertexInputBindingDescription bd{};bd.binding=0;bd.stride=sizeof(Vertex);
    bd.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)};
    attrs[1]={1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)};
    attrs[2]={2,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color)};
    VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bd;
    vi.vertexAttributeDescriptionCount=3;vi.pVertexAttributeDescriptions=attrs;
    VkPipelineInputAssemblyStateCreateInfo ia{};ia.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // Viewport и scissor — динамические, чтобы не пересоздавать пайплайн при ресайзе
    const VkDynamicState states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};dyn.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount=2;dyn.pDynamicStates=states;
    VkPipelineViewportStateCreateInfo vp{};vp.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount=1;vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rast{};rast.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode=VK_POLYGON_MODE_FILL;rast.cullMode=VK_CULL_MODE_NONE;
    rast.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rast.lineWidth=1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{};depth.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable=VK_TRUE;depth.depthWriteEnable=VK_TRUE;depth.depthCompareOp=VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};blend.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount=1;blend.pAttachments=&att;
    VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;
    pi.pViewportState=&vp;pi.pRasterizationState=&rast;pi.pMultisampleState=&ms;
    pi.pDepthStencilState=&depth;pi.pColorBlendState=&blend;pi.pDynamicState=&dyn;
    pi.layout=gpu.pipeline_layout;pi.renderPass=ctx.render_pass;
    // Первый пайплайн — треугольники (тор и кубы-маркеры)
    const bool solid_ok=vkCreateGraphicsPipelines(ctx.device,VK_NULL_HANDLE,1,&pi,nullptr,&gpu.solid_pipeline)==VK_SUCCESS;
    // Второй пайплайн — линии (конусы прожекторов)
    ia.topology=VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    depth.depthWriteEnable=VK_FALSE;
    const bool line_ok=solid_ok && vkCreateGraphicsPipelines(ctx.device,VK_NULL_HANDLE,1,&pi,nullptr,&gpu.line_pipeline)==VK_SUCCESS;
    vkDestroyShaderModule(ctx.device,fs,nullptr);
    vkDestroyShaderModule(ctx.device,vs,nullptr);
    if(!solid_ok || !line_ok)std::cerr<<"Cannot create Blinn-Phong Vulkan pipelines\n";
    return solid_ok && line_ok;
}

bool createResources() {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    makeTorus(vertices,indices);
    if(!makeMesh(gpu.torus,vertices,indices))return false;
    vertices.clear();indices.clear();
    makeCube(vertices,indices);
    if(!makeMesh(gpu.cube,vertices,indices))return false;
    // UBO для сцены и модели SSBO под списки источников VBO под линии конусов
    if(!createBuffer(sizeof(SceneUniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,gpu.scene_ubo) ||
       !createBuffer(sizeof(ModelUniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,gpu.model_ubo) ||
       !createBuffer(sizeof(PointGpu)*max_lights,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,gpu.points_ssbo) ||
       !createBuffer(sizeof(SpotGpu)*max_lights,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,gpu.spots_ssbo) ||
       !createBuffer(sizeof(Vertex)*max_lights*cone_sides*4,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,gpu.cone_lines)) return false;

    // Пул рассчитан ровно на 2 UBO и 2 SSBO макc 2 сета
    const VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets=2;pi.poolSizeCount=2;pi.pPoolSizes=sizes;
    auto& ctx=graphics::internal::context;
    if(vkCreateDescriptorPool(ctx.device,&pi,nullptr,&gpu.pool)!=VK_SUCCESS)return false;
    // sets[0] — сцена sets[1] — модель.
    const VkDescriptorSetLayout layouts[]={gpu.scene_layout,gpu.model_layout};
    VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool=gpu.pool;ai.descriptorSetCount=2;ai.pSetLayouts=layouts;
    if(vkAllocateDescriptorSets(ctx.device,&ai,gpu.sets.data())!=VK_SUCCESS)return false;

    // Привязка буферов к конкретным биндингам: три в set 0, один в set 1
    const VkDescriptorBufferInfo infos[]={
        {gpu.scene_ubo.handle,0,sizeof(SceneUniform)},
        {gpu.points_ssbo.handle,0,sizeof(PointGpu)*max_lights},
        {gpu.spots_ssbo.handle,0,sizeof(SpotGpu)*max_lights},
        {gpu.model_ubo.handle,0,sizeof(ModelUniform)}
    };
    VkWriteDescriptorSet writes[4]{};
    for(uint32_t i=0;i<4;++i) {
        auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet=gpu.sets[i==3?1:0];w.dstBinding=i==3?0:i;
        w.descriptorType=(i==1||i==2)?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.descriptorCount=1;w.pBufferInfo=&infos[i];
    }
    vkUpdateDescriptorSets(ctx.device,4,writes,0,nullptr);
    return true;
}

ModelUniform buildModel() {
    const float t=static_cast<float>(ui.animation_time);
    Vec3 p=v3(ui.position);
    p=add(p,{ui.trajectory_radius*std::cos(t),ui.trajectory_height*std::sin(2*t),
             ui.trajectory_radius*std::sin(t)});
    // тор стоит точно в заданной позиции,
    // иначе при t = 0 смещение траектории не нулевое
    if(!ui.playing && ui.animation_time==0.0) p=v3(ui.position);
    const Mat4 rot=mmul(ry(ui.rotation[1]+t*ui.spin_speed),
                   mmul(rx(ui.rotation[0]+t*ui.spin_speed*0.35f),rz(ui.rotation[2])));
    const Vec3 s=v3(ui.scale);
    ModelUniform m{};
    m.model=mmul(translation(p),mmul(rot,scale(s)));
    // Нормальная матрица = (M^-1)^T. Для M = T * R * S это R * S^-1
    // трансляция не влияет, R ортогональна, S диагональна
    m.normal_matrix=mmul(rot,scale({1.0f/s.x,1.0f/s.y,1.0f/s.z}));
    m.albedo=vec4(v3(ui.albedo),1.0f);
    m.specular=vec4(v3(ui.specular),ui.shininess);
    return m;
}
SceneUniform buildScene(float aspect) {
    const Vec3 eye=v3(ui.camera_position);
    const Vec3 forward=cameraForward();
    const Mat4 view=lookAt(eye,add(eye,forward),{0,1,0});
    const Mat4 projection=ui.projection==0 ? perspective(ui.fov,aspect,0.1f,150.0f)
                                               : ortho(ui.ortho_size,aspect,0.1f,150.0f);
    SceneUniform s{};
    s.view_projection=mmul(projection,view);
    s.camera_position=vec4(eye,1.0f);
    s.ambient=vec4(v3(ui.ambient),ui.ambient_intensity);
    s.sun_direction=vec4(normal(v3(ui.sun_direction)));
    s.sun_color=vec4(v3(ui.sun_color),ui.sun_intensity);
    s.counts[0]=static_cast<int>(ui.points.size());
    s.counts[1]=static_cast<int>(ui.spots.size());
    return s;
}

// Строим линии-конусы для всех прожекторов
std::vector<Vertex> buildConeLines() {
    std::vector<Vertex> lines;
    lines.reserve(ui.spots.size()*cone_sides*4);
    for(const auto& spot:ui.spots) {
        const Vec3 origin=v3(spot.position);
        const Vec3 d=normal(v3(spot.direction));
        // Ортогональный базис вокруг направления ref выбираем так, чтобы cross не был нулём
        const Vec3 ref=(std::abs(d.y)<0.95f)?Vec3{0,1,0}:Vec3{1,0,0};
        const Vec3 right=normal(cross(d,ref)), up=normal(cross(right,d));
        constexpr float length=1.7f;
        const float radius=length*std::tan(radians(spot.outer_angle));
        const Vec3 center=add(origin,mul(d,length));
        for(int j=0;j<cone_sides;++j) {
            const auto rim=[&](int i) {
                const float a=two_pi*float(i)/float(cone_sides);
                return add(center,add(mul(right,radius*std::cos(a)),mul(up,radius*std::sin(a))));
            };
            const Vec3 current=rim(j), next=rim((j+1)%cone_sides);
            const Vertex a{origin,{0,1,0},{1,1,1}};
            const Vertex b{current,{0,1,0},{1,1,1}};
            const Vertex c{next,{0,1,0},{1,1,1}};
            lines.insert(lines.end(),{a,b,b,c});
        }
    }
    return lines;
}
// Пересчёт и заливка всех буферов перед записью команд
void uploadFrame(float aspect) {
    const SceneUniform scene=buildScene(aspect);
    const ModelUniform model=buildModel();
    upload(gpu.scene_ubo,&scene,sizeof(scene));
    upload(gpu.model_ubo,&model,sizeof(model));
    if(!ui.points.empty()) {
        std::vector<PointGpu> list;list.reserve(ui.points.size());
        for(const auto& p:ui.points) list.push_back({vec4(v3(p.position),1.0f),vec4(v3(p.color),p.intensity)});
        upload(gpu.points_ssbo,list.data(),list.size()*sizeof(PointGpu));
    }
    if(!ui.spots.empty()) {
        std::vector<SpotGpu> list;list.reserve(ui.spots.size());
        // cosуглов чтобы шейдеру не пришлось считать acos на каждом фрагменте
        for(const auto& p:ui.spots) list.push_back({
            vec4(v3(p.position),1.0f),vec4(normal(v3(p.direction))),
            vec4(v3(p.color),p.intensity),
            {std::cos(radians(p.inner_angle)),std::cos(radians(p.outer_angle)),0,0}
        });
        upload(gpu.spots_ssbo,list.data(),list.size()*sizeof(SpotGpu));
    }
    if(ui.show_spot_cones && !ui.spots.empty()) {
        const std::vector<Vertex> cone=buildConeLines();
        upload(gpu.cone_lines,cone.data(),cone.size()*sizeof(Vertex));
    }
}
void drawMesh(VkCommandBuffer cmd, const Mesh& mesh) {
    const VkDeviceSize zero=0;
    vkCmdBindVertexBuffers(cmd,0,1,&mesh.vertices.handle,&zero);
    vkCmdBindIndexBuffer(cmd,mesh.indices.handle,0,VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd,mesh.index_count,1,0,0,0);
}
// Отправка push constants перед очередным draw cal
void pushDraw(VkCommandBuffer cmd, const Mat4& model, Vec4 emission, bool unlit) {
    DrawPush push{};push.model=model;push.emission=emission;push.unlit=unlit?1:0;
    vkCmdPushConstants(cmd,gpu.pipeline_layout,
        VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(push),&push);
}
// Управление камеро
void moveCamera(GLFWwindow* window,double dt) {
    if(!window)return;
    const ImGuiIO& io=ImGui::GetIO();
    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS && !io.WantCaptureMouse) {
        double x=0.0,y=0.0;glfwGetCursorPos(window,&x,&y);
        if(ui.last_mouse_valid) {
            ui.yaw+=static_cast<float>((x-ui.mouse_x)*0.12);
            ui.pitch-=static_cast<float>((y-ui.mouse_y)*0.12);
            // Ограничение pitch, иначе gimbal lock на ±90°.
            ui.pitch=std::clamp(ui.pitch,-89.0f,89.0f);
        }
        ui.mouse_x=x;ui.mouse_y=y;ui.last_mouse_valid=true;
    } else ui.last_mouse_valid=false;
    if(io.WantCaptureKeyboard)return;
    const Vec3 forward=cameraForward();
    const Vec3 right=normal(cross(forward,{0,1,0}));
    Vec3 delta{};
    if(glfwGetKey(window,GLFW_KEY_W)==GLFW_PRESS)delta=add(delta,forward);
    if(glfwGetKey(window,GLFW_KEY_S)==GLFW_PRESS)delta=sub(delta,forward);
    if(glfwGetKey(window,GLFW_KEY_A)==GLFW_PRESS)delta=sub(delta,right);
    if(glfwGetKey(window,GLFW_KEY_D)==GLFW_PRESS)delta=add(delta,right);
    if(glfwGetKey(window,GLFW_KEY_E)==GLFW_PRESS)delta.y+=1;
    if(glfwGetKey(window,GLFW_KEY_Q)==GLFW_PRESS)delta.y-=1;
    if(dot(delta,delta)>1e-6f) {
        // Нормализация
        const Vec3 next=add(v3(ui.camera_position),mul(normal(delta),ui.camera_speed*static_cast<float>(dt)));
        ui.camera_position[0]=next.x;ui.camera_position[1]=next.y;ui.camera_position[2]=next.z;
    }
}
void pointControls() {
    if(ImGui::Button("Add point light") && ui.points.size()<max_lights)ui.points.emplace_back();
    ImGui::SameLine();ImGui::Text("%zu / %u",ui.points.size(),max_lights);
    // Ручной цикл erase инвалидирует итераторы, поэтому нельзя range-based for.
    for(size_t i=0;i<ui.points.size();) {
        // Уникальный id на элемент иначе путает поля
        ImGui::PushID(static_cast<int>(i)+1000);
        const bool open=ImGui::TreeNode("Point light");
        bool remove=false;
        if(open) {
            ImGui::DragFloat3("Position",ui.points[i].position,0.06f,-30,30);
            ImGui::ColorEdit3("Color",ui.points[i].color);
            ImGui::SliderFloat("Intensity",&ui.points[i].intensity,0,100);
            remove=ImGui::Button("Delete point light");
            ImGui::TreePop();
        }
        ImGui::PopID();
        if(remove)ui.points.erase(ui.points.begin()+static_cast<std::ptrdiff_t>(i));
        else ++i;
    }
}
void spotControls() {
    if(ImGui::Button("Add spotlight") && ui.spots.size()<max_lights)ui.spots.emplace_back();
    ImGui::SameLine();ImGui::Text("%zu / %u",ui.spots.size(),max_lights);
    for(size_t i=0;i<ui.spots.size();) {
        ImGui::PushID(static_cast<int>(i)+2000);
        const bool open=ImGui::TreeNode("Spotlight");
        bool remove=false;
        if(open) {
            auto& s=ui.spots[i];
            ImGui::DragFloat3("Position",s.position,0.06f,-30,30);
            ImGui::DragFloat3("Direction",s.direction,0.03f,-3,3);
            ImGui::ColorEdit3("Color",s.color);
            ImGui::SliderFloat("Intensity",&s.intensity,0,150);
            ImGui::SliderFloat("Inner angle",&s.inner_angle,1,79);
            ImGui::SliderFloat("Outer angle",&s.outer_angle,1,85);
            //  внешний угол не меньше внутреннего, иначе сломается.
            if(s.outer_angle<s.inner_angle)s.outer_angle=s.inner_angle;
            remove=ImGui::Button("Delete spotlight");
            ImGui::TreePop();
        }
        ImGui::PopID();
        if(remove)ui.spots.erase(ui.spots.begin()+static_cast<std::ptrdiff_t>(i));
        else ++i;
    }
}

} // namespace

bool initialize() {
    // Сначала пайплайны (создают layout), потом ресурсы (аллоцируют сеты по layout).
    if(!createPipelines() || !createResources()) {
        std::cerr<<"Torus lighting initialization failed\n";
        shutdown();return false;
    }
    return true;
}
void shutdown() {
    auto& ctx=graphics::internal::context;
    if(ctx.graphics_queue)vkQueueWaitIdle(ctx.graphics_queue);
    destroyMesh(gpu.torus);destroyMesh(gpu.cube);
    destroyBuffer(gpu.cone_lines);
    destroyBuffer(gpu.scene_ubo);destroyBuffer(gpu.model_ubo);
    destroyBuffer(gpu.points_ssbo);destroyBuffer(gpu.spots_ssbo);
    if(gpu.pool)vkDestroyDescriptorPool(ctx.device,gpu.pool,nullptr);
    if(gpu.solid_pipeline)vkDestroyPipeline(ctx.device,gpu.solid_pipeline,nullptr);
    if(gpu.line_pipeline)vkDestroyPipeline(ctx.device,gpu.line_pipeline,nullptr);
    if(gpu.pipeline_layout)vkDestroyPipelineLayout(ctx.device,gpu.pipeline_layout,nullptr);
    if(gpu.scene_layout)vkDestroyDescriptorSetLayout(ctx.device,gpu.scene_layout,nullptr);
    if(gpu.model_layout)vkDestroyDescriptorSetLayout(ctx.device,gpu.model_layout,nullptr);
    gpu=GpuResources{};
}
void update(double time, GLFWwindow* window) {
    double dt=0.0;
    if(ui.previous_time>=0.0)dt=std::clamp(time-ui.previous_time,0.0,0.1);
    ui.previous_time=time;
    if(ui.playing)ui.animation_time+=dt*ui.speed;

    ImGui::SetNextWindowSize(ImVec2(430,640),ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2 - Torus / Blinn-Phong");
    ImGui::Text("WASD: move, Q/E: down/up, RMB drag: look");
    if(ImGui::CollapsingHeader("Camera",ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* modes[]={"Perspective","Orthographic"};
        ImGui::Combo("Projection",&ui.projection,modes,2);
        ImGui::DragFloat3("Camera position",ui.camera_position,0.04f,-100,100);
        ImGui::SliderFloat("Yaw",&ui.yaw,-180,180);
        ImGui::SliderFloat("Pitch",&ui.pitch,-89,89);
        ImGui::SliderFloat("Field of view",&ui.fov,20,110);
        if(ui.projection==1)ImGui::SliderFloat("Ortho half size",&ui.ortho_size,1,15);
        ImGui::SliderFloat("Camera move speed",&ui.camera_speed,0.2f,18);
        if(ImGui::Button("Reset camera")) {
            ui.camera_position[0]=0;ui.camera_position[1]=2.8f;ui.camera_position[2]=8;
            ui.yaw=-90;ui.pitch=-19.3f;ui.fov=55;
        }
    }
    if(ImGui::CollapsingHeader("Torus transform and animation")) {
        ImGui::DragFloat3("Position",ui.position,0.03f,-15,15);
        ImGui::DragFloat3("Rotation degrees",ui.rotation,0.5f,-360,360);
        ImGui::DragFloat3("Scale",ui.scale,0.01f,0.05f,6);
        if(ImGui::Button(ui.playing?"Pause":"Play"))ui.playing=!ui.playing;
        ImGui::SameLine();
        if(ImGui::Button("Restart animation"))ui.animation_time=0;
        ImGui::SliderFloat("Animation speed",&ui.speed,0,4);
        ImGui::SliderFloat("Path radius",&ui.trajectory_radius,0,3);
        ImGui::SliderFloat("Path height",&ui.trajectory_height,0,3);
        ImGui::SliderFloat("Spin speed",&ui.spin_speed,-240,240);
    }
    if(ImGui::CollapsingHeader("Material",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo",ui.albedo);
        ImGui::ColorEdit3("Specular color",ui.specular);
        ImGui::SliderFloat("Shininess",&ui.shininess,1.0f,256.0f,"%.0f");
        ImGui::TextUnformatted("Albedo multiplies procedural vertex colors.");
    }
    if(ImGui::CollapsingHeader("Ambient and directional light",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color",ui.ambient);
        ImGui::SliderFloat("Ambient intensity",&ui.ambient_intensity,0,3);
        ImGui::DragFloat3("Sun direction",ui.sun_direction,0.02f,-3,3);
        ImGui::ColorEdit3("Sun color",ui.sun_color);
        ImGui::SliderFloat("Sun intensity",&ui.sun_intensity,0,6);
    }
    if(ImGui::CollapsingHeader("Point lights (SSBO)"))pointControls();
    if(ImGui::CollapsingHeader("Spotlights (SSBO)"))spotControls();
    if(ImGui::CollapsingHeader("Light visualization",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show light cubes",&ui.show_light_markers);
        ImGui::Checkbox("Show spotlight cones",&ui.show_spot_cones);
    }
    ImGui::Separator();
    ImGui::Text("Torus: %u vertices, %u triangles",torus_major_segments*torus_tube_segments,
                2*torus_major_segments*torus_tube_segments);
    ImGui::End();
    moveCamera(window,dt);
}
void render(const graphics::internal::FrameData& fd) {
    auto& ctx=graphics::internal::context;
    if(!fd.command_buffer || !fd.framebuffer || !ctx.swapchain_extent.width || !ctx.swapchain_extent.height)return;
    const float aspect=float(ctx.swapchain_extent.width)/float(ctx.swapchain_extent.height);
    //  уже дождался предыдущего кадра,
    // можно безопасно перезаписывать UBO.
    uploadFrame(aspect);
    vkResetCommandBuffer(fd.command_buffer,0);
    VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(fd.command_buffer,&begin)!=VK_SUCCESS)return;
    VkClearValue clear[2]{};
    clear[0].color={{0.055f,0.063f,0.09f,1.0f}};
    clear[1].depthStencil={1.0f,0}; // Z = 1 — дальняя граница.
    VkRenderPassBeginInfo pass{};pass.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass=ctx.render_pass;pass.framebuffer=fd.framebuffer;
    pass.renderArea.extent=ctx.swapchain_extent;pass.clearValueCount=2;pass.pClearValues=clear;
    vkCmdBeginRenderPass(fd.command_buffer,&pass,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{};vp.width=float(ctx.swapchain_extent.width);vp.height=float(ctx.swapchain_extent.height);
    vp.minDepth=0;vp.maxDepth=1;
    VkRect2D sc{};sc.extent=ctx.swapchain_extent;
    vkCmdSetViewport(fd.command_buffer,0,1,&vp);
    vkCmdSetScissor(fd.command_buffer,0,1,&sc);
    // Биндим оба сета одним вызовом: set 0 — сцена, set 1 — модель.
    vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpu.pipeline_layout,0,2,gpu.sets.data(),0,nullptr);
    vkCmdBindPipeline(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,gpu.solid_pipeline);
    pushDraw(fd.command_buffer,identity(),{0,0,0,0},false);
    drawMesh(fd.command_buffer,gpu.torus);

    if(ui.show_light_markers) {
        // Кубик-маркер: T * S, emission = цвет источника, unlit = true.
        const auto drawLight=[&](const float pos[3],const float col[3]) {
            const Mat4 m=mmul(translation(v3(pos)),scale({0.24f,0.24f,0.24f}));
            pushDraw(fd.command_buffer,m,vec4(v3(col),1.0f),true);
            drawMesh(fd.command_buffer,gpu.cube);
        };
        for(const auto& p:ui.points)drawLight(p.position,p.color);
        for(const auto& s:ui.spots)drawLight(s.position,s.color);
    }
    if(ui.show_spot_cones && !ui.spots.empty()) {
        // Линии конусов: отдельный пайплайн, без индексного буфера.
        vkCmdBindPipeline(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,gpu.line_pipeline);
        pushDraw(fd.command_buffer,identity(),{1,1,1,1},true);
        const VkDeviceSize off=0;
        vkCmdBindVertexBuffers(fd.command_buffer,0,1,&gpu.cone_lines.handle,&off);
        vkCmdDraw(fd.command_buffer,static_cast<uint32_t>(ui.spots.size()*cone_sides*4),1,0,0);
    }
    vkCmdEndRenderPass(fd.command_buffer);
    if(vkEndCommandBuffer(fd.command_buffer)!=VK_SUCCESS)
        std::cerr<<"Failed to record torus lighting commands\n";
}

} // namespace application