#pragma once

#include "graphics_internal.hpp"

#include <cstdint>

namespace application {

// Вариант 7: тор. Число вершин: 64 * 32 = 2048,
// число треугольников: 64 * 32 * 2 = 4096.
inline constexpr std::uint32_t torus_major_segments = 64;
inline constexpr std::uint32_t torus_tube_segments = 32;
inline constexpr float torus_major_radius = 1.45f;
inline constexpr float torus_tube_radius = 0.52f;

// Вызываются из готового GLFW + ImGui + Vulkan цикла в main.cpp.
bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application
