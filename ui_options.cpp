#include "ui_options.hpp"
#include <imgui/imgui.h>
#include <vec/vec.hpp>

void ui_options::tick()
{
    ImGui::Begin("Tools");

    ImGui::Combo("Tools", &brush, options::names, options::COUNT);

    ImGui::SliderInt("Brush Width", &brush_size, 1, 10);

    brush_size = clamp(brush_size, 1, 10);

    ImGui::End();
}
