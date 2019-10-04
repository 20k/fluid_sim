#include "ui_options.hpp"
#include <imgui/imgui.h>

void ui_options::tick()
{
    ImGui::Begin("Tools");

    ImGui::Combo("Tools", &brush, options::names, options::COUNT);

    ImGui::End();
}
