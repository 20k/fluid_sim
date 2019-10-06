#ifndef UI_OPTIONS_HPP_INCLUDED
#define UI_OPTIONS_HPP_INCLUDED

#include <vector>
#include <string>
#include <vec/vec.hpp>

namespace options
{
    enum options
    {
        FLUID_VELOCITY,
        FLUID_DYE,
        SAND,
        RIGID,
        BOUNDARY,
        COUNT
    };

    inline
    const char* names[]
    {
        "Fluid Velocity",
        "Fluid Dye",
        "Sand",
        "Rigidbody",
        "Boundary",
    };
}

struct ui_options
{
    int brush = options::SAND;
    int brush_size = 1;
    vec3f col = {1,1,1};

    void tick();
};

#endif // UI_OPTIONS_HPP_INCLUDED
