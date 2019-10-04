#ifndef UI_OPTIONS_HPP_INCLUDED
#define UI_OPTIONS_HPP_INCLUDED

#include <vector>
#include <string>

namespace options
{
    enum options
    {
        FLUID,
        SAND,
        RIGID,
        BOUNDARY,
        COUNT
    };

    inline
    const char* names[]
    {
        "Fluid",
        "Sand",
        "Rigidbody",
        "Boundary",
    };
}

struct ui_options
{
    int brush = options::SAND;
    int brush_size = 1;

    void tick();
};

#endif // UI_OPTIONS_HPP_INCLUDED
