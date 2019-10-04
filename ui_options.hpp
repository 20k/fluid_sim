#ifndef UI_OPTIONS_HPP_INCLUDED
#define UI_OPTIONS_HPP_INCLUDED

namespace options
{
    enum options
    {
        FLUID,
        SAND,
        RIGID,
        COUNT
    };
}

struct ui_options
{
    options::options brush = options::COUNT;

    void tick();
};

#endif // UI_OPTIONS_HPP_INCLUDED
