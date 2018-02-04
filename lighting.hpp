#ifndef LIGHTING_HPP_INCLUDED
#define LIGHTING_HPP_INCLUDED

struct lighting_manager
{
    cl::buffer* colour;

    vec2i saved_dim;

    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue, vec2i screen_dim)
    {
        colour = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> colour_buf;

        for(int y=0; y < screen_dim.y(); y++)
        {
            for(int x=0; x < screen_dim.x(); x++)
            {
                colour_buf.push_back({0,0,0,1});
            }
        }

        colour->alloc_img(cqueue, colour_buf, screen_dim, CL_RGBA, CL_FLOAT);

        saved_dim = screen_dim;
    }

    void tick(cl::cl_gl_interop_texture* interop, cl::buffer_manager& buffers,cl::command_queue& cqueue, vec2f mpos, cl::buffer* fluid)
    {
        mpos.y() = saved_dim.y() - mpos.y();

        float rad = 800.f;

        int num_tracers = ceil(2 * M_PI * rad)*2.5;

        cl::args raytrace_args;
        raytrace_args.push_back(mpos);
        raytrace_args.push_back(rad);
        raytrace_args.push_back(num_tracers);
        raytrace_args.push_back(fluid);
        raytrace_args.push_back(interop);

        cqueue.exec("lighting_raytrace_point", raytrace_args, {num_tracers}, {128});
    }
};

#endif // LIGHTING_HPP_INCLUDED
