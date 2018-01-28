#ifndef FLUID_HPP_INCLUDED
#define FLUID_HPP_INCLUDED

struct fluid_manager
{
    int which_vel = 0;

    cl::buffer* velocity[2];

    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue)
    {
        velocity[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        velocity[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> idata;

        for(int i=0; i < 800*600; i++)
        {
            idata.push_back({randf_s(-0.01f, 0.01f) + (float)i / (800 * 600), 0, 0, 1});
        }

        velocity[0]->alloc_img(cqueue, idata, (vec2i){800, 600});
        velocity[1]->alloc_img(cqueue, idata, (vec2i){800, 600});
    }

    cl::buffer* get_velocity_buf(int offset)
    {
        return velocity[(which_vel + offset) % 2];
    }

    void flip_velocity()
    {
        which_vel = (which_vel + 1) % 2;
    }

    void tick(cl::cl_gl_interop_texture* interop, cl::buffer_manager& buffers, cl::program& program, cl::command_queue& cqueue)
    {
        float timestep_s = 16.f/1000.f;

        cl::buffer* v1 = get_velocity_buf(0);
        cl::buffer* v2 = get_velocity_buf(1);

        cl::args args;
        args.push_back(v1);
        args.push_back(v1);
        args.push_back(v2);
        args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advection", args, {800, 600}, {16, 16});


        interop->acquire(cqueue);

        cl::args debug;
        debug.push_back(v2);
        debug.push_back(interop);

        cqueue.exec(program, "fluid_render", debug, {800, 600}, {16, 16});

        flip_velocity();
    }
};

#endif // FLUID_HPP_INCLUDED
