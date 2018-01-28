#ifndef FLUID_HPP_INCLUDED
#define FLUID_HPP_INCLUDED

struct fluid_manager
{
    int which_vel = 0;
    int which_pressure = 0;

    cl::buffer* velocity[2];
    cl::buffer* pressure[2];

    cl::buffer* divergence;

    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue)
    {
        velocity[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        velocity[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        pressure[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        pressure[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        divergence = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> idata;
        std::vector<vec4f> zero_data;

        for(int i=0; i < 800*600; i++)
        {
            idata.push_back({randf_s(-0.3f, 0.3f) + (float)i / (800 * 600), 0, 0, 1});

            zero_data.push_back({0,0,0,0});
        }

        velocity[0]->alloc_img(cqueue, idata, (vec2i) {800, 600});
        velocity[1]->alloc_img(cqueue, idata, (vec2i) {800, 600});

        pressure[0]->alloc_img(cqueue, zero_data, (vec2i) {800, 600});
        pressure[1]->alloc_img(cqueue, zero_data, (vec2i) {800, 600});

        divergence->alloc_img(cqueue, zero_data, (vec2i) {800, 600});
    }

    cl::buffer* get_velocity_buf(int offset)
    {
        return velocity[(which_vel + offset) % 2];
    }

    void flip_velocity()
    {
        which_vel = (which_vel + 1) % 2;
    }

    cl::buffer* get_pressure_buf(int offset)
    {
        return pressure[(which_pressure + offset) % 2];
    }

    void flip_pressure()
    {
        which_pressure = (which_pressure + 1) % 2;
    }

    void tick(cl::cl_gl_interop_texture* interop, cl::buffer_manager& buffers, cl::program& program, cl::command_queue& cqueue)
    {
        float timestep_s = 16.f/1000.f;

        cl::buffer* v1 = get_velocity_buf(0);
        cl::buffer* v2 = get_velocity_buf(1);

        cl::args advect_args;
        advect_args.push_back(v1);
        advect_args.push_back(v1);
        advect_args.push_back(v2);
        advect_args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advection", advect_args, {800, 600}, {16, 16});

        flip_velocity();

        int jacobi_iterations_diff = 4;

        float dx = 1.f;

        for(int i=0; i < jacobi_iterations_diff; i++)
        {
            float viscosity = 0.01f;

            float vdt = viscosity * timestep_s;

            float alpha = dx * dx / vdt;
            float rbeta = 1.f / (4 + alpha);

            //printf("%f\n", rbeta);

            cl::buffer* dv1 = get_velocity_buf(0);
            cl::buffer* dv2 = get_velocity_buf(1);

            cl::args diffuse_args;

            diffuse_args.push_back(dv1);
            diffuse_args.push_back(dv1);
            diffuse_args.push_back(dv2);
            diffuse_args.push_back(alpha);
            diffuse_args.push_back(rbeta);

            cqueue.exec(program, "fluid_jacobi", diffuse_args, {800, 600}, {16, 16});

            flip_velocity();
        }

        ///so. First we caclculate divergence, as bvector
        ///then we calculate x, which is the pressure, which is blank initially
        ///then we jacobi iterate the pressure

        cl::buffer* cv1 = get_velocity_buf(0);

        cl::args divergence_args;
        divergence_args.push_back(cv1);
        divergence_args.push_back(divergence);

        cqueue.exec(program, "fluid_divergence", divergence_args, {800, 600}, {16, 16});

        int pressure_iterations_diff = 20;

        for(int i=0; i < pressure_iterations_diff; i++)
        {
            float alpha = -(dx * dx);
            float rbeta = 1/4.f;

            cl::buffer* p1 = get_pressure_buf(0);
            cl::buffer* p2 = get_pressure_buf(1);

            cl::args pressure_args;
            pressure_args.push_back(p1);
            pressure_args.push_back(divergence);
            pressure_args.push_back(p2);
            pressure_args.push_back(alpha);
            pressure_args.push_back(rbeta);

            cqueue.exec(program, "fluid_jacobi", pressure_args, {800, 600}, {16, 16});

            flip_pressure();
        }

        cl::buffer* cpressure = get_pressure_buf(0);
        cl::buffer* cur_v1 = get_velocity_buf(0);
        cl::buffer* cur_v2 = get_velocity_buf(1);

        cl::args subtract_args;
        subtract_args.push_back(cpressure);
        subtract_args.push_back(cur_v1);
        subtract_args.push_back(cur_v2);

        cqueue.exec(program, "fluid_gradient", subtract_args, {800, 600}, {16, 16});

        flip_velocity();

        interop->acquire(cqueue);


        cl::buffer* debug_velocity = get_velocity_buf(0);

        cl::args debug;
        debug.push_back(debug_velocity);
        debug.push_back(interop);

        cqueue.exec(program, "fluid_render", debug, {800, 600}, {16, 16});

    }
};

#endif // FLUID_HPP_INCLUDED
