#ifndef FLUID_HPP_INCLUDED
#define FLUID_HPP_INCLUDED

struct fluid_particle
{
    vec2f pos = {0,0};
};

struct fluid_manager
{
    int which_vel = 0;
    int which_pressure = 0;

    cl::buffer* velocity[2];
    cl::buffer* pressure[2];

    cl::buffer* divergence;

    cl::buffer* dye[2];

    cl::buffer* fluid_particles;
    std::vector<fluid_particle> cpu_particles;

    int which_dye = 0;

    vec2i velocity_dim = {0,0};
    vec2i dye_dim = {0,0};

    vec2f velocity_to_display_ratio = {0,0};

    ///TODO:
    ///decouple fluid dye resolution (aka actual interesting quantity) from underlying fluid simulation
    ///so we can run it at lower res
    ///TODO:
    ///do wavelet fun
    ///TODO:
    ///investigate not using jacobi
    ///TODO:
    ///make fluid particles look nice
    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue, vec2i vdim, vec2i ddim)
    {
        velocity_dim = vdim;
        dye_dim = ddim;

        velocity_to_display_ratio = (vec2f){dye_dim.x(), dye_dim.y()} / (vec2f){velocity_dim.x(), velocity_dim.y()};

        velocity[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        velocity[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        pressure[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        pressure[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        divergence = buffers.fetch<cl::buffer>(ctx, nullptr);

        dye[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        dye[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        fluid_particles = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> zero_data;

        std::vector<vec4f> dye_concentrates;

        std::vector<vec2f> velocity_info;

        std::vector<float> zero;

        for(int y=0; y < velocity_dim.y(); y++)
        for(int x=0; x < velocity_dim.x(); x++)
        {
            vec2f centre = {velocity_dim.x()/2.f, velocity_dim.y()/2.f};

            vec2f fluid_val = randv<2, float>(-0.2f, 0.2f);

            velocity_info.push_back(fluid_val);

            zero_data.push_back({0,0,0,0});
        }

        for(int y=0; y < dye_dim.y(); y++)
        for(int x=0; x < dye_dim.x(); x++)
        {
            vec2f centre = {dye_dim.x()/2.f, dye_dim.y()/2.f};

            vec2f dye_val;

            dye_val.x() = ((vec2f){x, y} - centre).length() / dye_dim.length();

            dye_val += fabs(randv<2, float>(-0.2f, 0.2f));

            dye_concentrates.push_back({dye_val.x(), dye_val.y(), 0.f, 1.f});
        }

        velocity[0]->alloc_img(cqueue, velocity_info, velocity_dim, CL_RG, CL_FLOAT);
        velocity[1]->alloc_img(cqueue, velocity_info, velocity_dim, CL_RG, CL_FLOAT);

        pressure[0]->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_HALF_FLOAT);
        pressure[1]->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_HALF_FLOAT);

        divergence->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_HALF_FLOAT);

        dye[0]->alloc_img(cqueue, dye_concentrates, dye_dim);
        dye[1]->alloc_img(cqueue, dye_concentrates, dye_dim);

        for(int i=0; i < 10000; i++)
        {
            vec2f pos = randv<2, float>(0, 600);

            cpu_particles.push_back({pos});
        }

        fluid_particles->alloc(cqueue, cpu_particles);
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

    void velocity_boundary(cl::program& program, cl::command_queue& cqueue)
    {
        cl::buffer* v1 = get_velocity_buf(0);

        float scale = -1;

        cl::args vel_args;
        vel_args.push_back(v1);
        vel_args.push_back(v1);
        vel_args.push_back(scale);

        cqueue.exec(program, "fluid_boundary", vel_args, velocity_dim, {16, 16});
    }

    void pressure_boundary(cl::program& program, cl::command_queue& cqueue)
    {
        cl::buffer* v1 = get_pressure_buf(0);
        cl::buffer* v2 = get_pressure_buf(1);

        float scale = 1;

        cl::args vel_args;
        vel_args.push_back(v1);
        vel_args.push_back(v1);
        vel_args.push_back(scale);

        cqueue.exec(program, "fluid_boundary", vel_args, velocity_dim, {16, 16});
    }

    void advect_quantity(cl::buffer* quantity[2], int& which, cl::program& program, cl::command_queue& cqueue, float timestep_s, vec2i dim)
    {
        cl::buffer* q1 = quantity[which];
        cl::buffer* q2 = quantity[(which + 1) % 2];

        cl::buffer* v1 = get_velocity_buf(0);

        cl::args advect_args;

        advect_args.push_back(v1);
        advect_args.push_back(q1);
        advect_args.push_back(q2);
        advect_args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advection", advect_args, dim, {16, 16});

        which = (which + 1) % 2;
    }

    void apply_force(cl::program& program, cl::command_queue& cqueue, float force, vec2f location, vec2f direction)
    {
        cl::buffer* v1 = get_velocity_buf(0);
        //cl::buffer* v2 = get_velocity_buf(1);

        location.y() = dye_dim.y() - location.y();
        direction.y() = -direction.y();

        location = location / velocity_to_display_ratio;

        cl::args force_args;
        force_args.push_back(v1);
        force_args.push_back(v1);
        force_args.push_back(force);
        force_args.push_back(location);
        force_args.push_back(direction);

        cqueue.exec(program, "fluid_apply_force", force_args, velocity_dim, {16, 16});

        //flip_velocity();
        velocity_boundary(program, cqueue);
    }

    void handle_particles(cl::cl_gl_interop_texture* interop, cl::program& program, cl::command_queue& cqueue, float timestep_s)
    {
        interop->acquire(cqueue);

        cl::buffer* v1 = get_velocity_buf(0);

        int num_particles = cpu_particles.size();

        vec2f scale = velocity_to_display_ratio;

        cl::args advect_args;
        advect_args.push_back(v1);
        advect_args.push_back(fluid_particles);
        advect_args.push_back(num_particles);
        advect_args.push_back(timestep_s);
        advect_args.push_back(scale);

        cqueue.exec(program, "fluid_advect_particles", advect_args, {num_particles}, {128});

        cl::args render_args;
        render_args.push_back(fluid_particles);
        render_args.push_back(num_particles);
        render_args.push_back(interop);
        //render_args.push_back(scale);

        cqueue.exec(program, "fluid_render_particles", render_args, {num_particles}, {128});
    }

    void tick(cl::cl_gl_interop_texture* interop, cl::buffer_manager& buffers, cl::program& program, cl::command_queue& cqueue)
    {
        float timestep_s = 4600.f/1000.f;

        cl::buffer* v1 = get_velocity_buf(0);
        cl::buffer* v2 = get_velocity_buf(1);

        cl::args advect_args;
        advect_args.push_back(v1);
        advect_args.push_back(v1);
        advect_args.push_back(v2);
        advect_args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advection", advect_args, velocity_dim, {16, 16});

        flip_velocity();

        velocity_boundary(program, cqueue);

        advect_quantity(dye, which_dye, program, cqueue, timestep_s, dye_dim);

        int jacobi_iterations_diff = 10;

        float dx = 1.f;

        for(int i=0; i < jacobi_iterations_diff; i++)
        {
            float viscosity = 0.0000001f;

            float vdt = viscosity * timestep_s;

            float alpha = dx * dx / vdt;
            float rbeta = 1.f / (4 + alpha);

            cl::buffer* dv1 = get_velocity_buf(0);
            cl::buffer* dv2 = get_velocity_buf(1);

            cl::args diffuse_args;

            diffuse_args.push_back(dv1);
            diffuse_args.push_back(dv1);
            diffuse_args.push_back(dv2);
            diffuse_args.push_back(alpha);
            diffuse_args.push_back(rbeta);

            cqueue.exec(program, "fluid_jacobi", diffuse_args, velocity_dim, {16, 16});

            flip_velocity();

            //velocity_boundary(program, cqueue);
        }


        velocity_boundary(program, cqueue);

        ///so. First we caclculate divergence, as bvector
        ///then we calculate x, which is the pressure, which is blank initially
        ///then we jacobi iterate the pressure

        cl::buffer* cv1 = get_velocity_buf(0);

        cl::args divergence_args;
        divergence_args.push_back(cv1);
        divergence_args.push_back(divergence);

        cqueue.exec(program, "fluid_divergence", divergence_args, velocity_dim, {16, 16});

        int pressure_iterations_diff = 20;

        ///source of slowdown
        ///need the ability to create specific textures
        ///aka we want half float single channel
        ///not full float quad channel
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

            cqueue.exec(program, "fluid_jacobi", pressure_args, velocity_dim, {16, 16});

            flip_pressure();

            //pressure_boundary(program, cqueue);
        }

        pressure_boundary(program, cqueue);

        cl::buffer* cpressure = get_pressure_buf(0);
        cl::buffer* cur_v1 = get_velocity_buf(0);
        cl::buffer* cur_v2 = get_velocity_buf(1);

        cl::args subtract_args;
        subtract_args.push_back(cpressure);
        subtract_args.push_back(cur_v1);
        subtract_args.push_back(cur_v2);

        cqueue.exec(program, "fluid_gradient", subtract_args, velocity_dim, {16, 16});

        flip_velocity();

        velocity_boundary(program, cqueue);

        interop->acquire(cqueue);


        //cl::buffer* debug_velocity = get_velocity_buf(0);

        cl::buffer* debug_velocity = dye[which_dye];

        cl::args debug;
        debug.push_back(debug_velocity);
        debug.push_back(interop);

        cqueue.exec(program, "fluid_render", debug, dye_dim, {16, 16});

        handle_particles(interop, program, cqueue, timestep_s);

    }
};

#endif // FLUID_HPP_INCLUDED
