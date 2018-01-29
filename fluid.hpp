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

    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue)
    {
        velocity[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        velocity[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        pressure[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        pressure[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        divergence = buffers.fetch<cl::buffer>(ctx, nullptr);

        dye[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        dye[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        fluid_particles = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> idata;
        std::vector<vec4f> zero_data;

        std::vector<vec4f> dye_concentrates;

        //for(int i=0; i < 800*600; i++)

        for(int y=0; y < 600; y++)
        for(int x=0; x < 800; x++)
        {
            vec2f centre = {400, 300};

            vec2f fluid_val = {0,0};
            vec2f dye_val = {0,0};

            dye_val.x() = ((vec2f){x, y} - centre).length() / 600.f;

            fluid_val.x() += randf_s(-0.2f, 0.2f);
            fluid_val.y() += randf_s(-0.2f, 0.2f);

            dye_val += fluid_val;

            idata.push_back({fluid_val.x(), fluid_val.y(), 0, 1});

            zero_data.push_back({0,0,0,0});

            dye_concentrates.push_back({fabs(dye_val.x()), fabs(dye_val.y()), 0, 1});
        }

        velocity[0]->alloc_img(cqueue, idata, (vec2i) {800, 600});
        velocity[1]->alloc_img(cqueue, idata, (vec2i) {800, 600});

        pressure[0]->alloc_img(cqueue, zero_data, (vec2i) {800, 600});
        pressure[1]->alloc_img(cqueue, zero_data, (vec2i) {800, 600});

        divergence->alloc_img(cqueue, zero_data, (vec2i) {800, 600});

        dye[0]->alloc_img(cqueue, dye_concentrates, (vec2i) {800, 600});
        dye[1]->alloc_img(cqueue, dye_concentrates, (vec2i) {800, 600});

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

        cqueue.exec(program, "fluid_boundary", vel_args, {800, 600}, {16, 16});
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

        cqueue.exec(program, "fluid_boundary", vel_args, {800, 600}, {16, 16});
    }

    void advect_quantity(cl::buffer* quantity[2], int& which, cl::program& program, cl::command_queue& cqueue, float timestep_s)
    {
        cl::buffer* q1 = quantity[which];
        cl::buffer* q2 = quantity[(which + 1) % 2];

        cl::buffer* v1 = get_velocity_buf(0);

        cl::args advect_args;

        advect_args.push_back(v1);
        advect_args.push_back(q1);
        advect_args.push_back(q2);
        advect_args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advection", advect_args, {800, 600}, {16, 16});

        which = (which + 1) % 2;
    }

    void apply_force(cl::program& program, cl::command_queue& cqueue, float force, vec2f location, vec2f direction)
    {
        cl::buffer* v1 = get_velocity_buf(0);
        //cl::buffer* v2 = get_velocity_buf(1);

        cl::args force_args;
        force_args.push_back(v1);
        force_args.push_back(v1);
        force_args.push_back(force);
        force_args.push_back(location);
        force_args.push_back(direction);

        cqueue.exec(program, "fluid_apply_force", force_args, {800, 600}, {16, 16});

        //flip_velocity();
        velocity_boundary(program, cqueue);
    }

    void handle_particles(cl::cl_gl_interop_texture* interop, cl::program& program, cl::command_queue& cqueue, float timestep_s)
    {
        interop->acquire(cqueue);

        cl::buffer* v1 = get_velocity_buf(0);

        int num_particles = cpu_particles.size();

        cl::args advect_args;
        advect_args.push_back(v1);
        advect_args.push_back(fluid_particles);
        advect_args.push_back(num_particles);
        advect_args.push_back(timestep_s);

        cqueue.exec(program, "fluid_advect_particles", advect_args, {num_particles}, {128});

        cl::args render_args;
        render_args.push_back(fluid_particles);
        render_args.push_back(num_particles);
        render_args.push_back(interop);

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

        cqueue.exec(program, "fluid_advection", advect_args, {800, 600}, {16, 16});

        flip_velocity();

        velocity_boundary(program, cqueue);

        advect_quantity(dye, which_dye, program, cqueue, timestep_s);

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

            cqueue.exec(program, "fluid_jacobi", diffuse_args, {800, 600}, {16, 16});

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

        cqueue.exec(program, "fluid_divergence", divergence_args, {800, 600}, {16, 16});

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

            cqueue.exec(program, "fluid_jacobi", pressure_args, {800, 600}, {16, 16});

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

        cqueue.exec(program, "fluid_gradient", subtract_args, {800, 600}, {16, 16});

        flip_velocity();

        velocity_boundary(program, cqueue);

        interop->acquire(cqueue);


        //cl::buffer* debug_velocity = get_velocity_buf(0);

        cl::buffer* debug_velocity = dye[which_dye];

        cl::args debug;
        debug.push_back(debug_velocity);
        debug.push_back(interop);

        cqueue.exec(program, "fluid_render", debug, {800, 600}, {16, 16});

        handle_particles(interop, program, cqueue, timestep_s);

    }
};

#endif // FLUID_HPP_INCLUDED
