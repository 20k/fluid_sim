#ifndef FLUID_HPP_INCLUDED
#define FLUID_HPP_INCLUDED

struct fluid_particle
{
    vec2f pos = {0,0};
};

struct physics_particle
{
    vec2f pos = {0,0};
    //vec2f unused_velocity = {0,0};
    //vec4f col = {0, 0, 1, 1};
    uint32_t col = 0;
    uint32_t pad = 0;
};

vec3f hue_from_h(float H)
{
    float R = fabs(H * 6 - 3) - 1;
    float G = 2 - fabs(H * 6 - 2);
    float B = 2 - fabs(H * 6 - 4);
    return clamp((vec3f){R,G,B}, 0, 1);
}

vec3f hsv_to_rgb(vec3f in)
{
    return ((hue_from_h(in.x()) - 1) * in.y() + 1) * in.z();
}

struct fluid_manager
{
    int which_vel = 0;
    int which_pressure = 0;

    cl::buffer* velocity[2];
    cl::buffer* pressure[2];

    cl::buffer* divergence;

    cl::buffer* boundaries;

    cl::buffer* dye[2];

    cl::buffer* fluid_particles;
    std::vector<fluid_particle> cpu_particles;

    ///falling sand
    cl::buffer* physics_particles;
    cl::buffer* physics_tex[2];
    cl::buffer* physics_particles_boundary;
    int which_physics_tex = 0;

    cl::buffer* noise;
    cl::buffer* w_of;
    cl::buffer* upscaled_advected_velocity;

    int which_dye = 0;

    vec2i velocity_dim = {0,0};
    vec2i dye_dim = {0,0};
    vec2i wavelet_dim = {0,0};

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
    void init(cl::context& ctx, cl::buffer_manager& buffers, cl::command_queue& cqueue, vec2i vdim, vec2i ddim, vec2i ndim)
    {
        velocity_dim = vdim;
        dye_dim = ddim;
        wavelet_dim = ndim;

        velocity_to_display_ratio = (vec2f){dye_dim.x(), dye_dim.y()} / (vec2f){velocity_dim.x(), velocity_dim.y()};

        velocity[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        velocity[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        pressure[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        pressure[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        divergence = buffers.fetch<cl::buffer>(ctx, nullptr);

        boundaries = buffers.fetch<cl::buffer>(ctx, nullptr);

        dye[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        dye[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        fluid_particles = buffers.fetch<cl::buffer>(ctx, nullptr);
        physics_particles = buffers.fetch<cl::buffer>(ctx, nullptr);
        physics_particles_boundary = buffers.fetch<cl::buffer>(ctx, nullptr);
        physics_tex[0] = buffers.fetch<cl::buffer>(ctx, nullptr);
        physics_tex[1] = buffers.fetch<cl::buffer>(ctx, nullptr);

        noise = buffers.fetch<cl::buffer>(ctx, nullptr);
        w_of = buffers.fetch<cl::buffer>(ctx, nullptr);
        upscaled_advected_velocity = buffers.fetch<cl::buffer>(ctx, nullptr);

        std::vector<vec4f> zero_data;
        std::vector<vec4f> dye_concentrates;
        std::vector<vec2f> velocity_info;
        std::vector<cl_uchar> boundary_data;

        for(int y=0; y < velocity_dim.y(); y++)
        for(int x=0; x < velocity_dim.x(); x++)
        {
            //vec2f centre = {velocity_dim.x()/2.f, velocity_dim.y()/2.f};

            vec2f fluid_val = randv<2, float>({-0.2f, -0.2f}, {0.2f, 0.2f});

            velocity_info.push_back(fluid_val);

            zero_data.push_back({0,0,0,0});

            if(x == 0 || x == velocity_dim.x() - 1 || y == 0 || y == velocity_dim.y() - 1)
            {
                boundary_data.push_back(1);
            }
            else
            {
                boundary_data.push_back(0);
            }
        }

        for(int y=0; y < dye_dim.y(); y++)
        for(int x=0; x < dye_dim.x(); x++)
        {
            vec2f centre = {dye_dim.x()/2.f, dye_dim.y()/2.f};

            //#define SKY
            #ifdef SKY
            vec3f dye_val = {0.3, 0.3, 0.7 + randf_s(0, 0.3)};
            #else
            vec3f dye_val = {0,0,0};

            dye_val.x() = ((vec2f){x, y} - centre).length() / dye_dim.length();

            dye_val.xy() += fabs(randv<2, float>({-0.2f, -0.2f}, {0.2f, 0.2f}));
            #endif // SKY

            dye_concentrates.push_back({dye_val.x(), dye_val.y(), dye_val.z(), 1.f});
        }

        std::vector<float> noise_data;

        for(int y=0; y < wavelet_dim.y(); y++)
        for(int x=0; x < wavelet_dim.x()*2; x++)
        {
            noise_data.push_back(randf_s(0.f, 1.f));
        }

        velocity[0]->alloc_img(cqueue, velocity_info, velocity_dim, CL_RG, CL_FLOAT);
        velocity[1]->alloc_img(cqueue, velocity_info, velocity_dim, CL_RG, CL_FLOAT);

        pressure[0]->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_FLOAT);
        pressure[1]->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_FLOAT);

        divergence->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_FLOAT);
        boundaries->alloc_img(cqueue, boundary_data, velocity_dim, CL_R, CL_SIGNED_INT8);

        physics_particles_boundary->alloc_img(cqueue, zero_data, velocity_dim, CL_R, CL_FLOAT);

        dye[0]->alloc_img(cqueue, dye_concentrates, dye_dim);
        dye[1]->alloc_img(cqueue, dye_concentrates, dye_dim);

        noise->alloc_img(cqueue, noise_data, velocity_dim, CL_R, CL_FLOAT);
        w_of->alloc_img(cqueue, noise_data, velocity_dim, CL_RG, CL_FLOAT);
        upscaled_advected_velocity->alloc_img(cqueue, noise_data, wavelet_dim, CL_RG, CL_FLOAT);

        cpu_particles.reserve(20000);
        for(int i=0; i < 20000; i++)
        {
            vec2f pos = randv<2, float>({0, 0}, {600, 600});
            //vec2f pos2 = randv<2, float>(600, 1000);

            cpu_particles.push_back({pos});

            //uint32_t col = rgba_to_uint((vec4f){0.3f, 0.3f, 1.f, 1.f});

            //cpu_physics_particles.push_back({pos2, col});
        }

        //uint32_t col_mod = 0;

        std::vector<physics_particle> cpu_physics_particles;

        float hue_start_angle = 0;
        float hue_end_angle = 1;

        for(float y=0; y < dye_dim.y()/2; y+=1.f)
        {
            for(float x=0; x < dye_dim.x(); x+=1.f)
            {
                /*uint32_t col = rgba_to_uint((vec4f){0.3f, 1.f, 0.3f, 1.f});

                if((col_mod % 2) == 0)
                {
                    col = rgba_to_uint((vec4f){0.3f, 0.3f, 1.f, 1.f});
                }*/

                float hangle = mix(hue_start_angle, hue_end_angle, x / dye_dim.x());

                uint32_t col = rgba_to_uint(hsv_to_rgb({hangle, 0.9, 0.9}));

                vec2f pos = {x, y};

                cpu_physics_particles.push_back({pos, col});

                //col_mod++;
            }
        }

        //std::cout << "total num " << col_mod << " ddim " << ddim.x() * ddim.y() << std::endl;

        fluid_particles->alloc(cqueue, cpu_particles);
        physics_particles->alloc(cqueue, cpu_physics_particles);

        //std::cout << "allocated bytes " << physics_particles->alloc_size << " real elements " << cpu_physics_particles.size() << " expected " << cpu_physics_particles.size() * sizeof(physics_particle) << std::endl;


        ///need a double buffer class
        physics_tex[0]->alloc_img(cqueue, zero_data, velocity_dim, CL_RG, CL_FLOAT);
        physics_tex[1]->alloc_img(cqueue, zero_data, velocity_dim, CL_RG, CL_FLOAT);

        cl::args w_of_args;
        w_of_args.push_back(noise);
        w_of_args.push_back(w_of);

        cqueue.exec("wavelet_w_of", w_of_args, velocity_dim, {16, 16});
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

    void velocity_boundary(cl::command_queue& cqueue)
    {
        cl::buffer* v1 = get_velocity_buf(0);

        float scale = -1;

        cl::args vel_args;
        vel_args.push_back(v1);
        vel_args.push_back(v1);
        vel_args.push_back(scale);

        cqueue.exec("fluid_boundary", vel_args, velocity_dim, {16, 16});

        vel_args.push_back(boundaries);
        vel_args.push_back(physics_particles_boundary);

        cqueue.exec("fluid_boundary_tex", vel_args, velocity_dim, {16, 16});
    }

    void pressure_boundary(cl::command_queue& cqueue)
    {
        cl::buffer* v1 = get_pressure_buf(0);
        //cl::buffer* v2 = get_pressure_buf(1);

        float scale = 1;

        cl::args vel_args;
        vel_args.push_back(v1);
        vel_args.push_back(v1);
        vel_args.push_back(scale);

        cqueue.exec("fluid_boundary", vel_args, velocity_dim, {16, 16});

        vel_args.push_back(boundaries);
        vel_args.push_back(physics_particles_boundary);

        cqueue.exec("fluid_boundary_tex", vel_args, velocity_dim, {16, 16});
    }

    void advect_quantity_with(cl::buffer* quantity[2], int& which, cl::command_queue& cqueue, float timestep_s, vec2i dim, cl::buffer* with, bool flip)
    {
        cl::buffer* q1 = quantity[which];
        cl::buffer* q2 = quantity[(which + 1) % 2];

        cl::args advect_args;

        advect_args.push_back(with);
        advect_args.push_back(q1);
        advect_args.push_back(q2);
        advect_args.push_back(timestep_s);

        cqueue.exec("fluid_advection", advect_args, dim, {16, 16});

        if(flip)
            which = (which + 1) % 2;
    }

    void apply_force(cl::command_queue& cqueue, float force, vec2f location, vec2f direction)
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

        cqueue.exec("fluid_apply_force", force_args, velocity_dim, {16, 16});

        //flip_velocity();
        velocity_boundary(cqueue);
    }

    void write_boundary(cl::command_queue& cqueue, vec2f location, float angle)
    {
        location = location / velocity_to_display_ratio;

        location.y() = velocity_dim.y() - location.y();

        cl::args bound_dim;
        bound_dim.push_back(boundaries);
        bound_dim.push_back(location);
        bound_dim.push_back(angle);

        cqueue.exec("fluid_set_boundary", bound_dim, {1}, {1});
    }

    /*void handle_particles(cl::cl_gl_interop_texture* interop, cl::command_queue& cqueue, float timestep_s)
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

        cqueue.exec("fluid_advect_particles", advect_args, {num_particles}, {128});

        cl::args render_args;
        render_args.push_back(fluid_particles);
        render_args.push_back(num_particles);
        render_args.push_back(interop);

        cqueue.exec("fluid_render_particles", render_args, {num_particles}, {128});
    }*/

    uint32_t fsand_id = 0;
    uint32_t phys_counter = 0;

    void handle_falling_sand(cl::command_queue& cqueue, float timestep_s)
    {
        cl::buffer* v1 = get_velocity_buf(0);
        //cl::buffer* v2 = get_velocity_buf(1);

        int num_particles = physics_particles->size() / sizeof(physics_particle);

        vec2f scale = velocity_to_display_ratio;

        cl::buffer* p1 = physics_tex[which_physics_tex];
        cl::buffer* p2 = physics_tex[(which_physics_tex + 1) % 2];

        cl::args physics_args;
        physics_args.push_back(v1);
        physics_args.push_back(physics_particles);
        physics_args.push_back(num_particles);
        physics_args.push_back(timestep_s);
        physics_args.push_back(scale);
        physics_args.push_back(p1);
        physics_args.push_back(p2);
        physics_args.push_back(boundaries);

        cqueue.exec("falling_sand_physics", physics_args, {num_particles}, {128});

        p1->clear_to_zero(cqueue);

        vec2i offset = {0,0};

        if((fsand_id % 2) == 0)
            offset = {1, 1};

        cl::args disimpact_args;
        disimpact_args.push_back(physics_particles);
        disimpact_args.push_back(num_particles);
        disimpact_args.push_back(p2); ///fupped
        disimpact_args.push_back(p1);
        disimpact_args.push_back(boundaries);
        disimpact_args.push_back(offset);

        vec2f upper = ceil((vec2f){velocity_dim.x(), velocity_dim.y()} / (vec2f){2, 2});

        cqueue.exec("falling_sand_disimpact", disimpact_args, (vec2i){upper.x(), upper.y()}, (vec2i){16, 16});

        physics_particles_boundary->clear_to_zero(cqueue);

        #define PARTICLES_INTERFERE_WITH_FLUID
        #ifdef PARTICLES_INTERFERE_WITH_FLUID
        cl::args generate_args;
        generate_args.push_back(p1);
        generate_args.push_back(boundaries);
        generate_args.push_back(physics_particles_boundary);
        generate_args.push_back(scale);
        generate_args.push_back(v1);
        generate_args.push_back(v1);

        cqueue.exec("falling_sand_edge_boundary_condition", generate_args, velocity_dim, {16, 16});
        #endif // PARTICLES_INTERFERE_WITH_FLUID

        /*cl::args render_args;
        render_args.push_back(physics_particles);
        render_args.push_back(num_particles);
        render_args.push_back(interop);

        cqueue.exec("falling_sand_render", render_args, {num_particles}, {128});*/

        //which_physics_tex = (which_physics_tex + 1) % 2;

        p2->clear_to_zero(cqueue);

        fsand_id++;
        phys_counter++;
    }

    void render_sand(cl::cl_gl_interop_texture* interop, cl::command_queue& cqueue)
    {
        interop->acquire(cqueue);

        int num_particles = physics_particles->size() / sizeof(physics_particle);

        cl::args render_args;
        render_args.push_back(physics_particles);
        render_args.push_back(num_particles);
        render_args.push_back(interop);

        cqueue.exec("falling_sand_render", render_args, {num_particles}, {128});
    }

    float timestep_s = 4600.f/1000.f;

    ///future improvement: When decoupling dye/visuals from velocity
    ///keep underlying velocity field at full res, try just performing jacobi at lower res
    ///that way we get full res advection etc, which should maintain most of the quality
    void tick(cl::buffer_manager& buffers, cl::command_queue& cqueue)
    {
        cl::buffer* v1 = get_velocity_buf(0);
        cl::buffer* v2 = get_velocity_buf(1);

        cl::args advect_args;
        advect_args.push_back(v1);
        advect_args.push_back(v1);
        advect_args.push_back(v2);
        advect_args.push_back(timestep_s);

        cqueue.exec("fluid_advection", advect_args, velocity_dim, {16, 16});

        flip_velocity();

        velocity_boundary(cqueue);

        int jacobi_iterations_diff = 10;

        ///ok. Dx seems to = grid scale
        ///if we have a 2x2 deficit in size, we need a 4 grid scale
        float dx = 1;

        for(int i=0; i < jacobi_iterations_diff; i++)
        {
            float viscosity = 0.001f;

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

            cqueue.exec("fluid_jacobi", diffuse_args, velocity_dim, {16, 16});

            flip_velocity();

            //velocity_boundary(program, cqueue);
        }


        velocity_boundary(cqueue);

        ///so. First we caclculate divergence, as bvector
        ///then we calculate x, which is the pressure, which is blank initially
        ///then we jacobi iterate the pressure

        cl::buffer* cv1 = get_velocity_buf(0);

        cl::args divergence_args;
        divergence_args.push_back(cv1);
        divergence_args.push_back(divergence);

        cqueue.exec("fluid_divergence", divergence_args, velocity_dim, {16, 16});

        int pressure_iterations_diff = 40;

        //https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html
        ///thanks berkley!

        for(int i=0; i < pressure_iterations_diff; i++)
        {
            float alpha = -(dx * dx);
            float rbeta = 1/4.f;

            cl::buffer* p1 = get_pressure_buf(0);
            cl::buffer* p2 = get_pressure_buf(1);

            float optimal_w = 1.7;

            int red = 0;

            cl::args pressure_args;
            pressure_args.push_back(p1);
            pressure_args.push_back(divergence);
            pressure_args.push_back(p1);
            pressure_args.push_back(alpha);
            pressure_args.push_back(rbeta);
            pressure_args.push_back(red);
            pressure_args.push_back(optimal_w);

            cqueue.exec("fluid_jacobi_rb", pressure_args, {velocity_dim.x() / 2, velocity_dim.y()}, {16, 16});

            red = 1;

            cl::args pressure_args_red;
            pressure_args_red.push_back(p1);
            pressure_args_red.push_back(divergence);
            pressure_args_red.push_back(p1);
            pressure_args_red.push_back(alpha);
            pressure_args_red.push_back(rbeta);
            pressure_args_red.push_back(red);
            pressure_args_red.push_back(optimal_w);

            cqueue.exec("fluid_jacobi_rb", pressure_args_red, {velocity_dim.x() / 2, velocity_dim.y()}, {16, 16});
        }

        pressure_boundary(cqueue);

        cl::buffer* cpressure = get_pressure_buf(0);
        cl::buffer* cur_v1 = get_velocity_buf(0);
        cl::buffer* cur_v2 = get_velocity_buf(1);

        cl::args subtract_args;
        subtract_args.push_back(cpressure);
        subtract_args.push_back(cur_v1);
        subtract_args.push_back(cur_v2);

        cqueue.exec("fluid_gradient", subtract_args, velocity_dim, {16, 16});

        flip_velocity();

        velocity_boundary(cqueue);


        #ifdef UNSUCCESSFUL_UPSCALE
        cl::buffer* velocity_to_upscale = get_velocity_buf(0);

        cl::args upscale_args;
        upscale_args.push_back(w_of);
        upscale_args.push_back(velocity_to_upscale);
        upscale_args.push_back(upscaled_advected_velocity);
        upscale_args.push_back(timestep_s);

        cqueue.exec("wavelet_upscale", upscale_args, wavelet_dim, {16, 16});
        #endif // UNSUCCESSFUL_UPSCALE

        advect_quantity_with(dye, which_dye, cqueue, timestep_s, dye_dim, get_velocity_buf(0), true);

        #ifdef UNSUCCESSFUL_UPSCALE
        advect_quantity_with(dye, which_dye, program, cqueue, timestep_s, dye_dim, upscaled_advected_velocity, false);


        cl::buffer* ndye = dye[(which_dye + 1) % 2];
        #else
        cl::buffer* ndye = dye[which_dye];
        #endif // UNSUCCESSFUL_UPSCALE

        /*cl::buffer* debug_velocity = ndye;

        cl::args debug;
        debug.push_back(debug_velocity);
        debug.push_back(interop);
        debug.push_back(boundaries);

        cqueue.exec("fluid_render", debug, dye_dim, {16, 16});*/

        //handle_particles(interop, program, cqueue, timestep_s);
        handle_falling_sand(cqueue, timestep_s);
    }

    void render_fluid(cl::cl_gl_interop_texture* interop, cl::command_queue& cqueue)
    {
        interop->acquire(cqueue);

        cl::buffer* debug_velocity = dye[which_dye];

        cl::args debug;
        debug.push_back(debug_velocity);
        debug.push_back(interop);
        debug.push_back(boundaries);

        cqueue.exec("fluid_render", debug, dye_dim, {16, 16});
    }
};

#endif // FLUID_HPP_INCLUDED
