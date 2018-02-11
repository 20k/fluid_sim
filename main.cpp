#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include <SFML/Graphics.hpp>
#include "fluid.hpp"
#include "lighting.hpp"
#include "physics.hpp"
#include "physics_gpu.hpp"
#include "util.hpp"

extern int b3OpenCLUtils_clewInit();

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    vec2i window_size = {1500, 1000};

    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    settings.majorVersion = 4;

    sf::RenderWindow win;
    win.create(sf::VideoMode(window_size.x(), window_size.y()), "Test", sf::Style::Default, settings);

    #if 1

    b3OpenCLUtils_clewInit();

    cl::context ctx;

    cl::program program(ctx, "fluid.cl");
    program.build_with(ctx, "");

    ctx.register_program(program);

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);
    cl::command_queue readback_queue(ctx); ///erm. Sure. Lets pretend nothing can go wrong with this
    cl::command_queue phys_queue(ctx);

    cl::buffer_manager buffer_manage;

    cl::cl_gl_interop_texture* screen_textures[2];

    for(int i=0; i < 2; i++)
    {
        screen_textures[i] = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr);
        screen_textures[i]->create_renderbuffer(win.getSize().x, win.getSize().y);
        screen_textures[i]->acquire(cqueue);
    }

    int next_screen = 1;
    int current_screen = 0;

    vec2i screen_dim = {win.getSize().x, win.getSize().y};

    fluid_manager fluid_manage;
    fluid_manage.init(ctx, buffer_manage, cqueue, screen_dim, screen_dim, screen_dim*2);


    lighting_manager lighting_manage;
    lighting_manage.init(ctx, buffer_manage, cqueue, screen_dim);

    bool use_cpu_physics = true;

    phys_cpu::physics_rigidbodies physics;

    if(use_cpu_physics)
        physics.init(ctx, buffer_manage);

    phys_gpu::physics_rigidbodies physics_gpu;

    if(!use_cpu_physics)
        physics_gpu.init(ctx, phys_queue);

    ///BEGIN HACKY CIRCLE TEXTURE STUFF
    sf::RenderTexture intermediate_tex;
    intermediate_tex.create(10, 10);

    sf::CircleShape shape;
    shape.setRadius(500.f);
    shape.setFillColor(sf::Color(255,255,255,255));

    shape.setPosition(5, 5);
    shape.setOrigin(5, 5);

    intermediate_tex.setActive(true);
    intermediate_tex.draw(shape);
    intermediate_tex.display();

    const sf::Texture& ctex = intermediate_tex.getTexture();
    unsigned int glid = ctex.getNativeHandle();

    cl::cl_gl_interop_texture* circletex = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr);

    circletex->create_from_texture(glid, cl::cl_gl_storage_base());
    circletex->acquire(cqueue);
    #endif // 0
    ///END HACKY CIRCLE TEXTURE STUFF

    sf::Clock clk;
    sf::Keyboard key;

    sf::Mouse mouse;

    vec2f last_mouse = {0,0};
    vec2f cur_mouse = {0,0};

    bool running = true;

    win.resetGLStates();
    PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebufferEXT = (PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebufferEXT");
    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, 0);

    vec2f start_pos = {0,0};
    bool middle_going = false;

    while(running)
    {
        sf::Event event;

        double elapsed_s = clk.restart().asMicroseconds() / 1000. / 1000.;

        if(key.isKeyPressed(sf::Keyboard::N))
        {
            std::cout << elapsed_s * 1000. << std::endl;
        }

        auto mpos = mouse.getPosition(win);
        last_mouse = cur_mouse;
        cur_mouse = {mpos.x, mpos.y};

        vec2f diff = cur_mouse - last_mouse;

        if(mouse.isButtonPressed(sf::Mouse::Left))
        {
            fluid_manage.apply_force(cqueue, 0.1f, cur_mouse, diff);
        }

        if(mouse.isButtonPressed(sf::Mouse::Right))
        {
            vec2f mdiff = (cur_mouse - last_mouse);

            float max_diff = ceil(mdiff.largest_elem());

            if(max_diff == 0)
                fluid_manage.write_boundary(cqueue, cur_mouse, 0.f);
            else
            {
                vec2f step = mdiff / max_diff;
                vec2f start = last_mouse;

                for(int i=0; i < max_diff + 1; i++, start += step)
                {
                    fluid_manage.write_boundary(cqueue, start, 0.f);
                }
            }
        }

        if(ONCE_MACRO(sf::Mouse::Middle))
        {
            start_pos = cur_mouse;
            middle_going = true;
        }

        if(!mouse.isButtonPressed(sf::Mouse::Middle) && middle_going)
        {
            vec2f end_pos = cur_mouse;

            physics.register_user_physics_body(start_pos, end_pos);

            middle_going = false;
        }

        while(win.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                running = false;
        }

        if(use_cpu_physics)
            physics.issue_gpu_reads(cqueue, fluid_manage.get_velocity_buf(0), fluid_manage.physics_tex[fluid_manage.which_physics_tex], fluid_manage.velocity_to_display_ratio);

        cl::cl_gl_interop_texture* interop = screen_textures[next_screen];

        interop->acquire(cqueue);

        fluid_manage.tick(interop, buffer_manage, cqueue);
        fluid_manage.render_fluid(interop, cqueue);

        if(!use_cpu_physics)
        {
            physics_gpu.render(cqueue, interop, circletex);
            physics_gpu.tick(elapsed_s, fluid_manage.timestep_s, fluid_manage.get_velocity_buf(0), phys_queue);
        }

        //lighting_manage.tick(interop, buffer_manage, cqueue, cur_mouse, fluid_manage.dye[fluid_manage.which_dye]);

        fluid_manage.render_sand(interop, cqueue);

        if(use_cpu_physics)
        {
            physics.tick(elapsed_s, fluid_manage.timestep_s);
        }

        if(key.isKeyPressed(sf::Keyboard::Escape))
            system("Pause");

        cl::cl_gl_interop_texture* to_render = screen_textures[current_screen];

        ///render LAST frame
        to_render->gl_blit_me(0, cqueue);
        to_render->acquire(cqueue); ///here for performance, not correctness

        ///need to use last frames occlusion backing
        if(use_cpu_physics)
        {
            physics.render(win, fluid_manage.rendered_occlusion[fluid_manage.which_occlusion], cqueue);
        }

        win.display();
        //win.clear();

        cqueue.block();

        if(use_cpu_physics)
        {
            physics.process_gpu_reads();
        }

        current_screen = (current_screen + 1) % 2;
        next_screen = (next_screen + 1) % 2;
    }

    return 0;
}
