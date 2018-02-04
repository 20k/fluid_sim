#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include <SFML/Graphics.hpp>
#include "fluid.hpp"
#include "lighting.hpp"
#include "physics.hpp"
#include "physics_gpu.hpp"

extern int b3OpenCLUtils_clewInit();

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    vec2i window_size = {1500, 1000};

    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;

    sf::RenderWindow win;
    win.create(sf::VideoMode(window_size.x(), window_size.y()), "Test", sf::Style::Default, settings);

    b3OpenCLUtils_clewInit();

    cl::context ctx;

    cl::program program(ctx, "fluid.cl");
    program.build_with(ctx, "");

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);
    cl::command_queue readback_queue(ctx); ///erm. Sure. Lets pretend nothing can go wrong with this

    cl::buffer_manager buffer_manage;


    cl::cl_gl_interop_texture* interop = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr, win.getSize().x, win.getSize().y);
    interop->acquire(cqueue);


    vec2i screen_dim = {win.getSize().x, win.getSize().y};

    fluid_manager fluid_manage;
    fluid_manage.init(ctx, buffer_manage, program, cqueue, screen_dim, screen_dim, screen_dim*2);


    lighting_manager lighting_manage;
    lighting_manage.init(ctx, buffer_manage, program, cqueue, screen_dim);

    #ifdef PHYSICS_CPU
    phys_cpu::physics_rigidbodies physics;

    physics.init();
    #endif

    phys_gpu::physics_rigidbodies physics_gpu;
    physics_gpu.init(ctx, cqueue, program);

    ///BEGIN HACKY CIRCLE TEXTURE STUFF
    sf::RenderTexture intermediate_tex;
    intermediate_tex.create(10, 10);

    sf::CircleShape shape;
    shape.setRadius(5.f);

    shape.setPosition(5, 5);
    shape.setOrigin(5, 5);

    intermediate_tex.setActive(true);
    intermediate_tex.draw(shape);
    intermediate_tex.display();

    const sf::Texture& ctex = intermediate_tex.getTexture();
    unsigned int glid = ctex.getNativeHandle();

    cl::cl_gl_interop_texture* circletex = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr, (GLuint)glid);
    circletex->acquire(cqueue);
    ///END HACKY CIRCLE TEXTURE STUFF

    sf::Clock clk;
    sf::Keyboard key;

    sf::Mouse mouse;

    vec2f last_mouse = {0,0};
    vec2f cur_mouse = {0,0};

    bool running = true;

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
            fluid_manage.apply_force(program, cqueue, 0.1f, cur_mouse, diff);
        }

        if(mouse.isButtonPressed(sf::Mouse::Right))
        {
            vec2f mdiff = (cur_mouse - last_mouse);

            float max_diff = ceil(mdiff.largest_elem());

            if(max_diff == 0)
                fluid_manage.write_boundary(program, cqueue, cur_mouse, 0.f);
            else
            {
                vec2f step = mdiff / max_diff;
                vec2f start = last_mouse;

                for(int i=0; i < max_diff + 1; i++, start += step)
                {
                    fluid_manage.write_boundary(program, cqueue, start, 0.f);
                }
            }
        }

        while(win.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                running = false;
        }

        /*cqueue.exec(program, "fluid_test", none, {800, 600}, {16, 16});
        cqueue.block();*/

        fluid_manage.tick(interop, buffer_manage, program, cqueue);

        #ifdef PHYSICS_CPU
        physics.issue_gpu_reads(readback_queue, fluid_manage.get_velocity_buf(0));
        #endif

        ///for some reason nothing shows up if we render after ticking
        ///dont understand why
        physics_gpu.render(cqueue, program, interop, circletex);
        physics_gpu.tick(elapsed_s, fluid_manage.timestep_s, fluid_manage.get_velocity_buf(0), cqueue, program);

        //lighting_manage.tick(interop, buffer_manage, program, cqueue, cur_mouse, fluid_manage.dye[fluid_manage.which_dye]);


        interop->gl_blit_me(0, cqueue);

        #ifdef PHYSICS_CPU
        physics.tick(elapsed_s, fluid_manage.timestep_s);
        physics.render(win);
        #endif // PHYSICS_CPU

        if(key.isKeyPressed(sf::Keyboard::Escape))
            system("Pause");


        win.display();
        win.clear();

        ///TODO:
        ///should do one frame ahead shenanigans
        cqueue.block();

        #ifdef PHYSICS_CPU
        physics.process_gpu_reads();
        #endif // PHYSICS_CPU

    }

    return 0;
}
