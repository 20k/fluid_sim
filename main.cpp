#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include <SFML/Graphics.hpp>
#include "fluid.hpp"
#include "lighting.hpp"
#include "physics.hpp"

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

    cl::context ctx;

    cl::program program(ctx, "fluid.cl");
    program.build_with(ctx, "");

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);
    cl::command_queue readback_queue(ctx); ///erm. Sure. Lets pretend nothing can go wrong with this

    cl::buffer_manager buffer_manage;

    #if 0
    sf::Clock clk2;

    cl::kernel blank(program, "blank");

    std::vector<int> zero{0};
    cl::buffer* buf = buffer_manage.fetch<cl::buffer>(ctx, nullptr);
    buf->alloc(cqueue, zero);

    cl::args valargs;
    valargs.push_back(buf);

    for(int i=0; i < 22000; i++)
    {
        cqueue.exec(blank, valargs, {1024}, {256});
    }

    cqueue.block();

    std::cout << "TIME " << clk2.getElapsedTime().asMicroseconds() / 1000. / 1000. << std::endl;
    #endif

    /*cl::buffer* buf = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    std::vector<int> data;

    for(int i=0; i < window_size.x() * window_size.y(); i++)
    {
        data.push_back(i);
    }

    buf->alloc(cqueue, data);

    std::vector<vec4f> idata;

    for(int i=0; i < window_size.x() * window_size.y(); i++)
    {
        idata.push_back({randf_s(-0.01f, 0.01f) + (float)i / (window_size.x() * window_size.y()), 0, 0, 1});
    }

    cl::buffer* image = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    image->alloc_img(cqueue, idata, window_size);*/

    cl::cl_gl_interop_texture* interop = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr, win.getSize().x, win.getSize().y);
    interop->acquire(cqueue);

    /*cl::args none;
    //none.push_back(buf);
    none.push_back(interop);
    none.push_back(image);

    cqueue.exec(program, "fluid_test", none, {128}, {16});

    cqueue.block();*/

    vec2i screen_dim = {win.getSize().x, win.getSize().y};

    fluid_manager fluid_manage;
    fluid_manage.init(ctx, buffer_manage, program, cqueue, screen_dim, screen_dim, screen_dim*2);


    lighting_manager lighting_manage;
    lighting_manage.init(ctx, buffer_manage, program, cqueue, screen_dim);

    physics_rigidbodies physics;

    physics.init();

    sf::Clock clk;
    sf::Keyboard key;

    sf::Mouse mouse;

    vec2f last_mouse = {0,0};
    vec2f cur_mouse = {0,0};

    while(win.isOpen())
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
                win.close();
        }

        /*cqueue.exec(program, "fluid_test", none, {800, 600}, {16, 16});
        cqueue.block();*/


        fluid_manage.tick(interop, buffer_manage, program, cqueue);
        physics.issue_gpu_reads(readback_queue, fluid_manage.get_velocity_buf(0));

        //lighting_manage.tick(interop, buffer_manage, program, cqueue, cur_mouse, fluid_manage.dye[fluid_manage.which_dye]);

        interop->gl_blit_me(0, cqueue);

        physics.tick(elapsed_s, fluid_manage.timestep_s);
        physics.render(win);

        if(key.isKeyPressed(sf::Keyboard::Escape))
            system("Pause");


        win.display();
        win.clear();

        ///TODO:
        ///should do one frame ahead shenanigans
        cqueue.block();

        physics.process_gpu_reads();

    }

    return 0;
}
