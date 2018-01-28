#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include <SFML/Graphics.hpp>
#include "fluid.hpp"

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    sf::RenderWindow win;
    win.create(sf::VideoMode(800, 600), "Test");

    cl::context ctx;

    cl::program program(ctx, "fluid.cl");
    program.build_with(ctx, "");

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);

    cl::buffer_manager buffer_manage;

    cl::buffer* buf = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    std::vector<int> data;

    for(int i=0; i < 800*600; i++)
    {
        data.push_back(i);
    }

    buf->alloc(cqueue, data);

    std::vector<vec4f> idata;

    for(int i=0; i < 800*600; i++)
    {
        idata.push_back({randf_s(-0.01f, 0.01f) + (float)i / (800 * 600), 0, 0, 1});
    }

    cl::buffer* image = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    image->alloc_img(cqueue, idata, (vec2i){800, 600});

    cl::cl_gl_interop_texture* interop = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr, win.getSize().x, win.getSize().y);
    interop->acquire(cqueue);

    cl::args none;
    //none.push_back(buf);
    none.push_back(interop);
    none.push_back(image);

    cqueue.exec(program, "fluid_test", none, {128}, {16});

    cqueue.block();

    fluid_manager fluid_manage;
    fluid_manage.init(ctx, buffer_manage, cqueue);

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
            fluid_manage.apply_force(program, cqueue, 1.f, cur_mouse, diff);
        }

        while(win.pollEvent(event))
        {

        }

        /*cqueue.exec(program, "fluid_test", none, {800, 600}, {16, 16});
        cqueue.block();*/

        fluid_manage.tick(interop, buffer_manage, program, cqueue);

        interop->gl_blit_me(0, cqueue);


        win.display();
        win.clear();

        cqueue.block();
    }

    return 0;
}
