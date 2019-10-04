#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include "fluid.hpp"
#include "lighting.hpp"
#include "physics.hpp"
#include "physics_gpu.hpp"
#include "util.hpp"
#include <windows.h>
#include <wingdi.h>
#include <GL/glew.h>
#include <gl/gl.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <imgui/examples/imgui_impl_glfw.h>
#include <imgui/examples/imgui_impl_opengl3.h>
#include "ui_options.hpp"

extern int b3OpenCLUtils_clewInit();

void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    vec2i window_size = {1500, 1000};

    glfwSetErrorCallback(glfw_error_callback);

    if(!glfwInit())
        throw std::runtime_error("Could not init glfw");

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    glfwWindowHint(GLFW_SAMPLES, 8);

    //glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(window_size.x(), window_size.y(), "Falling Sand Sim", NULL, NULL);

    if (window == NULL)
        throw std::runtime_error("Nullptr window in glfw");

    glfwMakeContextCurrent(window);

    if(glewInit() != GLEW_OK)
        throw std::runtime_error("Bad Glew");

    ImFontAtlas atlas = {};

    ImGui::CreateContext(&atlas);

    printf("ImGui create context\n");

    ImGuiIO& io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    //ImGui::SetStyleLinearColor(true);

    ImGui::PushSrgbStyleColor(ImGuiCol_WindowBg, ImVec4(30/255., 30/255., 30/255., 1));

    ImGuiStyle& style = ImGui::GetStyle();

    style.FrameRounding = 0;
    style.WindowRounding = 0;
    style.ChildRounding = 0;
    style.ChildBorderSize = 0;
    style.FrameBorderSize = 0;
    style.WindowBorderSize = 1;

    if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    io.Fonts->Clear();
    io.Fonts->AddFontDefault();
    ImGuiFreeType::BuildFontAtlas(&atlas, 0, 1);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    printf("Init ogl\n");

    //b3OpenCLUtils_clewInit();

    cl::context ctx;

    cl::program program(ctx, "fluid.cl");
    program.build_with(ctx, "-cl-single-precision-constant -cl-denorms-are-zero -cl-fast-relaxed-math");

    ctx.register_program(program);

    cl::command_queue cqueue(ctx);
    cl::command_queue readback_queue(ctx); ///erm. Sure. Lets pretend nothing can go wrong with this
    cl::command_queue phys_queue(ctx);

    cl::buffer_manager buffer_manage;

    cl::cl_gl_interop_texture* screen_textures[2];
    cl::cl_gl_interop_texture* sand_textures[2];

    for(int i=0; i < 2; i++)
    {
        screen_textures[i] = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr);
        screen_textures[i]->create_rendertexture(window_size.x(), window_size.y());
        screen_textures[i]->acquire(cqueue);

        sand_textures[i] = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr);
        sand_textures[i]->create_rendertexture(window_size.x(), window_size.y());
        sand_textures[i]->acquire(cqueue);
    }

    printf("Init stextures\n");

    int next_screen = 1;
    int current_screen = 0;

    vec2i screen_dim = window_size;

    ui_options options;

    fluid_manager fluid_manage;
    fluid_manage.init(ctx, buffer_manage, cqueue, screen_dim, screen_dim, screen_dim*2);

    printf("Init fluid\n");

    lighting_manager lighting_manage;
    lighting_manage.init(ctx, buffer_manage, cqueue, screen_dim);

    printf("Init lighting\n");

    bool use_cpu_physics = true;

    phys_cpu::physics_rigidbodies physics;

    if(use_cpu_physics)
        physics.init(ctx, buffer_manage);

    phys_gpu::physics_rigidbodies physics_gpu;

    printf("Init phys gpu\n");

    /*if(!use_cpu_physics)
        physics_gpu.init(ctx, phys_queue);*/

    sf::Clock clk;

    vec2f last_mouse = {0,0};
    vec2f cur_mouse = {0,0};

    bool running = true;

    vec2f start_pos = {0,0};
    bool middle_going = false;

    std::vector<vec2f> permanent_forces;

    printf("Pfngl main\n");

    while(running)
    {
        double elapsed_s = clk.restart().asMicroseconds() / 1000. / 1000.;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if(ImGui::IsKeyDown(GLFW_KEY_N))
        {
            std::cout << elapsed_s * 1000. << std::endl;
        }

        int wxpos = 0;
        int wypos = 0;
        glfwGetWindowPos(window, &wxpos, &wypos);

        vec2f screen_absolute_pos = {wxpos, wypos};

        auto mpos = (vec2f){io.MousePos.x, io.MousePos.y} - screen_absolute_pos;
        last_mouse = cur_mouse;
        cur_mouse = mpos;

        vec2f diff = cur_mouse - last_mouse;

        if(ImGui::IsMouseDown(0) && !ImGui::IsAnyWindowFocused())
        {
            if(options.brush == options::FLUID)
            {
                float min_v = 0.05;
                float max_v = 1;

                float frac = (options.brush_size - 1) / (10.f - 1.f);

                float my_v = mix(min_v, max_v, frac);

                fluid_manage.apply_force(cqueue, my_v, cur_mouse, diff);
            }

            ///uuh
            if(options.brush == options::SAND)
            {
                int old_size = fluid_manage.physics_particles->size();
                int old_num = old_size / sizeof(physics_particle);

                std::vector<physics_particle> next;

                for(int y=-options.brush_size + 1; y < options.brush_size; y++)
                {
                    for(int x=-options.brush_size + 1; x < options.brush_size; x++)
                    {
                        physics_particle part;
                        part.pos = {mpos.x() + x, screen_dim.y() - (mpos.y() + y)};
                        part.col = 0xFFFFFFFF;

                        next.push_back(part);
                    }
                }

                fluid_manage.physics_particles->resize(cqueue, old_size + next.size() * sizeof(physics_particle));

                fluid_manage.physics_particles->async_write(cqueue, next, {old_num, 0});
            }

            if(options.brush == options::BOUNDARY)
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
        }

        if(ImGui::IsMouseClicked(0, false) && options.brush == options::RIGID && !ImGui::IsAnyWindowFocused())
        {
            start_pos = cur_mouse;
            middle_going = true;
        }

        if(!ImGui::IsMouseDown(0) && middle_going)
        {
            vec2f end_pos = cur_mouse;

            physics.register_user_physics_body(start_pos, end_pos);

            middle_going = false;
        }

        for(auto& i : permanent_forces)
        {
            fluid_manage.apply_force(cqueue, 0.3, i, {0, -1});
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_V))
        {
            permanent_forces.push_back(cur_mouse);
        }

        if(glfwWindowShouldClose(window))
            running = false;

        if(use_cpu_physics)
            physics.issue_gpu_reads(cqueue, fluid_manage.get_velocity_buf(0), fluid_manage.physics_tex[fluid_manage.which_physics_tex], fluid_manage.velocity_to_display_ratio);

        cl::cl_gl_interop_texture* interop = screen_textures[next_screen];

        interop->acquire(cqueue);
        fluid_manage.tick(buffer_manage, cqueue);
        fluid_manage.render_fluid(interop, cqueue);
        interop->unacquire(cqueue);

        /*if(!use_cpu_physics)
        {
            physics_gpu.render(cqueue, interop, circletex);
            physics_gpu.tick(elapsed_s, fluid_manage.timestep_s, fluid_manage.get_velocity_buf(0), phys_queue);
        }*/

        //lighting_manage.tick(interop, buffer_manage, cqueue, cur_mouse, fluid_manage.dye[fluid_manage.which_dye]);

        if(use_cpu_physics)
        {
            physics.tick(elapsed_s, fluid_manage.timestep_s);
        }

        //if(key.isKeyPressed(sf::Keyboard::Escape))
        //    system("Pause");

        ImDrawList* lst = ImGui::GetBackgroundDrawList();

        options.tick();


        {
            screen_textures[current_screen]->unacquire(cqueue);

            vec2f tl = screen_absolute_pos;
            vec2f br = screen_absolute_pos + (vec2f){screen_textures[current_screen]->w, screen_textures[current_screen]->h};

            lst->AddImage((void*)screen_textures[current_screen]->texture_id, ImVec2(tl.x(),tl.y()), ImVec2(br.x(), br.y()));
        }

        if(use_cpu_physics)
        {
            ///NEEDS UPDATING
            //physics.render(win, fluid_manage.rendered_occlusion[fluid_manage.which_occlusion], cqueue);

            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            for(phys_cpu::physics_body* phys : physics.elems)
            {
                std::vector<vec2f> world = phys->get_world_vertices();
                vec3f col = phys->col*255;

                assert((world.size() % 3) == 0);

                for(int i=0; i < world.size(); i+=3)
                {
                    vec2f v1 = world[i] + screen_absolute_pos;
                    vec2f v2 = world[i+1] + screen_absolute_pos;
                    vec2f v3 = world[i+2] + screen_absolute_pos;

                    lst->AddTriangleFilled({v1.x(), v1.y()}, {v2.x(), v2.y()}, {v3.x(), v3.y()}, IM_COL32(col.x(), col.y(), col.z(), 255));
                }
            }
        }

        {
            sand_textures[next_screen]->acquire(cqueue);

            cl::args debug;
            debug.push_back(sand_textures[next_screen]);

            cqueue.exec("clear_image", debug, {sand_textures[next_screen]->w, sand_textures[next_screen]->h}, {16, 16});

            fluid_manage.render_sand(sand_textures[next_screen], cqueue);
            sand_textures[next_screen]->unacquire(cqueue);

            sand_textures[current_screen]->unacquire(cqueue);

            vec2f tl = screen_absolute_pos;
            vec2f br = screen_absolute_pos + (vec2f){sand_textures[current_screen]->w, sand_textures[current_screen]->h};

            lst->AddImage((void*)sand_textures[current_screen]->texture_id, ImVec2(tl.x(),tl.y()), ImVec2(br.x(), br.y()));
        }

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        //glDrawBuffer(GL_BACK);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, 0);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
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
