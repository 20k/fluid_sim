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

    const char* glsl_version = "#version 410";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

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

    ImGui::SetStyleLinearColor(true);

    ImGui::PushSrgbStyleColor(ImGuiCol_WindowBg, ImVec4(30/255., 30/255., 30/255., 255.));

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

    #if 1

    //b3OpenCLUtils_clewInit();

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
        screen_textures[i]->create_renderbuffer(window_size.x(), window_size.y());
        screen_textures[i]->acquire(cqueue);
    }

    printf("Init stextures\n");

    int next_screen = 1;
    int current_screen = 0;

    vec2i screen_dim = window_size;

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

    ///BEGIN HACKY CIRCLE TEXTURE STUFF
    /*sf::RenderTexture intermediate_tex;
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
    circletex->acquire(cqueue);*/
    #endif // 0
    ///END HACKY CIRCLE TEXTURE STUFF

    sf::Clock clk;

    vec2f last_mouse = {0,0};
    vec2f cur_mouse = {0,0};

    bool running = true;

    //win.resetGLStates();
    PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebufferEXT = (PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebufferEXT");
    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, 0);

    vec2f start_pos = {0,0};
    bool middle_going = false;

    std::vector<vec2f> permanent_forces;

    printf("Pfngl main\n");

    while(running)
    {
        sf::Event event;

        double elapsed_s = clk.restart().asMicroseconds() / 1000. / 1000.;

        if(ImGui::IsKeyDown(GLFW_KEY_N))
        {
            std::cout << elapsed_s * 1000. << std::endl;
        }

        //auto mpos = mouse.getPosition(win);

        int wxpos = 0;
        int wypos = 0;
        glfwGetWindowPos(window, &wxpos, &wypos);

        auto mpos = (vec2f){io.MousePos.x, io.MousePos.y} - (vec2f){wxpos, wypos};
        last_mouse = cur_mouse;
        cur_mouse = mpos;

        vec2f diff = cur_mouse - last_mouse;

        if(ImGui::IsMouseDown(0))
        {
            fluid_manage.apply_force(cqueue, 0.3f, cur_mouse, diff);
            //fluid_manage.apply_force(cqueue, 0.3f, cur_mouse + (vec2f){1, 0}, diff);
            //fluid_manage.apply_force(cqueue, 0.3f, cur_mouse + (vec2f){-1, 0}, diff);
            //fluid_manage.apply_force(cqueue, 0.3f, cur_mouse + (vec2f){0, 1}, diff);
            //fluid_manage.apply_force(cqueue, 0.3f, cur_mouse + (vec2f){0, -1}, diff);
        }

        for(auto& i : permanent_forces)
        {
            fluid_manage.apply_force(cqueue, 0.3, i, {0, -1});
        }

        if(ImGui::IsMouseDown(1))
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

        if(ImGui::IsMouseClicked(2, false))
        {
            start_pos = cur_mouse;
            middle_going = true;
        }

        if(!ImGui::IsMouseDown(2) && middle_going)
        {
            vec2f end_pos = cur_mouse;

            physics.register_user_physics_body(start_pos, end_pos);

            middle_going = false;
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_V))
        {
            permanent_forces.push_back(cur_mouse);
        }

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if(glfwWindowShouldClose(window))
            running = false;

        if(use_cpu_physics)
            physics.issue_gpu_reads(cqueue, fluid_manage.get_velocity_buf(0), fluid_manage.physics_tex[fluid_manage.which_physics_tex], fluid_manage.velocity_to_display_ratio);

        cl::cl_gl_interop_texture* interop = screen_textures[next_screen];

        interop->acquire(cqueue);

        fluid_manage.tick(interop, buffer_manage, cqueue);
        fluid_manage.render_fluid(interop, cqueue);

        /*if(!use_cpu_physics)
        {
            physics_gpu.render(cqueue, interop, circletex);
            physics_gpu.tick(elapsed_s, fluid_manage.timestep_s, fluid_manage.get_velocity_buf(0), phys_queue);
        }*/

        //lighting_manage.tick(interop, buffer_manage, cqueue, cur_mouse, fluid_manage.dye[fluid_manage.which_dye]);

        fluid_manage.render_sand(interop, cqueue);

        if(use_cpu_physics)
        {
            physics.tick(elapsed_s, fluid_manage.timestep_s);
        }

        //if(key.isKeyPressed(sf::Keyboard::Escape))
        //    system("Pause");

        cl::cl_gl_interop_texture* to_render = screen_textures[current_screen];

        ///render LAST frame
        to_render->gl_blit_me(0, cqueue);
        to_render->acquire(cqueue); ///here for performance, not correctness

        ///need to use last frames occlusion backing
        if(use_cpu_physics)
        {
            ///NEEDS UPDATING
            //physics.render(win, fluid_manage.rendered_occlusion[fluid_manage.which_occlusion], cqueue);
        }

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);
        //glClearColor(0, 0, 0, 0);
        //glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);

        //win.display();
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
