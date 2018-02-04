#ifndef PHYSICS_GPU_HPP_INCLUDED
#define PHYSICS_GPU_HPP_INCLUDED

#include <assert.h>
#include <vector>
#include <vec/vec.hpp>
#include <ocl/ocl.hpp>

namespace sf
{
    struct RenderWindow;
}

namespace phys_gpu
{
    struct GpuDemoInternalData;
    struct session_data;

    struct physics_rigidbodies
    {
        GpuDemoInternalData* m_clData = nullptr;
        session_data* m_data = nullptr;

        void init(cl::context& ctx, cl::command_queue& cqueue, cl::program& prog);

        void make_sphere(float mass, float radius, vec3f start_pos = {0,0,0});
        void make_cube(float mass, vec3f half_extents, vec3f start_pos = {0,0,0});
        ///void make_plane(float mass, float plane_constant, vec3f normal, vec3f pos);
        void make_obj(float mass, vec3f half_extents, int colIndex, vec3f start_pos);

        void tick(double timestep_s, double fluid_timestep_s, cl::buffer* velocity);
        void render(cl::command_queue& cqueue, cl::program& program, cl::cl_gl_interop_texture* screen_tex, cl::cl_gl_interop_texture* circle_tex); ///gunna have to internal manage textures for opencl stuff
    };
}

#endif // PHYSICS_GPU_HPP_INCLUDED
