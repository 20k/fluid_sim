#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <assert.h>
#include <vector>
#include <vec/vec.hpp>
#include <ocl/ocl.hpp>

namespace sf
{
    struct RenderWindow;
}

struct btDiscreteDynamicsWorld;
struct btCollisionDispatcher;
struct btDynamicsWorld;
struct btRigidBody;
struct btConvexShape;

namespace phys_cpu
{

struct physics_body
{
    std::vector<vec2f> vertices;
    vec2f local_centre;

    float current_mass = 1.f;
    btRigidBody* body = nullptr;
    btConvexShape* saved_shape = nullptr;

    vec2f unprocessed_fluid_velocity = {0,0};

    void calculate_center();

    cl::read_event<vec2f> last_read;

    //std::vector<cl::read_event<vec2f>> unfinished_reads;

    void process_read();
    void issue_read(cl::command_queue& cqueue, cl::buffer* velocity_buffer);

    std::vector<vec2f> decompose_centrally(const std::vector<vec2f>& vert_in);

    void init_sphere(float mass, float rad, vec3f start_pos = {0,0,0});
    void init_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0});
    void init(float mass, btConvexShape* shape_3d, vec3f start_pos = {0,0,0});

    vec2f get_pos();
    vec2f get_velocity();

    void tick(double timestep_s, double fluid_timestep_s);
    void render(sf::RenderWindow& win);

    void add(btDynamicsWorld* world);
};

struct physics_rigidbodies
{
    std::vector<physics_body*> elems;

    void init();

    physics_body* make_sphere(float mass, float rad, vec3f start_pos = {0,0,0});
    physics_body* make_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0});

    btDiscreteDynamicsWorld* dynamicsWorld;

    void make_2d(btCollisionDispatcher* dispatcher);

    void tick(double timestep_s, double fluid_timestep_s);
    void render(sf::RenderWindow& win);

    void process_gpu_reads();
    void issue_gpu_reads(cl::command_queue& cqueue, cl::buffer* velocity);

    ~physics_rigidbodies();
};
}

#endif // PHYSICS_HPP_INCLUDED
