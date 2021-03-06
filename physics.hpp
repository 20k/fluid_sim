#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <assert.h>
#include <vector>
#include <vec/vec.hpp>
#include <ocl/ocl.hpp>
#include <mutex>
#include <atomic>

namespace sf
{
    struct RenderWindow;
    struct RenderTarget;
    struct Shader;
    struct RenderStates;
    struct Texture;
    struct Vertex;
}

struct btDiscreteDynamicsWorld;
struct btCollisionDispatcher;
struct btDynamicsWorld;
struct btRigidBody;
struct btConvexShape;

namespace phys_cpu
{

///TODO: Decouple physics sampling points from
///rendering points
///so we can use circles without killing the framerate
struct physics_body
{
    std::vector<vec2f> vertices;
    std::vector<vec2f> physics_vertices;
    vec2f local_centre;
    vec3f col = {1,1,1};

    float current_mass = 1.f;
    btRigidBody* body = nullptr;
    btConvexShape* saved_shape = nullptr;

    //vec2f unprocessed_fluid_velocity = {0,0};

    std::vector<vec2f> unprocessed_fluid_vel;
    std::vector<int> unprocessed_is_blocked;

    btDiscreteDynamicsWorld* world = nullptr;

    physics_body(btDiscreteDynamicsWorld* world) : world(world){}

    void calculate_center();

    std::vector<vec2f> decompose_centrally(std::vector<vec2f> vert_in);
    std::vector<vec2f> get_world_vertices();
    std::vector<vec2f> get_world_physics_vertices();

    void init_sphere(float mass, float rad, vec3f start_pos = {0,0,0}, float angle = 0.f);
    void init_rectangle(float mass, vec3f half_extents, vec3f start_pos = {0,0,0}, float angle = 0.f);
    void init(float mass, btConvexShape* shape_3d, vec3f start_pos = {0,0,0}, float angle = 0.f);

    vec2f get_pos();
    vec2f get_velocity();

    void tick(double timestep_s, double fluid_timestep_s);
    //void render(std::vector<sf::Vertex>& out);

    void add(btDynamicsWorld* world);
};

struct physics_rigidbodies
{
    std::vector<physics_body*> elems;

    std::vector<float> cpu_positions;
    cl::buffer* to_read_positions = nullptr;
    cl::buffer* positions_out = nullptr;

    std::atomic_int data_written;

    sf::Shader* cull_shader;

    //volatile int data_written = 0;
    volatile int num_written = 0;
    std::mutex data_lock;

    int max_physics_vertices = 100000;

    void init(cl::context& ctx, cl::buffer_manager& buffers);

    physics_body* make_sphere(float mass, float rad, vec3f start_pos = {0,0,0}, float angle = 0.f);
    physics_body* make_rectangle(float mass, vec3f half_extents, vec3f start_pos = {0,0,0}, float angle = 0.f);

    void register_user_physics_body(vec2f start, vec2f finish);

    btDiscreteDynamicsWorld* dynamicsWorld;

    void make_2d(btCollisionDispatcher* dispatcher);

    void tick(double timestep_s, double fluid_timestep_s);
    //void render(sf::RenderTarget& win, cl::cl_gl_interop_texture* cull_texture, cl::command_queue& cqueue);

    void process_gpu_reads();
    void issue_gpu_reads(cl::command_queue& cqueue, cl::buffer* velocity, cl::buffer* particle_buffer, vec2f velocity_scale);

    ~physics_rigidbodies();
};
}

#endif // PHYSICS_HPP_INCLUDED
