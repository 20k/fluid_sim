#include "physics.hpp"

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btBox2dShape.h>
#include <BulletCollision/CollisionShapes/btConvex2dShape.h>
#include <BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h>
#include <BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h>
#include <BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h>

#include <SFML/Graphics.hpp>

void phys_cpu::physics_body::calculate_center()
{
    assert(vertices.size() > 0);

    vec2f centre = {0,0};

    for(vec2f pos : vertices)
    {
        centre += pos;
    }

    local_centre = centre / (float)vertices.size();
}

std::vector<vec2f> phys_cpu::physics_body::decompose_centrally(const std::vector<vec2f>& vert_in)
{
    assert(vert_in.size() > 0);

    vec2f centre = {0,0};

    for(vec2f pos : vert_in)
    {
        centre += pos;
    }

    centre = centre / (float)vert_in.size();

    std::vector<vec2f> decomp;

    for(int i=0; i < (int)vert_in.size(); i++)
    {
        int cur = i;
        int next = (i + 1) % vert_in.size();

        vec2f cur_pos = vert_in[cur];
        vec2f next_pos = vert_in[next];

        decomp.push_back(cur_pos);
        decomp.push_back(next_pos);
        decomp.push_back(centre);
    }

    return decomp;
}

void phys_cpu::physics_body::init_sphere(float mass, float rad, vec3f start_pos, float angle)
{
    btSphereShape* shape = new btSphereShape(rad);

    sf::CircleShape cshape(shape->getRadius(), 10);
    int num_points = cshape.getPointCount();

    for(int i=0; i < num_points; i++)
    {
        auto vert = cshape.getPoint(i);

        vertices.push_back({vert.x - rad, vert.y - rad});
    }

    physics_vertices = vertices;

    vertices = decompose_centrally(vertices);

    init(mass, shape, start_pos, angle);
}

void phys_cpu::physics_body::init_rectangle(float mass, vec3f half_extents, vec3f start_pos, float angle)
{
    btBox2dShape* shape = new btBox2dShape(btVector3(half_extents.x(), half_extents.y(), half_extents.z()));

    int num_vertices = shape->getNumVertices();

    vec2f verts[4];
    assert(num_vertices == 4);

    for(int i=0; i < num_vertices; i++)
    {
        btVector3 out;
        shape->getVertex(i, out);
        verts[i] = {out.getX(), out.getY()};
    }

    vertices.push_back(verts[0]);
    vertices.push_back(verts[1]);
    vertices.push_back(verts[3]);
    vertices.push_back(verts[2]);

    physics_vertices = vertices;

    vertices = decompose_centrally(vertices);

    init(mass, shape, start_pos, angle);
}

void phys_cpu::physics_body::init(float mass, btConvexShape* shape_3d, vec3f start_pos, float angle)
{
    quat q;
    q.load_from_axis_angle({0, 0, 1, angle});

    btQuaternion start_rot(q.q.x(), q.q.y(), q.q.z(), q.q.w());


    btDefaultMotionState* fallMotionState =
        new btDefaultMotionState(btTransform(start_rot, btVector3(start_pos.x(), start_pos.y(), start_pos.z())));

    btConvexShape* shape = new btConvex2dShape(shape_3d);

    btVector3 fallInertia(0, 0, 0);
    shape->calculateLocalInertia(mass, fallInertia);

    btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI(mass, fallMotionState, shape, fallInertia);

    fallRigidBodyCI.m_restitution = 0.5f;
    fallRigidBodyCI.m_friction = 0.1f;

    body = new btRigidBody(fallRigidBodyCI);

    saved_shape = shape;

    calculate_center();

    current_mass = mass;

    body->setLinearFactor(btVector3(1, 1, 0));
    body->setAngularFactor(btVector3(0, 0, 1));
    //body->setActivationState(DISABLE_DEACTIVATION);

    //physics_vertices = vertices;

    unprocessed_fluid_vel.resize(physics_vertices.size());
    unprocessed_is_blocked.resize(physics_vertices.size());

    //body->setRestitution(1.f);
}

vec2f phys_cpu::physics_body::get_pos()
{
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);

    btVector3 pos = trans.getOrigin();
    //btQuaternion rotation = trans.getRotation();

    return {pos.getX(), pos.getY()};
}

vec2f phys_cpu::physics_body::get_velocity()
{
    btVector3 velocity = body->getLinearVelocity();

    return {velocity.x(), velocity.y()};
}

void phys_cpu::physics_body::tick(double timestep_s, double fluid_timestep_s)
{
    if(timestep_s < 0.000001)
        return;

    #if 0
    vec2f vel = unprocessed_fluid_velocity;
    vel.y() = -vel.y();

    vec2f target = vel * (float)(fluid_timestep_s / timestep_s);

    //body->applyImpulse(btVector3(to_add.x(), to_add.y(), 0), btVector3(0,0,0));

    vec2f current_velocity = get_velocity();

    ///YEAH THIS ISN'T RIGHT
    vec2f destination_velocity = target;//(target + current_velocity)/2.f;

    vec2f velocity_diff = (destination_velocity - current_velocity) * current_mass;

    body->applyCentralImpulse(btVector3(velocity_diff.x(),velocity_diff.y(), 0));

    unprocessed_fluid_velocity = {0,0};
    #endif // 0

    int num_unprocessed = 0;

    for(int i=0; i < physics_vertices.size(); i++)
    {
        if(unprocessed_is_blocked[i])
            num_unprocessed++;
    }

    //printf("%i num\n", num_unprocessed);

    float fluid_velocity_fraction = 0.001f;

    body->applyCentralForce(btVector3(0, 9.8, 0));

    for(int i=0; i < physics_vertices.size(); i++)
    {
        vec2f vel = unprocessed_fluid_vel[i];
        vel.y() = -vel.y();

        vec2f target = vel * (float)(fluid_timestep_s / timestep_s);

        vec2f local_pos = physics_vertices[i];

        btVector3 bt_local_pos = btVector3(local_pos.x(), local_pos.y(), 0.f);

        btVector3 global_velocity_in_local_point = body->getVelocityInLocalPoint(bt_local_pos);

        ///somethign to do with getvelocityinlocalpoint is completely broken, with the way i understand it
        //btVector3 angular_velocity = body->getAngularVelocity();

        //btVector3 global_velocity_in_local_point = body->getLinearVelocity();
        vec2f global_vel = {global_velocity_in_local_point.getX(), global_velocity_in_local_point.getY()};

        vec2f velocity_diff = (target - global_vel) * current_mass;

        velocity_diff = velocity_diff / (float)physics_vertices.size();

        ///ALERT: TODO: HACK
        ///if there's only one corner in the ground, we process fluid
        ///if there's more than one corner in the ground, negative velocity
        ///can't do anything more intelligent until I have proper occlusion fractions
        if(!unprocessed_is_blocked[i] && num_unprocessed <= 1)
        {
            velocity_diff = velocity_diff * fluid_velocity_fraction;

            body->applyImpulse(btVector3(velocity_diff.x(), velocity_diff.y(), 0), bt_local_pos);
        }
    }

    for(int i=0; i < physics_vertices.size(); i++)
    {
        if(unprocessed_is_blocked[i] && num_unprocessed > 0)
        {
            //float remove_frac = num_unprocessed / 2;

            //if(num_unprocessed > 2)
            //    remove_frac = 1;

            //float remove_frac = 0.01f * num_unprocessed;

            float remove_frac = 0.05f;

            if(num_unprocessed > 1)
                remove_frac = 0.5f;

            if(num_unprocessed > 2)
                remove_frac = 1.f;

            float velocity_remove = 1.f;
            float angular_remove = 1.f;

            vec3f cvel = bt_xyz_to_vec(body->getLinearVelocity());
            vec3f cang = bt_xyz_to_vec(body->getAngularVelocity());

            vec3f to_remove_velocity = -velocity_remove * cvel * remove_frac;
            vec3f to_remove_angular = -angular_remove * cang * remove_frac;


            btMatrix3x3 in_tensor = body->getInvInertiaTensorWorld().inverse();

            to_remove_angular = bt_xyz_to_vec(in_tensor * btVector3(to_remove_angular.x(), to_remove_angular.y(), to_remove_angular.z()));

            body->applyCentralImpulse(btVector3(to_remove_velocity.x(), to_remove_velocity.y(), 0.f));
            body->applyTorqueImpulse(btVector3(to_remove_angular.x(), to_remove_angular.y(), to_remove_angular.z()));

            //auto nvel = body->getLinearVelocity();
            //auto nang = body->getAngularVelocity();
        }

        #ifdef DEBUG_STATE
        if(num_unprocessed == 0)
        {
            col = {1,1,1};
        }

        if(num_unprocessed == 1)
        {
            col = {1, 0, 0};
        }

        if(num_unprocessed == 2)
        {
            col = {0, 0, 1};
        }

        if(num_unprocessed == 3)
        {
            col = {0, 1, 0};
        }

        if(num_unprocessed >= 4)
        {
            col = {1, 0, 1};
        }
        #endif

        unprocessed_fluid_vel[i] = {0,0};
        unprocessed_is_blocked[i] = 0;
    }

    //auto pos = body->getLinearVelocity();
    //auto ang = body->getAngularVelocity();

    //std::cout << pos.getX() << " " << pos.getY() << " ang " << ang.getZ() << std::endl;
}

std::vector<vec2f> phys_cpu::physics_body::get_world_vertices()
{
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);

    btVector3 pos = trans.getOrigin();
    btQuaternion rotation = trans.getRotation();

    quat q = convert_from_bullet_quaternion(rotation);

    std::vector<vec2f> ret;

    for(int i=0; i < (int)vertices.size(); i++)
    {
        vec2f local_pos = vertices[i];

        vec3f transformed_local = rot_quat({local_pos.x(), local_pos.y(), 0.f}, q);
        vec2f global_pos = transformed_local.xy() + (vec2f){pos.getX(), pos.getY()};

        ret.push_back(global_pos);
    }

    return ret;
}

std::vector<vec2f> phys_cpu::physics_body::get_world_physics_vertices()
{
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);

    btVector3 pos = trans.getOrigin();
    btQuaternion rotation = trans.getRotation();

    quat q = convert_from_bullet_quaternion(rotation);

    std::vector<vec2f> ret;

    for(int i=0; i < (int)physics_vertices.size(); i++)
    {
        vec2f local_pos = physics_vertices[i];

        vec3f transformed_local = rot_quat({local_pos.x(), local_pos.y(), 0.f}, q);
        vec2f global_pos = transformed_local.xy() + (vec2f){pos.getX(), pos.getY()};

        ret.push_back(global_pos);
    }

    return ret;
}

void phys_cpu::physics_body::render(std::vector<sf::Vertex>& out)
{
    std::vector<vec2f> world = get_world_vertices();

    for(int i=0; i < (int)world.size(); i++)
    {
        sf::Vertex vert;
        vert.position = sf::Vector2f(world[i].x(), world[i].y());
        vert.color = sf::Color(col.x() * 255, col.y() * 255, col.z() * 255);

        out.push_back(vert);
    }
}

void phys_cpu::physics_body::add(btDynamicsWorld* world)
{
    world->addRigidBody(body);
}

phys_cpu::physics_body* phys_cpu::physics_rigidbodies::make_sphere(float mass, float rad, vec3f start_pos, float angle)
{
    physics_body* pbody = new physics_body(dynamicsWorld);

    pbody->init_sphere(mass, rad, start_pos, angle);

    elems.push_back(pbody);

    return pbody;
}

phys_cpu::physics_body* phys_cpu::physics_rigidbodies::make_rectangle(float mass, vec3f half_extents, vec3f start_pos, float angle)
{
    physics_body* pbody = new physics_body(dynamicsWorld);

    pbody->init_rectangle(mass, half_extents, start_pos, angle);

    elems.push_back(pbody);

    return pbody;
}

void phys_cpu::physics_rigidbodies::register_user_physics_body(vec2f start, vec2f finish)
{
    float length = (finish - start).length();

    if(length < 0.0001f)
        return;

    float width = 5.f;

    vec2f avg = (finish + start)/2.f;

    float angle = (finish - start).angle();

    phys_cpu::physics_body* body = make_rectangle(1.f, {length/2.f, width/2.f, 0.f}, {avg.x(), avg.y(), 0.f}, angle);

    body->add(dynamicsWorld);
}

void phys_cpu::physics_rigidbodies::make_2d(btCollisionDispatcher* dispatcher)
{
    auto pdsolver = new btMinkowskiPenetrationDepthSolver();
    auto simplex = new btVoronoiSimplexSolver();
    auto convexalgo2d = new btConvex2dConvex2dAlgorithm::CreateFunc(simplex, pdsolver);
    auto box2dbox2dalgo = new btBox2dBox2dCollisionAlgorithm::CreateFunc();

    dispatcher->registerCollisionCreateFunc(CONVEX_2D_SHAPE_PROXYTYPE,CONVEX_2D_SHAPE_PROXYTYPE,convexalgo2d);
    dispatcher->registerCollisionCreateFunc(BOX_2D_SHAPE_PROXYTYPE,CONVEX_2D_SHAPE_PROXYTYPE,convexalgo2d);
    dispatcher->registerCollisionCreateFunc(CONVEX_2D_SHAPE_PROXYTYPE,BOX_2D_SHAPE_PROXYTYPE,convexalgo2d);
    dispatcher->registerCollisionCreateFunc(BOX_2D_SHAPE_PROXYTYPE,BOX_2D_SHAPE_PROXYTYPE,convexalgo2d);
}

void phys_cpu::physics_rigidbodies::init(cl::context& ctx, cl::buffer_manager& buffers)
{
    btBroadphaseInterface* broadphase = new btDbvtBroadphase();

    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

    make_2d(dispatcher);

    btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);

    //dynamicsWorld->setGravity(btVector3(0, 9.8, 0));
    dynamicsWorld->setGravity(btVector3(0, 0, 0));

    btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);


    btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
    btRigidBody::btRigidBodyConstructionInfo
    groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));

    groundRigidBodyCI.m_restitution = 0.5f;
    groundRigidBodyCI.m_friction = 0.1f;

    btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    dynamicsWorld->addRigidBody(groundRigidBody);

    //fall.init_sphere(1.f, {0, 50, 0});

    //for(int i=0; i < 509; i++)
    for(int y=0; y < 31; y++)
    for(int x=0; x < 3; x++)
    {
        //physics_body* pb1 = make_sphere(1.f, 5.f, {500 + 5 * x, 50 + y * 5, 0});

        physics_body* pb1 = make_rectangle(1.f, {20, 5, 0}, {500 + 50 * x, 50 + y * 20, 0});

        pb1->add(dynamicsWorld);
    }

    to_read_positions = buffers.fetch<cl::buffer>(ctx, nullptr);
    positions_out = buffers.fetch<cl::buffer>(ctx, nullptr);

    to_read_positions->alloc_bytes(sizeof(vec2f) * max_physics_vertices);
    positions_out->alloc_bytes(sizeof(vec2f) * max_physics_vertices);
    cpu_positions.resize(max_physics_vertices*3);

    cull_shader = new sf::Shader();
    cull_shader->loadFromFile("Shaders/cull.vglsl", "Shaders/cull.fglsl");

    //shader.setUniform("windowHeight", (float)win.getSize().y);

    /*sf::RenderStates state;
    //state.blendMode = sf::BlendAdd;
    state.shader = &shader;*/

    //physics_body* pb2 = make_sphere(1.f, 5.f, {501, 60, 0});

    //pb2->add(dynamicsWorld);
}

void phys_cpu::physics_rigidbodies::tick(double timestep_s, double fluid_timestep_s)
{
    for(physics_body* pbody : elems)
    {
        pbody->tick(timestep_s, fluid_timestep_s);
    }

    dynamicsWorld->stepSimulation(timestep_s, 10, 1/120.f);
}

void phys_cpu::physics_rigidbodies::render(sf::RenderTarget& win, cl::cl_gl_interop_texture* cull_texture, cl::command_queue& cqueue)
{
    cull_texture->unacquire(cqueue);

    sf::Texture* ptr = cull_texture->storage->fetch_storage_as<sf::Texture>();

    cull_shader->setUniform("cull_texture", *ptr);

    sf::RenderStates state;
    state.shader = cull_shader;

    std::vector<sf::Vertex> vertices;

    for(physics_body* pbody : elems)
    {
        pbody->render(vertices);
    }

    if(vertices.size() > 0)
        win.draw(&vertices[0], vertices.size(), sf::Triangles, state);
}

void phys_cpu::physics_rigidbodies::process_gpu_reads()
{
    int exchange = 1;

    if(data_written.compare_exchange_strong(exchange, 0))
    {
        std::lock_guard<std::mutex> guard(data_lock);

        int num = num_written;

        //std::cout << num << std::endl;

        ///we need to pass this out as a parameter
        ///between threads
        //int num_bodies = std::min((int)elems.size(), num);

        int num_bodies = num;

        int current_pbody = 0;
        int current_vert = 0;

        for(int i=0; i < num_bodies; i+=3)
        {
            if(current_pbody >= elems.size())
                continue;

            physics_body* pbody = elems[current_pbody];

            vec2f next_position;
            next_position.x() = cpu_positions[i];
            next_position.y() = cpu_positions[i+1];

            int is_blocked = cpu_positions[i+2];

            if(current_vert >= pbody->unprocessed_fluid_vel.size())
            {
                current_vert = 0;
                current_pbody++;
                i-=3;
                continue;
            }

            pbody->unprocessed_fluid_vel[current_vert] = next_position;
            pbody->unprocessed_is_blocked[current_vert] = is_blocked;

            current_vert++;

            cpu_positions[i] = 0;
            cpu_positions[i+1] = 0;
            cpu_positions[i+2] = 0;
        }
    }
}

struct completion_data
{
    phys_cpu::physics_rigidbodies* bodies = nullptr;
    cl::command_queue* cqueue = nullptr;
    cl::buffer* velocity = nullptr;
    cl::buffer* particle_buffer = nullptr;
    cl::buffer* to_read_positions = nullptr;
    cl::buffer* positions_out = nullptr;
    int num_positions = 0;

    std::vector<vec2f>* to_free = nullptr;
};

struct read_completion_data
{
    phys_cpu::physics_rigidbodies* body = nullptr;
    std::vector<float>* data = nullptr;
};

void on_read_complete(cl_event event, cl_int event_command_exec_status, void* user_data)
{
    read_completion_data* rdata = (read_completion_data*)user_data;

    std::vector<float>* data = rdata->data;

    std::lock_guard<std::mutex> guard(rdata->body->data_lock);

    for(int i=0; i < data->size(); i++)
    {
        rdata->body->cpu_positions[i] = (*data)[i];
    }

    rdata->body->num_written = data->size();
    rdata->body->data_written = 1;

    delete data;
    delete rdata;
}

void on_write_complete(cl_event event, cl_int event_command_exec_status, void* user_data)
{
    completion_data* dat = (completion_data*)user_data;

    cl::args args;
    args.push_back(dat->velocity);
    args.push_back(dat->particle_buffer);
    args.push_back(dat->to_read_positions);
    args.push_back(dat->num_positions);
    args.push_back(dat->positions_out);

    cl::event evt;

    dat->cqueue->exec("fluid_fetch_velocities", args, {dat->num_positions}, {128}, &evt);

    int to_read = dat->num_positions * 3;

    cl::read_event<float> read = dat->positions_out->async_read<float>(*dat->cqueue, {0,0}, to_read, false, {&evt});

    read_completion_data* rdata = new read_completion_data{dat->bodies, read.data};
    read.set_completion_callback(on_read_complete, rdata);

    assert(evt.invalid == false);

    delete dat->to_free;
    delete dat;
}

void phys_cpu::physics_rigidbodies::issue_gpu_reads(cl::command_queue& cqueue, cl::buffer* velocity, cl::buffer* particle_buffer, vec2f velocity_scale)
{
    ///hmm. The problem is, its quite difficult to scatter/gather a series of points as whole objects
    ///when they have different numbers of vertices in those objects
    ///can either use some sort of id based indirection scheme with two buffers, or...
    ///just naively readback the whole set of points. Increases readback bandwidth, BUT at the same time
    ///we may very well need that entire bandwidth anyway, so...
    ///so: per vertex, we need:
    ///fluid velocity (yay differential!), whether or not position is occupied by particle
    ///add/remove of physobjects is going to be a problem, need to do some sort of id cpuside and then use a map
    ///but ignore that for the moment
    std::vector<vec2f> positions;

    for(physics_body* pbody : elems)
    {
        std::vector<vec2f> pos = pbody->get_world_physics_vertices();

        for(vec2f& i : pos)
        {
            i = i / velocity_scale;
            positions.push_back(i);
        }

        //positions.push_back(pbody->get_pos() / velocity_scale);
    }

    //std::cout << "writing " << positions.size() * 2 << std::endl;

    int num_positions = positions.size();

    cl::write_event<vec2f> wrdata = to_read_positions->async_write(cqueue, positions);

    completion_data* dat = new completion_data{this, &cqueue, velocity, particle_buffer, to_read_positions, positions_out, num_positions, wrdata.data};

    #define SUPER_ASYNC
    #ifndef SUPER_ASYNC
    ///this is the conceptual pipeline of what happens on the completion callback chain
    ///however it is quite slow, if sadly drastically simpler
    cl::args args;
    args.push_back(velocity);
    args.push_back(to_read_positions);
    args.push_back(num_positions);
    args.push_back(positions_out);

    cl::event kernel_evt;

    cqueue.exec("fluid_fetch_velocities", args, {num_positions}, {128}, &kernel_evt, {&wrdata});

    cl::read_event<vec2f> read = positions_out->async_read<vec2f>(cqueue, 0, dat->num_positions, false, {&kernel_evt});

    read_completion_data* rdata = new read_completion_data{this, read.data};
    read.set_completion_callback(on_read_complete, rdata);
    #else

    wrdata.set_completion_callback(on_write_complete, dat);

    #endif


    //assert(kernel_evt.invalid == false);

    //std::cout << "queue\n";
}

phys_cpu::physics_rigidbodies::~physics_rigidbodies()
{
    /*dynamicsWorld->removeRigidBody(fallRigidBody);
    delete fallRigidBody->getMotionState();
    delete fallRigidBody;

    dynamicsWorld->removeRigidBody(groundRigidBody);
    delete groundRigidBody->getMotionState();
    delete groundRigidBody;


    delete fallShape;

    delete groundShape;


    delete dynamicsWorld;
    delete solver;
    delete collisionConfiguration;
    delete dispatcher;
    delete broadphase;*/
}
