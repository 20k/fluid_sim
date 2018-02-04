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

    local_centre = centre / vertices.size();
}

//std::vector<cl::read_event<vec2f>> unfinished_reads;

void  phys_cpu::physics_body::process_read()
{
    if(!last_read.bad())
    {
        unprocessed_fluid_velocity = last_read[0];

        unprocessed_fluid_velocity.y() = -unprocessed_fluid_velocity.y();

        last_read.del();
    }

    /*for(int i=0; i < unfinished_reads.size(); i++)
    {
        cl::read_event<vec2f> cur = unfinished_reads[i];

        if(cur.finished())
        {
            for(int kk=0; kk < i + 1; kk++)
            {
                unfinished_reads.erase(unfinished_reads.begin());
            }

            unprocessed_fluid_velocity = cur[0];
            return;
        }
    }*/
}

void phys_cpu::physics_body::issue_read(cl::command_queue& cqueue, cl::buffer* velocity_buffer)
{
    vec2f pos = get_pos();

    vec2i ipos = {pos.x(), pos.y()};

    last_read = velocity_buffer->async_read<vec2f>(cqueue, ipos, {1, 1}, true);

    /*if(!last_read.bad())
    {
        unfinished_reads.push_back(last_read);
    }*/
}

std::vector<vec2f> phys_cpu::physics_body::decompose_centrally(const std::vector<vec2f>& vert_in)
{
    assert(vert_in.size() > 0);

    vec2f centre = {0,0};

    for(vec2f pos : vert_in)
    {
        centre += pos;
    }

    centre = centre / vert_in.size();

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

void phys_cpu::physics_body::init_sphere(float mass, float rad, vec3f start_pos)
{
    btSphereShape* shape = new btSphereShape(rad);

    sf::CircleShape cshape(shape->getRadius());
    int num_points = cshape.getPointCount();

    for(int i=0; i < num_points; i++)
    {
        auto vert = cshape.getPoint(i);

        vertices.push_back({vert.x - rad, vert.y - rad});
    }

    vertices = decompose_centrally(vertices);

    init(mass, shape, start_pos);
}

void phys_cpu::physics_body::init_rectangle(float mass, vec3f full_dimensions, vec3f start_pos)
{
    vec3f half_extents = full_dimensions/2.f;

    btBox2dShape* shape = new btBox2dShape(btVector3(half_extents.x(), half_extents.y(), half_extents.z()));

    int num_vertices = shape->getNumVertices();

    for(int i=0; i < num_vertices; i++)
    {
        btVector3 out;
        shape->getVertex(i, out);

        vertices.push_back({out.getX(), out.getY()});
    }

    vertices = decompose_centrally(vertices);

    init(mass, shape, start_pos);
}

void phys_cpu::physics_body::init(float mass, btConvexShape* shape_3d, vec3f start_pos)
{
    btDefaultMotionState* fallMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(start_pos.x(), start_pos.y(), start_pos.z())));

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

    //body->setRestitution(1.f);
}

vec2f phys_cpu::physics_body::get_pos()
{
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);

    btVector3 pos = trans.getOrigin();
    btQuaternion rotation = trans.getRotation();

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

    vec2f target = unprocessed_fluid_velocity * fluid_timestep_s / timestep_s;

    //body->applyImpulse(btVector3(to_add.x(), to_add.y(), 0), btVector3(0,0,0));

    vec2f current_velocity = get_velocity();

    ///YEAH THIS ISN'T RIGHT
    vec2f destination_velocity = target;//(target + current_velocity)/2.f;

    vec2f velocity_diff = (destination_velocity - current_velocity) * current_mass;

    body->applyCentralImpulse(btVector3(velocity_diff.x(),velocity_diff.y(), 0));

    //printf("%f %f\n", velocity.x(), velocity.y());

    unprocessed_fluid_velocity = {0,0};
}

void phys_cpu::physics_body::render(sf::RenderWindow& win)
{
    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);

    btVector3 pos = trans.getOrigin();
    btQuaternion rotation = trans.getRotation();

    std::vector<sf::Vertex> verts;

    for(int i=0; i < vertices.size(); i++)
    {
        vec2f local_pos = vertices[i];
        vec2f global_pos = local_pos + (vec2f) {pos.getX(), pos.getY()};

        sf::Vertex vert;
        vert.position = sf::Vector2f(global_pos.x(), global_pos.y());

        verts.push_back(vert);
    }

    /*sf::CircleShape circle;
    circle.setRadius(10);
    circle.setPosition(pos.getX(), pos.getY());*/

    //win.draw(circle);

    win.draw(&verts[0], verts.size(), sf::Triangles);
}

void phys_cpu::physics_body::add(btDynamicsWorld* world)
{
    world->addRigidBody(body);
}

phys_cpu::physics_body* phys_cpu::physics_rigidbodies::make_sphere(float mass, float rad, vec3f start_pos)
{
    physics_body* pbody = new physics_body;

    pbody->init_sphere(mass, rad, start_pos);

    elems.push_back(pbody);

    return pbody;
}

phys_cpu::physics_body* phys_cpu::physics_rigidbodies::make_rectangle(float mass, vec3f full_dimensions, vec3f start_pos)
{
    physics_body* pbody = new physics_body;

    pbody->init_rectangle(mass, full_dimensions, start_pos);

    elems.push_back(pbody);

    return pbody;
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

void phys_cpu::physics_rigidbodies::init()
{
    btBroadphaseInterface* broadphase = new btDbvtBroadphase();

    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

    make_2d(dispatcher);

    btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);

    dynamicsWorld->setGravity(btVector3(0, -9.8, 0));
    btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);


    btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
    btRigidBody::btRigidBodyConstructionInfo
    groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));

    groundRigidBodyCI.m_restitution = 0.5f;
    groundRigidBodyCI.m_friction = 0.1f;

    btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    dynamicsWorld->addRigidBody(groundRigidBody);

    //fall.init_sphere(1.f, {0, 50, 0});

    for(int i=0; i < 100; i++)
    {
        physics_body* pb1 = make_sphere(1.f, 5.f, {500 + 5 * i, 50, 0});

        pb1->add(dynamicsWorld);
    }
    //physics_body* pb2 = make_sphere(1.f, 5.f, {501, 60, 0});

    //pb2->add(dynamicsWorld);
}

void phys_cpu::physics_rigidbodies::tick(double timestep_s, double fluid_timestep_s)
{
    for(physics_body* pbody : elems)
    {
        pbody->tick(timestep_s, fluid_timestep_s);
    }

    dynamicsWorld->stepSimulation(timestep_s, 10);
}

void phys_cpu::physics_rigidbodies::render(sf::RenderWindow& win)
{
    for(physics_body* pbody : elems)
    {
        /*btTransform trans;
        pbody->body->getMotionState()->getWorldTransform(trans);

        std::cout << "sphere height: " << trans.getOrigin().getY() << std::endl;*/

        pbody->render(win);
    }
}

void phys_cpu::physics_rigidbodies::process_gpu_reads()
{
    std::vector<cl::event*> events;

    for(physics_body* pbody : elems)
    {
        /*if(pbody->unfinished_reads.size() > 2)
        {
            events.push_back(&pbody->unfinished_reads[0]);
        }*/
        events.push_back(&pbody->last_read);
    }

    cl::wait_for(events);

    for(physics_body* pbody : elems)
    {
        pbody->process_read();
    }
}

void phys_cpu::physics_rigidbodies::issue_gpu_reads(cl::command_queue& cqueue, cl::buffer* velocity)
{
    for(physics_body* pbody : elems)
    {
        pbody->issue_read(cqueue, velocity);
    }

    clFlush(cqueue);
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
