#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <assert.h>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btBox2dShape.h>
#include <BulletCollision/CollisionShapes/btConvex2dShape.h>
#include <BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h>
#include <BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h>
#include <BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h>

struct physics_body
{
    std::vector<vec2f> vertices;
    vec2f local_centre;

    btRigidBody* body = nullptr;
    btConvexShape* saved_shape = nullptr;

    vec2f unprocessed_fluid_velocity = {0,0};

    void calculate_center()
    {
        assert(vertices.size() > 0);

        vec2f centre = {0,0};

        for(vec2f pos : vertices)
        {
            centre += pos;
        }

        local_centre = centre / vertices.size();
    }

    cl::read_event<vec2f> last_read;

    void process_read()
    {
        if(!last_read.bad())
        {
            unprocessed_fluid_velocity += last_read[0];

            last_read.del();
        }
    }

    void issue_read(cl::command_queue& cqueue, cl::buffer* velocity_buffer)
    {
        vec2f pos = get_pos();

        vec2i ipos = {pos.x(), pos.y()};

        last_read = velocity_buffer->async_read<vec2f>(cqueue, ipos);
    }

    std::vector<vec2f> decompose_centrally(const std::vector<vec2f>& vert_in)
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

    void init_sphere(float mass, float rad, vec3f start_pos = {0,0,0})
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

    void init_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0})
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

    void init(float mass, btConvexShape* shape_3d, vec3f start_pos = {0,0,0})
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

        //body->setRestitution(1.f);
    }

    vec2f get_pos()
    {
        btTransform trans;
        body->getMotionState()->getWorldTransform(trans);

        btVector3 pos = trans.getOrigin();
        btQuaternion rotation = trans.getRotation();

        return {pos.getX(), pos.getY()};
    }

    void render(sf::RenderWindow& win)
    {
        btTransform trans;
        body->getMotionState()->getWorldTransform(trans);

        btVector3 pos = trans.getOrigin();
        btQuaternion rotation = trans.getRotation();

        std::vector<sf::Vertex> verts;

        for(int i=0; i < vertices.size(); i++)
        {
            vec2f local_pos = vertices[i];
            vec2f global_pos = local_pos + (vec2f){pos.getX(), pos.getY()};

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

    void add(btDynamicsWorld* world)
    {
        world->addRigidBody(body);
    }
};

struct physics_rigidbodies
{
    std::vector<physics_body*> elems;

    physics_body* make_sphere(float mass, float rad, vec3f start_pos = {0,0,0})
    {
        physics_body* pbody = new physics_body;

        pbody->init_sphere(mass, rad, start_pos);

        elems.push_back(pbody);

        return pbody;
    }

    physics_body* make_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0})
    {
        physics_body* pbody = new physics_body;

        pbody->init_rectangle(mass, full_dimensions, start_pos);

        elems.push_back(pbody);

        return pbody;
    }

    btDiscreteDynamicsWorld* dynamicsWorld;

    void make_2d(btCollisionDispatcher* dispatcher)
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

    void init()
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

        physics_body* pb1 = make_sphere(1.f, 5.f, {50, 50, 0});
        physics_body* pb2 = make_sphere(1.f, 5.f, {51, 60, 0});

        pb1->add(dynamicsWorld);
        pb2->add(dynamicsWorld);
    }

    void tick(double timestep_s)
    {
        dynamicsWorld->stepSimulation(timestep_s, 10);
    }

    void render(sf::RenderWindow& win)
    {
        for(physics_body* pbody : elems)
        {
            /*btTransform trans;
            pbody->body->getMotionState()->getWorldTransform(trans);

            std::cout << "sphere height: " << trans.getOrigin().getY() << std::endl;*/

            pbody->render(win);
        }
    }

    void process_gpu_reads()
    {
        std::vector<cl::event*> events;

        for(physics_body* pbody : elems)
        {
            events.push_back(&pbody->last_read);
        }

        cl::wait_for(events);

        for(physics_body* pbody : elems)
        {
            pbody->process_read();
        }
    }

    void issue_gpu_reads(cl::command_queue& cqueue, cl::buffer* velocity)
    {
        for(physics_body* pbody : elems)
        {
            pbody->issue_read(cqueue, velocity);
        }
    }

    ~physics_rigidbodies()
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
};

#endif // PHYSICS_HPP_INCLUDED
