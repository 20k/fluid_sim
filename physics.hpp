#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btBox2dShape.h>
#include <BulletCollision/CollisionShapes/btConvex2dShape.h>
#include <BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h>
#include <BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h>
#include <BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h>

struct physics_body
{
    std::vector<vec2f> vertices;

    btRigidBody* body = nullptr;
    btConvexShape* saved_shape = nullptr;

    void init_sphere(float mass, vec3f start_pos = {0,0,0})
    {
        btConvexShape* shape = new btSphereShape(1);

        init(mass, shape, start_pos);
    }

    void init_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0})
    {
        vec3f half_extents = full_dimensions/2.f;

        btConvexShape* shape = new btBoxShape(btVector3(half_extents.x(), half_extents.y(), half_extents.z()));

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

        //body->setRestitution(1.f);
    }

    void render(sf::RenderWindow& win)
    {
        btTransform trans;
        body->getMotionState()->getWorldTransform(trans);

        btVector3 pos = trans.getOrigin();
        btQuaternion rotation = trans.getRotation();

        sf::CircleShape circle;
        circle.setRadius(10);
        circle.setPosition(pos.getX(), pos.getY());

        win.draw(circle);
    }

    void add(btDynamicsWorld* world)
    {
        world->addRigidBody(body);
    }
};

struct physics_rigidbodies
{
    std::vector<physics_body*> elems;

    physics_body* make_sphere(float mass, vec3f start_pos = {0,0,0})
    {
        physics_body* pbody = new physics_body;

        pbody->init_sphere(mass, start_pos);

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

        physics_body* pb1 = make_sphere(1.f, {0, 50, 0});
        physics_body* pb2 = make_sphere(1.f, {1, 60, 0});

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
