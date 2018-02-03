#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <btBulletDynamicsCommon.h>

struct physics_body
{
    std::vector<vec2f> vertices;

    btRigidBody* body = nullptr;
    btCollisionShape* saved_shape = nullptr;

    void init_sphere(float mass, vec3f start_pos = {0,0,0})
    {
        btCollisionShape* shape = new btSphereShape(1);

        init(mass, shape, start_pos);
    }

    void init_rectangle(float mass, vec3f full_dimensions, vec3f start_pos = {0,0,0})
    {
        vec3f half_extents = full_dimensions/2.f;

        btCollisionShape* shape = new btBoxShape(btVector3(half_extents.x(), half_extents.y(), half_extents.z()));

        init(mass, shape, start_pos);
    }

    void init(float mass, btCollisionShape* shape, vec3f start_pos = {0,0,0})
    {
        btDefaultMotionState* fallMotionState =
                new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(start_pos.x(), start_pos.y(), start_pos.z())));

        btVector3 fallInertia(0, 0, 0);
        shape->calculateLocalInertia(mass, fallInertia);

        btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI(mass, fallMotionState, shape, fallInertia);

        fallRigidBodyCI.m_restitution = 0.5f;
        //fallRigidBodyCI.m_friction = 1.5f;

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

    void init()
    {
        btBroadphaseInterface* broadphase = new btDbvtBroadphase();

        btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
        btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

        btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

        dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);

        dynamicsWorld->setGravity(btVector3(0, -9.8, 0));
        btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);


        btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
        btRigidBody::btRigidBodyConstructionInfo
                groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));

        groundRigidBodyCI.m_restitution = 0.5f;

        btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
        dynamicsWorld->addRigidBody(groundRigidBody);

        //fall.init_sphere(1.f, {0, 50, 0});

        physics_body* pb1 = make_sphere(1.f, {0, 500, 0});

        dynamicsWorld->addRigidBody(pb1->body);
    }

    void tick(sf::RenderWindow& win)
    {
        dynamicsWorld->stepSimulation(1 / 60.f, 10);

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
