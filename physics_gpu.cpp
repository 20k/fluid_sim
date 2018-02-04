#include "physics_gpu.hpp"

#include <Bullet3Common/b3Quaternion.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuSapBroadphase.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuGridBroadphase.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvh.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvhBroadphase.h>
#include <Bullet3OpenCL/Initialize/b3OpenCLUtils.h>

#include <Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h>
#include <Bullet3OpenCL/RigidBody/b3GpuRigidBodyPipeline.h>
#include <Bullet3OpenCL/RigidBody/b3GpuNarrowPhase.h>
#include <Bullet3Collision/NarrowPhaseCollision/b3Config.h>

#include <Bullet3Collision/BroadPhaseCollision/b3DynamicBvhBroadphase.h>
#include <Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h>
#include <Bullet3OpenCL/RigidBody/b3GpuNarrowPhaseInternalData.h>

struct phys_gpu::GpuDemoInternalData
{
	cl_platform_id m_platformId;
	cl_context m_clContext;
	cl_device_id m_clDevice;
	cl_command_queue m_clQueue;

	bool m_clInitialized;
	const char*	m_clDeviceName;

	GpuDemoInternalData()
	:m_platformId(0),
	m_clContext(0),
	m_clDevice(0),
	m_clQueue(0),
	m_clInitialized(false),
	m_clDeviceName(0)
	{

	}
};

struct phys_gpu::session_data
{
    cl_kernel m_copyTransformsToVBOKernel;

    b3OpenCLArray<b3Vector4>*	m_instancePosOrnColor;

	class b3GpuRigidBodyPipeline* m_rigidBodyPipeline;

	class b3GpuNarrowPhase* m_np;
	class b3GpuBroadphaseInterface* m_bp;
	class b3DynamicBvhBroadphase* m_broadphaseDbvt;

	b3Vector3 m_pickPivotInA;
	b3Vector3 m_pickPivotInB;
	float m_pickDistance;
	int m_pickBody;
	int	m_pickConstraint;

	int m_altPressed;
	int m_controlPressed;

	int m_pickFixedBody;
	int m_pickGraphicsShapeIndex;
	int m_pickGraphicsShapeInstance;
	b3Config m_config;

	session_data()
	{
	    m_instancePosOrnColor = nullptr;
	    m_rigidBodyPipeline = nullptr;

        m_copyTransformsToVBOKernel = 0;
		m_np = 0;
		m_bp = 0;
		m_broadphaseDbvt = 0;
		m_pickConstraint = -1;
		m_pickFixedBody = -1;
		m_pickGraphicsShapeIndex = -1;
		m_pickGraphicsShapeInstance = -1;
		m_pickBody = -1;
		m_altPressed = 0;
		m_controlPressed = 0;
	}
};


int gGpuArraySizeX = 60;
int gGpuArraySizeY = 60;
int gGpuArraySizeZ = 60;

///position xyz, unused w, normal, uv
static const float cube_vertices[] =
{
	-1.0f, -1.0f, 1.0f, 1.0f,	0,0,1,	0,0,//0
	1.0f, -1.0f, 1.0f, 1.0f,	0,0,1,	1,0,//1
	1.0f,  1.0f, 1.0f, 1.0f,	0,0,1,	1,1,//2
	-1.0f,  1.0f, 1.0f, 1.0f,	0,0,1,	0,1	,//3

	-1.0f, -1.0f, -1.0f, 1.0f,	0,0,-1,	0,0,//4
	1.0f, -1.0f, -1.0f, 1.0f,	0,0,-1,	1,0,//5
	1.0f,  1.0f, -1.0f, 1.0f,	0,0,-1,	1,1,//6
	-1.0f,  1.0f, -1.0f, 1.0f,	0,0,-1,	0,1,//7

	-1.0f, -1.0f, -1.0f, 1.0f,	-1,0,0,	0,0,
	-1.0f, 1.0f, -1.0f, 1.0f,	-1,0,0,	1,0,
	-1.0f,  1.0f, 1.0f, 1.0f,	-1,0,0,	1,1,
	-1.0f,  -1.0f, 1.0f, 1.0f,	-1,0,0,	0,1,

	1.0f, -1.0f, -1.0f, 1.0f,	1,0,0,	0,0,
	1.0f, 1.0f, -1.0f, 1.0f,	1,0,0,	1,0,
	1.0f,  1.0f, 1.0f, 1.0f,	1,0,0,	1,1,
	1.0f,  -1.0f, 1.0f, 1.0f,	1,0,0,	0,1,

	-1.0f, -1.0f,  -1.0f, 1.0f,	0,-1,0,	0,0,
	-1.0f, -1.0f, 1.0f, 1.0f,	0,-1,0,	1,0,
	1.0f, -1.0f,  1.0f, 1.0f,	0,-1,0,	1,1,
	1.0f,-1.0f,  -1.0f,  1.0f,	0,-1,0,	0,1,

	-1.0f, 1.0f,  -1.0f, 1.0f,	0,1,0,	0,0,
	-1.0f, 1.0f, 1.0f, 1.0f,	0,1,0,	1,0,
	1.0f, 1.0f,  1.0f, 1.0f,	0,1,0,	1,1,
	1.0f,1.0f,  -1.0f,  1.0f,	0,1,0,	0,1,
};

void phys_gpu::physics_rigidbodies::make_cube(float mass, vec3f half_extents, vec3f start_pos)
{
    int strideInBytes = 9 * sizeof(float);
    int numVertices = sizeof(cube_vertices) / strideInBytes;

    b3Vector4 scaling = b3MakeVector4(half_extents.x(), half_extents.y(), half_extents.z(), 1);

    int colIndex = m_data->m_np->registerConvexHullShape(cube_vertices, strideInBytes, numVertices, scaling);

    make_obj(mass, colIndex, start_pos);
}

void phys_gpu::physics_rigidbodies::make_sphere(float mass, float radius, vec3f start_pos)
{
    int colIndex = m_data->m_np->registerSphereShape(radius);

    make_obj(mass, colIndex, start_pos);
}

/*void make_plane(float mass, vec3f pos, float plane_constant, vec3f normal, int& index)
{
    int colIndex = m_data->m_np->registerPlaneShape(b3MakeVector3(normal.x(), normal.y(), normal.z()), plane_constant);

    ///0 mass objects ruin performance for some reason
    make_obj(0.f, pos, plane_constant, index, colIndex);
}*/

void phys_gpu::physics_rigidbodies::make_obj(float mass, int colIndex, vec3f start_pos)
{
    b3Vector3 position = b3MakeVector3(start_pos.x(), start_pos.y(), start_pos.z());

    b3Quaternion orn(0,0,0,1);

    int pid = m_data->m_rigidBodyPipeline->registerPhysicsInstance(mass, position, orn, colIndex, -1, false);

    index++;
}

void phys_gpu::physics_rigidbodies::init(cl::context& ctx, cl::command_queue& cqueue, cl::program& prog)
{
    m_clData = new GpuDemoInternalData();

    m_data = new session_data;

    m_clData->m_clContext = ctx.ccontext;
    m_clData->m_platformId = ctx.platform;
    m_clData->m_clDevice = ctx.selected_device;
    m_clData->m_clQueue = cqueue.cqueue;

    m_clData->m_clInitialized = true;
    m_clData->m_clDeviceName = ctx.device_name.c_str();

    int errNum = 0;

    cl::kernel copyTransformsToVBOKernel(prog, "copyTransformsToVBOKernel");

    m_data->m_copyTransformsToVBOKernel = copyTransformsToVBOKernel.ckernel;

    printf("pmax %i\n", m_data->m_config.m_maxConvexBodies);

    m_data->m_config.m_maxConvexBodies = 5000;

    m_data->m_config.m_maxConvexBodies = b3Max(m_data->m_config.m_maxConvexBodies,gGpuArraySizeX*gGpuArraySizeY*gGpuArraySizeZ+10);
    m_data->m_config.m_maxConvexShapes = m_data->m_config.m_maxConvexBodies;

    int maxPairsPerBody = 16;
    m_data->m_config.m_maxBroadphasePairs = maxPairsPerBody*m_data->m_config.m_maxConvexBodies;
    m_data->m_config.m_maxContactCapacity = m_data->m_config.m_maxBroadphasePairs;

    b3GpuNarrowPhase* np = new b3GpuNarrowPhase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue,m_data->m_config);
    b3GpuBroadphaseInterface* bp =0;


    bool useUniformGrid = false;

    if (useUniformGrid)
    {
        bp = new b3GpuGridBroadphase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue);
    } else
    {
        bp = new b3GpuSapBroadphase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue);
    }

    //bp = new b3GpuParallelLinearBvhBroadphase(m_clData->m_clContext, m_clData->m_clDevice, m_clData->m_clQueue);

    m_data->m_np = np;
    m_data->m_bp = bp;
    m_data->m_broadphaseDbvt = new b3DynamicBvhBroadphase(m_data->m_config.m_maxConvexBodies);

    m_data->m_rigidBodyPipeline = new b3GpuRigidBodyPipeline(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue, np, bp,m_data->m_broadphaseDbvt,m_data->m_config);

    b3Vector3 gravity = b3MakeVector3(0, -9.8, 0);

    m_data->m_rigidBodyPipeline->setGravity(gravity);

    m_data->m_rigidBodyPipeline->writeAllInstancesToGpu();
    np->writeAllBodiesToGpu();
    bp->writeAabbsToGpu();

    int index = 0;

    float radius = 1.f;

    /*int colIndex = m_data->m_np->registerSphereShape(radius);

    for(int i=0; i < 5000; i++)
    {
        //make_sphere(1.f, {i * 2 + 400, 600, 0.f}, 21.f, index);

        //make_sphere(1, randv<3, float>(0, 600), 1, index);

        make_obj(1.f, randv<3, float>(0, 600), radius, index, colIndex);
    }*/

    for(int i=0; i < 500; i++)
    {
        //make_sphere(10.f, randf<3, float>(0, 600), radius/2, index);

        //make_cube(10.f, randf<3, float>(0, 600), radius, index);

        vec3f pos = randf<3, float>(0, 900);
        pos.z() = 0;

        make_sphere(10.f, radius, pos);
        //make_cube(1.f, radius, pos);
    }

    //make_plane(0.f, {0, 0, 0}, 1.f, {0, 1, 0}, index);

    ///for some reason, 0 mass objects make everything explode
    ///maybe we're getting actual divide by 0s
    ///anyway at least its known
    for(int x=0; x < 20; x++)
    {
        for(int y=0; y < 20; y++)
        {
            float mult = 20.f;

            //make_cube(0.f, {x*mult, 0, y*mult}, {mult, mult, mult}, index);
        }
    }

    //make_cube(0.f, {4000.f, 1.f, 4000.f}, {0,0,0});

    //make_cube(0.f, {0,0,0}, {4000, 1, 4000}, index);

    m_data->m_rigidBodyPipeline->writeAllInstancesToGpu();
    np->writeAllBodiesToGpu();
    bp->writeAabbsToGpu();
}

void phys_gpu::physics_rigidbodies::tick(double timestep_s, double fluid_timestep_s, cl::buffer* velocity, cl::command_queue& cqueue, cl::program& program)
{
    ///less than 1/10th of a ms
    if(timestep_s < 1/10000.f)
        return;

    int num_objects = m_data->m_rigidBodyPipeline->getNumBodies();

    ///so
    ///as far as i can tell this does not do fixed timestep substepping
    ///this is '''fine''' for the moment, but really i want to implement fixed timesteps
    ///handle substepping manually
    ///and perform interpolation, all on the gpu

    ///TODO: Check that this doesn't stall the pipeline
    m_data->m_rigidBodyPipeline->stepSimulation(timestep_s);

    float timestep = fluid_timestep_s / timestep_s;

    float frame_timestep_s = timestep_s;

    cl_mem buffer = m_data->m_rigidBodyPipeline->getBodyBuffer();

    cl::args args;
    args.push_back(buffer);
    args.push_back(num_objects);
    args.push_back(velocity);
    args.push_back(timestep);
    args.push_back(frame_timestep_s);

    cqueue.exec(program, "keep_upright_and_fluid", args, {num_objects}, {128});
}

void phys_gpu::physics_rigidbodies::render(cl::command_queue& cqueue, cl::program& program, cl::cl_gl_interop_texture* screen_tex, cl::cl_gl_interop_texture* circle_tex)
{
    screen_tex->acquire(cqueue);
    circle_tex->acquire(cqueue);

    //screen_tex->clear_to_zero(cqueue);

    int num_objects = m_data->m_rigidBodyPipeline->getNumBodies();

    if(num_objects)
    {
        cl_mem buffer = m_data->m_rigidBodyPipeline->getBodyBuffer();

        cl::args args;
        args.arg_list.reserve(4);
        args.push_back(circle_tex);
        args.push_back(screen_tex);
        args.push_back(buffer);
        args.push_back(num_objects);

        cqueue.exec(program, "hacky_render", args, {circle_tex->w * circle_tex->h, num_objects}, {16, 16});

        /*npData->m_bodyBufferGPU->copyToHost(*npData->m_bodyBufferCPU);

        sf::CircleShape circle;
        float radius = 5;
        circle.setRadius(radius);
        circle.setOrigin(radius, radius);*/

        /*for(int i=0; i < num_objects; i++)
        {
            b3Vector4 pos = (const b3Vector4&)npData->m_bodyBufferCPU->at(i).m_pos;

            //printf("%f %f %f\n", pos.x, pos.y, pos.z);


            circle.setPosition(sf::Vector2f(pos.x, pos.y));

            win.draw(circle);

            //circle.
        }*/
    }
}
