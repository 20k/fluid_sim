#pragma OPENCL EXTENSION cl_khr_gl_event : enable

__kernel
void fluid_test(__write_only image2d_t screen, __read_only image2d_t test)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    int gw = get_image_width(screen);
    int gh = get_image_height(screen);

    if(ix >= gw || iy >= gh)
        return;

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float4 val = read_imagef(test, sam, (int2){ix, iy});

    //printf("%f %f %f %f\n", val.x, val.y, val.z, val.w);

    write_imagef(screen, (int2){ix, iy}, val);
}

#define GRID_SCALE 1

///advection is the only time we need to deal with mixed resolution quantities
///advection is not a bottleneck, therefore a slow solution is fine
__kernel
void fluid_advection(__read_only image2d_t velocity, __read_only image2d_t advect_quantity_in, __write_only image2d_t advect_quantity_out, float timestep)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    float gw = get_image_width(advect_quantity_in);
    float gh = get_image_height(advect_quantity_in);

    if(pos.x >= gw || pos.y >= gh)
        return;

    float vw = get_image_width(velocity);
    float vh = get_image_height(velocity);

    float2 vdim = (float2){vw, vh};
    float2 adim = (float2){gw, gh};

    pos += 0.5f;

    float rdx = 1.f / GRID_SCALE;

    float2 new_pos = pos - timestep * rdx * read_imagef(velocity, sam, pos * vdim / adim).xy;

    float4 new_value = read_imagef(advect_quantity_in, sam, new_pos);

    write_imagef(advect_quantity_out, convert_int2(pos), new_value);
}

__kernel
void fluid_jacobi(__read_only image2d_t xvector, __read_only image2d_t bvector, __write_only image2d_t out, float alpha, float rbeta)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(xvector);
    int gh = get_image_height(xvector);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float4 xL = read_imagef(xvector, sam, pos - (float2){1, 0});
    float4 xR = read_imagef(xvector, sam, pos + (float2){1, 0});
    float4 xB = read_imagef(xvector, sam, pos - (float2){0, 1});
    float4 xT = read_imagef(xvector, sam, pos + (float2){0, 1});

    float4 bC = read_imagef(bvector, sam, pos);

    float4 xnew = (xL + xR + xB + xT + alpha * bC) * rbeta;

    write_imagef(out, convert_int2(pos), xnew);
}

__kernel
void fluid_divergence(__read_only image2d_t vector_field_in, __write_only image2d_t out)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(vector_field_in);
    int gh = get_image_height(vector_field_in);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float half_rdx = 0.5f / GRID_SCALE;

    float4 wL = read_imagef(vector_field_in, sam, pos - (float2){1, 0});
    float4 wR = read_imagef(vector_field_in, sam, pos + (float2){1, 0});
    float4 wB = read_imagef(vector_field_in, sam, pos - (float2){0, 1});
    float4 wT = read_imagef(vector_field_in, sam, pos + (float2){0, 1});

    float div = half_rdx * ((wR.x - wL.x) + (wT.y - wB.y));

    write_imagef(out, convert_int2(pos), div);
}

__kernel
void fluid_gradient(__read_only image2d_t pressure_field, __read_only image2d_t velocity_field, __write_only image2d_t velocity_out)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(pressure_field);
    int gh = get_image_height(pressure_field);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float half_rdx = 0.5f / GRID_SCALE;

    float pL = read_imagef(pressure_field, sam, pos - (float2){1, 0}).x;
    float pR = read_imagef(pressure_field, sam, pos + (float2){1, 0}).x;
    float pB = read_imagef(pressure_field, sam, pos - (float2){0, 1}).x;
    float pT = read_imagef(pressure_field, sam, pos + (float2){0, 1}).x;

    float4 new_velocity = read_imagef(velocity_field, sam, pos);

    new_velocity.xy -= half_rdx * (float2){pR - pL, pT - pB};

    write_imagef(velocity_out, convert_int2(pos), new_velocity);
}

__kernel
void fluid_render(__read_only image2d_t field, __write_only image2d_t screen, __read_only image2d_t boundaries)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_NONE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(screen);
    int gh = get_image_height(screen);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float4 val = read_imagef(field, sam, pos);

    val = fabs(val);

    int2 bound = read_imagei(boundaries, sam, pos).xy;

    if(bound.x == 1)
        val.xyz = 1;

    val = clamp(val, 0.f, 1.f);

    write_imagef(screen, convert_int2(pos), (float4)(val.xyz, 1.f));
}

__kernel
void fluid_boundary(__read_only image2d_t field_in, __write_only image2d_t field_out, float scale)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_NONE |
                    CLK_FILTER_NEAREST;

    int2 ipos = (int2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(field_in);
    int gh = get_image_height(field_in);

    if(ipos.x >= gw || ipos.y >= gh)
        return;

    float2 offset = {0,0};

    if(ipos.x == 0)
        offset.x = 1;
    if(ipos.y == 0)
        offset.y = 1;

    if(ipos.x == gw - 1)
        offset.x = -1;
    if(ipos.y == gh - 1)
        offset.y = -1;

    if(ipos.x == 0 || ipos.x == gw - 1 || ipos.y == 0 || ipos.y == gh - 1)
    {
        float2 pos = convert_float2(ipos) + 0.5f;

        float4 real_val = read_imagef(field_in, sam, pos + offset);

        real_val = real_val * scale;

        write_imagef(field_out, convert_int2(pos), real_val);
    }
}

float2 angle_to_offset(float angle)
{
    float2 normal = {cos(angle), sin(angle)};

    ///round off any error
    normal = round(normal * 100.f) / 100.f;

    float2 res = {0,0};

    if(normal.x > 0)
        res.x = 1;
    if(normal.x < 0)
        res.x = -1;

    if(normal.y > 0)
        res.y = 1;
    if(normal.y < 0)
        res.y = -1;

    return res;
}

///this method assumes that each fluid boundary pixel is connected to two others
///and then creates boundaries on both side of the line
///its not 1000% perfect but it works better than i expected
///need to do a pixelwise edge detect for particle blocks, aka sobel or something
///then pass the result into here

float get_boundary_strength(float2 pos, __read_only image2d_t boundary_texture, __read_only image2d_t particle_boundary_strength)
{
    sampler_t sam_near = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    int val = read_imagei(boundary_texture, sam_near, convert_float2(pos) + 0.5f).x;

    if(val == 1)
        return 1.f;

    return read_imagef(particle_boundary_strength, sam_near, convert_float2(pos) + 0.5f).x;
}

__kernel
void fluid_boundary_tex(__read_only image2d_t field_in, __write_only image2d_t field_out, float scale, __read_only image2d_t boundary_texture,
                        __read_only image2d_t particle_boundary_strength)
{
    int2 ipos = (int2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(field_in);
    int gh = get_image_height(field_in);

    if(ipos.x >= gw || ipos.y >= gh)
        return;

    float2 sdim = (float2){gw, gh};

    float2 pos = convert_float2(ipos);

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    /*int2 vals = read_imagei(boundary_texture, sam_near, convert_float2(ipos) + 0.5f).xy;

    if(vals.x != 1)
        return;*/

    float base_strength = get_boundary_strength(pos, boundary_texture, particle_boundary_strength);

    if(base_strength <= 0)
        return;

    ///should work this out automatically in the future
    ///bit of a ballache to handle the edge cases so for the moment this is explicit and
    ///buyer beware
    /*float angle = vals.y;

    float2 normal = {cos(angle), sin(angle)};

    float4 real_val = read_imagef(field_in, sam, convert_float2(ipos) + normal + 0.5f);

    real_val = real_val * scale;

    write_imagef(field_out, ipos, real_val);*/

    ///ok. Wheel one way until we find a boundary
    ///wheel the same way looking for another boundary
    ///if we find another boundary, find normal and offset position by normal, then offset again and write pressure
    ///we do this for both directions of normal so that a straight line is a boundary on both sides
    ///then update initial guess
    ///if we are in a block who cares
    float tl = 0;

    float angles = 8;

    int range_start = -999;

    float current_strength = 0.f;

    for(int i=0; i < angles; i++)
    {
        float angle_frac = 2 * M_PI * (float)i / angles;

        float2 offset = angle_to_offset(angle_frac);

        if(any(pos + offset < 0) || any(pos + offset >= sdim))
            continue;

        /*int2 nval = read_imagei(boundary_texture, sam_near, pos + 0.5f + offset).xy;

        if(nval.x == 1)
        {
            range_start = i;
        }*/

        float strength = get_boundary_strength(pos + offset, boundary_texture, particle_boundary_strength);

        if(strength > 0)
        {
            current_strength = strength;

            range_start = i;
        }
    }

    if(range_start < 0)
        return;

    float fnormalangle = 0;

    for(int i=range_start + 1; i < range_start + angles; i++)
    {
        int id = i % (int)angles;

        float angle_frac = 2 * M_PI * (float)id / angles;

        float2 offset = angle_to_offset(angle_frac);

        if(any(pos + offset < 0) || any(pos + offset >= sdim))
            continue;

        float strength = get_boundary_strength(pos + offset, boundary_texture, particle_boundary_strength);

        if(strength > 0)
        {
            current_strength = (current_strength + strength)/2.f;

            fnormalangle = (angle_frac + 2 * M_PI * (float)range_start / angles) / 2.f;
            break;
        }
    }

    float2 fnormal = {cos(fnormalangle), sin(fnormalangle)};

    int2 p1 = convert_int2(pos + fnormal);
    int2 p2 = convert_int2(pos - fnormal);

    float4 base1 = read_imagef(field_in, sam, p1);
    float4 base2 = read_imagef(field_in, sam, p2);

    float4 rv1 = read_imagef(field_in, sam, pos + fnormal * 2 + 0.5f);
    float4 rv2 = read_imagef(field_in, sam, pos - fnormal * 2 + 0.5f);

    rv1 = rv1 * scale;
    rv2 = rv2 * scale;

    rv1 = mix(base1, rv1, current_strength);
    rv2 = mix(base2, rv2, current_strength);

    write_imagef(field_out, p1, rv1);
    write_imagef(field_out, p2, rv2);
}

__kernel
void fluid_set_boundary(__write_only image2d_t buffer, float2 pos, float angle)
{
    int gid = get_global_id(0);

    ///yup
    if(gid >= 1)
        return;

    int gw = get_image_width(buffer);
    int gh = get_image_height(buffer);

    if(any(pos < 0) || any(pos >= (float2){gw, gh}))
       return;

    //write_imagef(buffer, convert_int2(pos), (float4)(1, angle, 0, 0));

    write_imagei(buffer, convert_int2(pos), 1);
}


__kernel
void fluid_apply_force(__read_only image2d_t velocity_in, __write_only image2d_t velocity_out, float force, float2 position, float2 direction)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_NONE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(velocity_in);
    int gh = get_image_height(velocity_in);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float max_len = 10;

    //position.y = gh - position.y;

    if(fast_length(pos - position) > max_len)
        return;

    direction = fast_normalize(direction);

    //direction.y = -direction.y;

    float flen = 1.f - fast_length(pos - position) / max_len;

    float2 extra = force * direction * flen;

    float2 old_vel = read_imagef(velocity_in, sam, pos).xy;

    float2 sum = old_vel + extra.xy;

    write_imagef(velocity_out, convert_int2(pos), (float4)(sum.xy, 0, 0));
}

struct fluid_particle
{
    float2 pos;
};

__kernel
void fluid_advect_particles(__read_only image2d_t velocity, __global struct fluid_particle* particles, int particles_num, float timestep, float2 scale)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    float2 pos = particles[gid].pos;

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    pos += 0.5f;

    float rdx = 1.f / GRID_SCALE;


    float2 new_pos = pos + timestep * rdx * read_imagef(velocity, sam, pos / scale).xy;

    particles[gid].pos = new_pos - 0.5f;
}

///READBACK INFO:
///float2 velocity
///float occupied

__kernel
void fluid_fetch_velocities(__read_only image2d_t velocity, __read_only image2d_t particles_in, __global float2* positions, int num_positions, __global float* out)
{
    int gid = get_global_id(0);

    if(gid >= num_positions)
        return;

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    sampler_t sam_near = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    int2 dim = get_image_dim(velocity);

    float2 pos = positions[gid];

    ///its easier to flip this here
    pos.y = dim.y - pos.y;

    float4 blocked = read_imagef(particles_in, sam_near, pos);

    float4 val = read_imagef(velocity, sam, pos);

    int found_gid = blocked.x - 1;

    int is_blocked = found_gid >= 0;

    out[gid*3 + 0] = val.x;
    out[gid*3 + 1] = val.y;
    out[gid*3 + 2] = is_blocked;
}

typedef uint uint32_t;
typedef uchar uint8_t;

struct physics_particle
{
    float2 pos;
    uint32_t icol;
};

uint32_t rgba_to_uint(float4 rgba)
{
    rgba = clamp(rgba, 0.f, 1.f);

    uint8_t r = rgba.x * 255;
    uint8_t g = rgba.y * 255;
    uint8_t b = rgba.z * 255;
    uint8_t a = rgba.w * 255;

    uint32_t ret = (r << 24) | (g << 16) | (b << 8) | a;

    return ret;
}

float4 uint_to_rgba(uint32_t val)
{
    uint8_t a = val & 0xFF;
    uint8_t b = (val >> 8) & 0xFF;
    uint8_t g = (val >> 16) & 0xFF;
    uint8_t r = (val >> 24) & 0xFF;

    return (float4){r, g, b, a} / 255.f;
}

///so. The problem with this function
///is that if we free a hole, next frame we may very well fill it again aimlessly
///aka this is very unhelpful
///what we need is a systematic bias per area i think
///if i can identify issue with this function and fix it
///its likely i can port it into the main advection step and have good performance
float2 any_free_neighbour_pos(float2 occupied, __read_only image2d_t physics_particles, __read_only image2d_t boundaries, int* found)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    int mult = 1;

    int2 iv = convert_int2(round(occupied + 0.5f));

    if((iv.y) % 2 == 0)
    {
        iv.x++;
    }

    int iocc = iv.x;

    if((iocc % 2) == 0)
    {
        mult = -1;
    }

    for(int y=-1; y<=-1; y++)
    {
        for(int x=-1*mult; x<=-1*mult; x++)
        {
            if(x == 0 && y == 0)
                continue;

            float2 rcd = occupied + (float2){x, y};

            float4 res = read_imagef(physics_particles, sam, rcd);

            if(res.x > 0)
                continue;

            float4 r2 = read_imagef(boundaries, sam, rcd);

            if(r2.x > 0)
                continue;

            *found = 1;

            return rcd;
        }
    }

    *found = 0;

    return occupied;
}

#if 1
float2 get_free_neighbour_pos(float2 move_vector, float2 initial, float2 occupied, __read_only image2d_t physics_particles, __read_only image2d_t boundaries, int* found)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    for(int y=-1; y<=1; y++)
    {
        for(int x=-1; x<=1; x++)
        {
            if(x == 0 && y == 0)
                continue;

            float2 to_initial_v = initial - occupied;
            float2 current_v = (float2){x, y};

            float angle = acos(dot(normalize(to_initial_v), normalize(current_v)));

            //if(fabs(angle) >= M_PI/2.f)
            //    continue;

            float2 nmove = normalize(move_vector);
            float2 ncurrent = normalize(current_v);

            float nangle = acos(dot(nmove, ncurrent));

            if(fabs(nangle) <= M_PI/2.f)
                continue;

            float2 rcd = occupied + (float2){x, y};

            float4 res = read_imagef(physics_particles, sam, rcd + 0.5f);

            if(res.x > 0)
                continue;

            int4 r2 = read_imagei(boundaries, sam, rcd + 0.5f);

            if(r2.x > 0)
                continue;

            float2 ret = rcd;

            *found = 1;

            return ret;
        }
    }

    *found = 0;

    return occupied;
}
#endif


///so first: need to check if any particles are between us and destination
///stop if we hit one
///ok having numerical issues
///what would be easier is giving them an integer coordinate
///accumulate velocities
///and then try moving them when accumulated > 1 in any direction

///implement a simple decision rule
///if we read the pixel texture and find a gid + 1 there that's > than our own
///we move our pixel somewhere else

///something is preventing particles from falling downwards in clumped conditions
///is it because above particles are falling down first and breaking stuff?

///ok, all fin
///next up, simulate fluid flow resistance by reducing impact of fluid velocity on particles depending on how many
///others are around the current particle
__kernel
void falling_sand_physics(__read_only image2d_t velocity, __global struct physics_particle* particles, int particles_num, float timestep, float2 scale,
                          __read_only image2d_t physics_particles_in, __write_only image2d_t physics_particles_out, __read_only image2d_t physics_boundaries)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    float2 pos = particles[gid].pos;
    //float4 ecol = particles[gid].col;

    //ecol = (float4)(0.3, 0.3, 1, 1);

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float rdx = 1.f / GRID_SCALE;

    float2 gravity = {0, -0.098};

    float2 new_pos = pos + timestep * rdx * read_imagef(velocity, sam, pos / scale).xy + gravity;

    float4 current_physp = read_imagef(physics_particles_in, sam, pos);

    int gcid = current_physp.x - 1;

    bool blocked = true;

    if(gcid > gid)
    {
        ///uuh.. resolve upwards?
        ///maybe should resolve away from velocity vector
        new_pos = pos + (float2){0, 1};
        blocked = false;
    }


    float2 diff = new_pos - pos;

    float max_dist = ceil(max(fabs(diff.x), fabs(diff.y)));

    if(max_dist == 0)
    {
        write_imagef(physics_particles_out, convert_int2(pos), (float4)(gid + 1,0,0,0));
        return;
    }

    float2 step = diff / max_dist;

    float2 cpos = pos;

    new_pos = cpos;

    int2 first_bound = read_imagei(physics_boundaries, sam, cpos).x;

    if(first_bound.x > 0)
    {
        write_imagef(physics_particles_out, convert_int2(pos), (float4)(gid + 1,0,0,0));
        return;
    }

    int would_move = 0;

    for(int i=0; i < max_dist + 1; i++, cpos += step)
    {
        int4 bound = read_imagei(physics_boundaries, sam, cpos);

        if(bound.x == 1)
            break;

        ///no self collision
        if(all(convert_int2(round(cpos + 0.5f)) == convert_int2(round(pos + 0.5))))
        {
            new_pos = cpos;
            continue;
        }

        if(blocked)
        {
            float4 val = read_imagef(physics_particles_in, sam, cpos);

            int found_gid = val.x - 1;

            ///ok. What we really need to do is look in the direction of motion
            ///and say, can we move to a neighbouring pixel?
            if(val.x > 0 && found_gid != gid)
            {
                would_move = 1;

                break;
            }
        }

        new_pos = cpos;
    }

    particles[gid].pos = new_pos;
    //particles[gid].col = ecol;

    write_imagef(physics_particles_out, convert_int2(new_pos), (float4)(gid + 1, would_move, 0, 0));
}


///new strategy
///single threading is proundly much easier to deal with
///so, each thread will have an 8x8 block that it is responsible for
///contract: Must set physics particles out
__kernel
void falling_sand_disimpact(__global struct physics_particle* particles, int particles_num,
                            __read_only image2d_t physics_particles_in, __write_only image2d_t physics_particles_out,
                            __read_only image2d_t physics_boundaries, int2 offset)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    int2 pos = (int2){get_global_id(0), get_global_id(1)};

    int2 global_bounds = (int2){get_image_width(physics_particles_in), get_image_height(physics_particles_out)};

    int2 block_size = (int2){2, 2};

    pos = pos * block_size + offset;

    bool broke = false;

    for(int y=0; y < block_size.y; y++)
    {
        for(int x=0; x < block_size.x; x++)
        {
            int2 global_pos = pos + (int2){x, y};

            float4 vals = read_imagef(physics_particles_in, sam, global_pos);

            int gid = vals.x - 1;

            if(gid < 0)
                continue;

            if(vals.y == 0)
                continue;

            if(read_imagei(physics_boundaries, sam, global_pos).x > 0)
                continue;

            int found = 0;

            float2 nval = any_free_neighbour_pos(particles[gid].pos, physics_particles_in, physics_boundaries, &found);

            if(found)
            {
                particles[gid].pos = nval;

                //particles[gid].col = (float4)(0, 1, 0, 1);

                broke = true;

                break;
            }
        }

        if(broke)
            break;
    }

    for(int y=0; y < block_size.y; y++)
    {
        for(int x=0; x < block_size.x; x++)
        {
            int2 lid = pos + (int2){x, y};

            int gid = read_imagef(physics_particles_in, sam, lid).x - 1;

            if(gid < 0)
                continue;

            float2 rpos = particles[gid].pos;

            write_imagef(physics_particles_out, convert_int2(rpos), (float4){gid + 1, 0,0,0});
        }
    }
}

///write strength fraction out
///the kernel which handles the boundary generation isn't robust enough yet
///for what i want to do here
__kernel
void falling_sand_edge_boundary_condition(__read_only image2d_t physics_particles_in, __read_only image2d_t fixed_boundaries,
                                          __write_only image2d_t boundaries_out, float2 scale,
                                          __read_only image2d_t velocity_in, __write_only image2d_t velocity_out)
{
    int2 pos = (int2){get_global_id(0), get_global_id(1)};

    int2 dim = (int2){get_image_width(physics_particles_in), get_image_height(physics_particles_in)};

    if(any(pos < 1) || any(pos >= dim-1))
        return;

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float val = read_imagef(physics_particles_in, sam, pos).x;

    int gid = val - 1;

    ///if we're a particle skippity skip
    ///remove this for no jitter
    //if(gid >= 0)
    //    return;

    if(read_imagei(fixed_boundaries, sam, pos).x > 0)
        return;

    float4 vel = read_imagef(velocity_in, sam, convert_int2(convert_float2(pos) / scale));

    int num_found = 0;

    for(int y=-1; y <= 1; y++)
    {
        for(int x=-1; x <= 1; x++)
        {
            //if(x == 0 && y == 0)
            //    continue;

            int2 global_offset = pos + (int2){x, y};

            if(any(global_offset < 1) || any(global_offset >= dim-1))
                continue;

            float nval = read_imagef(physics_particles_in, sam, global_offset).x;

            int ngid = nval - 1;

            if(ngid >= 0)
            {
                num_found++;
            }
        }
    }

    float frac = num_found / 9.f;

    ///TODO URGENT: TIMESTEP
    vel = vel - vel * 0.02f * frac;

    if(frac == 1)
        frac = 0;

    frac = frac / 16.f;

    frac = 0;

    write_imagef(boundaries_out, convert_int2(convert_float2(pos) / scale), frac);


    write_imagef(velocity_out, convert_int2(convert_float2(pos) / scale), vel);

    //if(frac > 0)
    //    printf("frac %f\n", frac);
}

///maybe this should work on a pixel by pixel basis?
///would run in constant time rather than variable on pixels
///probably more memory friendly too
__kernel
void falling_sand_render(__global struct physics_particle* particles, int particles_num, __write_only image2d_t screen)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    int gw = get_image_width(screen);
    int gh = get_image_height(screen);

    float2 pos = particles[gid].pos;

    if(pos.x >= gw-1 || pos.x < 0 || pos.y >= gh-1 || pos.y < 0)
        return;

    //float4 col = particles[gid].col;

    float4 col = uint_to_rgba(particles[gid].icol);

    write_imagef(screen, convert_int2(pos), (float4){col.xyz,1});

    /*for(int y=-1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            if(abs(x) == abs(y) && abs(x) == 1)
                continue;

            float2 new_pos = pos + (float2){x, y};

            write_imagef(screen, convert_int2(new_pos), (float4)(1,1,1,1));
        }
    }*/
}

__kernel
void fluid_render_particles(__global struct fluid_particle* particles, int particles_num, __write_only image2d_t screen)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    int gw = get_image_width(screen);
    int gh = get_image_height(screen);

    float2 pos = particles[gid].pos;

    if(pos.x >= gw-2 || pos.x < 1 || pos.y >= gh-2 || pos.y < 1)
        return;

    for(int y=-1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            if(abs(x) == abs(y) && abs(x) == 1)
                continue;

            float2 new_pos = pos + (float2){x, y};

            write_imagef(screen, convert_int2(new_pos), (float4)(1,1,1,1));
        }
    }
}

__kernel
void wavelet_w_of(__read_only image2d_t noise_in, __write_only image2d_t w_of)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(noise_in);
    int gh = get_image_height(noise_in);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float centre = read_imagef(noise_in, sam, pos).x;

    float y1 = read_imagef(noise_in, sam, pos + (float2){0.f, 1.f}).x;
    float x1 = read_imagef(noise_in, sam, pos + (float2){1.f, 0.f}).x;

    float2 w2d = (float2){y1 - centre, -(x1 - centre)};

    write_imagef(w_of, convert_int2(pos), (float4)(w2d, 0, 0));
}

///dim is UPSCALED dimension, pos is upscaled pos
///should absolutely be caching get y of
float2 get_y_of(float2 pos, __read_only image2d_t w_of_in, float imin, float imax, float2 dim)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_TRUE |
                    CLK_ADDRESS_REPEAT |
                    CLK_FILTER_LINEAR;
    float sum = 0;

    for(float i = imin; i < imax; i+=1.f)
    {
        float2 coord = pow(2, i) * (pos + 0.5f);

        float w_of = read_imagef(w_of_in, sam, coord / dim).x;

        sum += w_of * pow(2, -(5.f/6) * (i - imin));
    }

    return sum;
}

float calculate_energy(float2 vel)
{
    float len = fast_length(vel);

    //float nrg = 0.5 * len * len;

    len += 0.5f;

    len *= 10.f;

    len = max(len, 0.001f);

    len = 1.f / len;

    len = clamp(len, 0.00001f, 0.8f);

    return len * 10;

    len = clamp(len, 0.0001f, 1.f);

    float nrg = 1.f / len;

    //return 1.f;

    return (nrg) * 1000;

    /*float len = fast_length(vel);

    float nrg = 0.5 * len * len;

    return nrg * 100;*/

    //return 10.f;
}

///runs in upscaled space
__kernel
void wavelet_upscale(__read_only image2d_t w_of_in, __read_only image2d_t velocity_in, __write_only image2d_t upscaled_velocity_out, float timestep)
{
    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(upscaled_velocity_out);
    int gh = get_image_height(upscaled_velocity_out);

    if(pos.x >= gw || pos.y >= gh)
        return;

    sampler_t sam = CLK_NORMALIZED_COORDS_TRUE |
                    CLK_ADDRESS_REPEAT |
                    CLK_FILTER_LINEAR;

    float imin = -6;
    float imax = 2;

    /*float2 interpolated = read_imagef(velocity_in, sam, (pos + 0.5f) / (float2){gw, gh}).xy;

    float2 y_of = get_y_of(pos, w_of_in, imin, imax, (float2){gw, gh});

    ///ok we're gunna cheat a little bit
    ///they use spectral energy something something to estimate in the complex case
    ///to get local weighting of essentially energy to correctly only inject energy with forward scattering
    ///and in the simple case they use a global weight
    ///neither of these are really acceptable, so just blatantly cheat and use velocity to estimate local energy
    float et_term = calculate_energy(interpolated);

    float2 final_velocity = interpolated + pow(2.f, -5.f/6) * et_term * y_of;*/

    float2 final_velocity = read_imagef(velocity_in, sam, (pos + 0.5f) / (float2){gw, gh}).xy;

    /*float rdx = 1.f / GRID_SCALE;

    float2 new_pos = pos - timestep * rdx * read_imagef(velocity, sam, pos * vdim / adim).xy;

    float4 new_value = read_imagef(advect_quantity_in, sam, new_pos);

    write_imagef(advect_quantity_out, convert_int2(pos), new_value);*/

    float rdx = 1.f / GRID_SCALE;

    float2 back_in_time_pos = pos - timestep * rdx * final_velocity;

    float2 new_value = read_imagef(velocity_in, sam, (back_in_time_pos + 0.5f) / (float2){gw, gh}).xy;

    float2 new_y_of = get_y_of(back_in_time_pos, w_of_in, imin, imax, (float2){gw, gh});

    float new_et_term = calculate_energy(new_value);

    float2 advected_velocity = new_value + pow(2.f, -5.f/6) * new_et_term * new_y_of;

    write_imagef(upscaled_velocity_out, convert_int2(pos), (float4)(advected_velocity, 0, 0));

    ///so we wanna advect in this kernel. Its very expensive to store intermediate data
    ///So: Go back in time, and generate the velocity there as well so we can store that?
}

__kernel
void lighting_raytrace_point(float2 point, float radius, int num_tracers, __read_only image2d_t fluid, __write_only image2d_t screen)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    int gid = get_global_id(0);

    if(gid >= num_tracers)
        return;

    int sw = get_image_width(screen);
    int sh = get_image_height(screen);

    float angle = 2 * M_PI * (float)gid / num_tracers;

    float2 dir = (float2){cos(angle), sin(angle)};

    float2 finish = point + dir * radius;

    //float max_dir = max(fabs(dir.x), fabs(dir.y));

    //dir = dir / max_dir;

    //int num = max(fabs(dir.x * radius), fabs(dir.y * radius));

    int num = radius;

    float2 cur = point;

    float brightness = 1.f;

    ///after a distance of radius, we should be fully absorbed
    ///in... a uniform cloud..?
    for(int i=0; i < num; i++, cur += dir)
    {
        float density = read_imagef(fluid, sam, cur + 0.5f).x;

        float amount_reflected = brightness * density * 0.01f;

        brightness -= amount_reflected;

        float distance_curve = 1.f - (float)i / num;

        float extra_bright = 200;

        if(cur.x >= 0 && cur.y >= 0 && cur.x < sw && cur.y < sh)
        {
            write_imagef(screen, convert_int2(cur), (float4)(amount_reflected*distance_curve*extra_bright, 0, 0, 1));
        }
    }
}

__kernel
void blank(__global int* value)
{
    *value = *value + 1;
}

///BEGIN OPENCL BULLET STUFF

typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;
	unsigned int m_collidableIdx;
	float m_invMass;
	float m_restituitionCoeff; ///aha, here you are! TODO: FOUND RESTITUTION
	float m_frictionCoeff;
} Body;

__kernel void
	copyTransformsToVBOKernel( __global Body* gBodies, __global float4* posOrnColor, const int numNodes)
{
	int nodeID = get_global_id(0);
	if( nodeID < numNodes )
	{
		posOrnColor[nodeID] = (float4) (gBodies[nodeID].m_pos.xyz,1.0);
		posOrnColor[nodeID + numNodes] = gBodies[nodeID].m_quat;
	}
}

__kernel
void hacky_render(__read_only image2d_t tex, __write_only image2d_t screen, __global Body* gBodies, int max_bodies)
{
    int idx = get_global_id(0);

    int2 dim = get_image_dim(tex);

    if(idx >= dim.x * dim.y)
        return;

    int body_idx = get_global_id(1);

    if(body_idx >= max_bodies)
        return;

    int2 id;
    id.x = idx % dim.x;
    id.y = idx / dim.x;

    sampler_t sam_near = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_NONE |
                    CLK_FILTER_NEAREST;

    int4 val = read_imagei(tex, sam_near, id);

    int2 pos = convert_int2(gBodies[body_idx].m_pos.xy);

    int2 offset = convert_int2(id + pos);

    int2 sdim = get_image_dim(screen);

    if(any(offset < 0) || any(offset >= sdim))
        return;

    if(val.w == 0)
        return;

    ///check this works
    write_imagef(screen, offset, convert_float4(val) / 255.f);
}

__kernel
void keep_upright_and_fluid(__global Body* gBodies, int max_bodies, __read_only image2d_t fluid_velocity, float timestep_s, float frame_timestep_s)
{
    int idx = get_global_id(0);

    if(idx >= max_bodies)
        return;

    float2 dim = convert_float2(get_image_dim(fluid_velocity));

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    float2 pos = gBodies[idx].m_pos.xy;

    //gBodies[idx].m_pos.z = 0;
    //gBodies[idx].m_linVel.z = 0;

    //gBodies[idx].m_linVel.z = -gBodies[idx].m_pos.z/2.f;

    //gBodies[idx].m_restituitionCoeff = 0.25f;

    if(any(pos < 0) || any(pos >= dim))
        return;

    float2 current_velocity = gBodies[idx].m_linVel.xy;
    float2 destination_velocity = read_imagef(fluid_velocity, sam, pos).xy * timestep_s;

    float2 velocity_diff = (destination_velocity - current_velocity) * 0.1f;// * mass

    gBodies[idx].m_linVel.xy += velocity_diff;
}

///END OF OPENCL BULLET STUFF

///utility

__kernel
void clear_image(__write_only image2d_t img)
{
    int2 val = (int2){get_global_id(0), get_global_id(1)};

    int2 dim = get_image_dim(img);

    if(any(val < 0) || any(val >= dim))
        return;

    write_imagef(img, val, 0);
}
