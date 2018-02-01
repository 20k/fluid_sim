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

    float2 bound = read_imagef(boundaries, sam, pos).xy;

    if(bound.x == 1)
        val.xyz = 1;

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
__kernel
void fluid_boundary_tex(__read_only image2d_t field_in, __write_only image2d_t field_out, float scale, __read_only image2d_t boundary_texture)
{
    int2 ipos = (int2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(field_in);
    int gh = get_image_height(field_in);

    if(ipos.x >= gw || ipos.y >= gh)
        return;

    float2 pos = convert_float2(ipos);

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    sampler_t sam_near = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 vals = read_imagef(boundary_texture, sam_near, convert_float2(ipos) + 0.5f).xy;

    if(vals.x != 1)
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

    for(int i=0; i < angles; i++)
    {
        float angle_frac = 2 * M_PI * (float)i / angles;

        float2 offset = angle_to_offset(angle_frac);

        float2 nval = read_imagef(boundary_texture, sam_near, pos + 0.5f + offset).xy;

        if(nval.x == 1)
        {
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

        float2 nval = read_imagef(boundary_texture, sam_near, pos + 0.5f + offset).xy;

        if(nval.x == 1)
        {
            fnormalangle = (angle_frac + 2 * M_PI * (float)range_start / angles) / 2.f;
            break;
        }
    }

    float2 fnormal = {cos(fnormalangle), sin(fnormalangle)};

    int2 p1 = convert_int2(pos + fnormal);
    int2 p2 = convert_int2(pos - fnormal);

    float4 rv1 = read_imagef(field_in, sam, pos + fnormal * 2 + 0.5f);
    float4 rv2 = read_imagef(field_in, sam, pos - fnormal * 2 + 0.5f);

    rv1 = rv1 * scale;
    rv2 = rv2 * scale;

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

    write_imagef(buffer, convert_int2(pos), (float4)(1.f, angle, 0, 0));
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

struct physics_particle
{
    float2 pos;
    float2 unused_velocity;
    float4 col;
};

///so. The problem with this function
///is that if we free a hole, next frame we may very well fill it again aimlessly
///aka this is very unhelpful
///what we need is a systematic bias per area i think
float2 any_free_neighbour_pos(float2 occupied, __read_only image2d_t physics_particles, __read_only image2d_t boundaries, int* found)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    for(int y=-1; y<=-1; y++)
    {
        for(int x=-1; x<=-1; x++)
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

            float4 r2 = read_imagef(boundaries, sam, rcd + 0.5f);

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

#if 0
__kernel
void falling_sand_physics(__read_only image2d_t velocity, __global struct physics_particle* particles, int particles_num, float timestep, float2 scale,
                          __read_only image2d_t physics_particles_in, __write_only image2d_t physics_particles_out, __read_only image2d_t physics_boundaries)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    ///snapped to grid
    float2 pos = particles[gid].pos;
    float2 extra_vel = particles[gid].unused_velocity;

    ///uncomment for compiler bugs!
    ///i was really rather hoping the state of opencl would be better than this
    ///when i came back to it
    //particles[gid].unused_velocity = (float2)(0,0);

    float2 gravity = {0, -0.098};

    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float rdx = 1.f / GRID_SCALE;

    float2 new_pos = extra_vel + pos + timestep * rdx * read_imagef(velocity, sam, (pos + 0.5f) / scale).xy + gravity;

    float2 diff = new_pos - pos;

    float max_diff = max(fabs(diff.x), fabs(diff.y));

    //printf("%f %f %f %f md\n", new_pos.x, new_pos.y, pos.x, pos.y);

    ///uncomment for compiler bugs with the above comment
    //printf("%f %f\n", extra_vel.x, extra_vel.y);

    ///ensure we're stepping at least one in any direction
    if(max_diff < 1)
    {
        //printf("hello %f %f %f %f\n", diff.x, diff.y, extra_vel.x, extra_vel.y);

        particles[gid].unused_velocity = diff;
        write_imagef(physics_particles_out, convert_int2(pos), (float4)(gid+1,0,0,0));
        return;
    }

    ///so eg if we want to move 0.5, 1.5, we get a remainder of
    ///0.5, 0.5
    ///subtract that, our diff to move is 0, 1
    float2 extra = fmod(diff, 1.f);

    //printf("%f %f\n", extra.x, extra.y);

    diff -= extra;

    int steps = max(fabs(diff.x), fabs(diff.y));

    float2 to_step = diff / steps;

    float2 last_valid = pos;

    float2 to_test = pos;
    to_test += to_step;

    for(int i=0; i < steps; i++, to_test += to_step)
    {
        float4 bound = read_imagef(physics_boundaries, sam, to_test + 0.5f);

        ///hit a boundary, should not accumulate velocity past it
        if(bound.x == 1)
        {
            extra = (float2){0,0};
            break;
        }

        float4 part = read_imagef(physics_particles_in, sam, to_test + 0.5f);

        ///hit a particle, dont accumulate velocity
        if(part.x > 0)
        {
            extra = (float2){0,0};

            int found = 0;

            //float2 nval = get_free_neighbour_pos(pos, to_test, physics_particles_in, physics_boundaries, &found);

            if(found)
            {
                //last_valid = nval;
            }

            break;
        }

        last_valid = to_test;
    }

    particles[gid].unused_velocity = extra;

    particles[gid].pos = last_valid;

    write_imagef(physics_particles_out, convert_int2(last_valid), (float4)(gid+1,0,0,0));
}
#endif

#if 1

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
__kernel
void falling_sand_physics(__read_only image2d_t velocity, __global struct physics_particle* particles, int particles_num, float timestep, float2 scale,
                          __read_only image2d_t physics_particles_in, __write_only image2d_t physics_particles_out, __read_only image2d_t physics_boundaries,
                          unsigned int counter)
{
    int gid = get_global_id(0);

    if(gid >= particles_num)
        return;

    float2 pos = particles[gid].pos;
    float4 ecol = particles[gid].col;

    ecol = (float4)(0.3, 0.3, 1, 1);

    //pos += 0.5f;

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

        int found = 0;

        float2 nval = any_free_neighbour_pos(pos, physics_particles_in, physics_boundaries, &found);

        if(found && ((counter % 128) == (gid % 128)))
        {
            //new_pos = nval;
        }
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

    float4 first_bound = read_imagef(physics_boundaries, sam, cpos);

    if(first_bound.x > 0)
    {
        write_imagef(physics_particles_out, convert_int2(pos), (float4)(gid + 1,0,0,0));
        return;
    }

    //cpos += step;

    float2 desire_dir = (float2)(0,0);

    for(int i=0; i < max_dist + 1; i++, cpos += step)
    {
        float4 bound = read_imagef(physics_boundaries, sam, cpos);

        if(bound.x == 1)
            break;

        ///no self collision
        /*if(all(convert_int2(round(cpos + 0.5f)) == convert_int2(round(pos + 0.5))))
        {
            new_pos = cpos;
            continue;
        }*/

        float4 val = read_imagef(physics_particles_in, sam, cpos);

        int found_gid = val.x - 1;

        ///ok. What we really need to do is look in the direction of motion
        ///and say, can we move to a neighbouring pixel?
        if(val.x > 0 && found_gid != gid && blocked)
        {
            int found = 0;

            ///the problem here is that we're piling multiple pixels onto the same spot
            float2 nval = get_free_neighbour_pos(step, pos, cpos, physics_particles_in, physics_boundaries, &found);

            desire_dir.x = 1;
            desire_dir.y = 1;

            if(found && ((counter % 1024) == (gid % 1024)))
            {
                //desire_dir = nval - cpos;
                //new_pos = nval;
            }

            //ecol = (float4){0, 1, 0, 1};

            break;
        }

        new_pos = cpos;
    }

    particles[gid].pos = new_pos;
    particles[gid].col = ecol;

    write_imagef(physics_particles_out, convert_int2(new_pos), (float4)(gid + 1, desire_dir.x, desire_dir.y, 0));
}
#endif

///current plan:
///take empty pixel
///for every pixel around me, check desire dirs (and gid)
///if desire_dirs > 0 and we're a suitable pixel (angle < pi / 2 etc), move that particle into me
///should be stable
#if 0
__kernel
void falling_sand_disimpact(__global struct physics_particle* particles, int particles_num,
                            __read_only image2d_t physics_particles_in, __write_only image2d_t physics_particles_out,
                            __read_only image2d_t physics_boundaries)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_NEAREST;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    float4 in = read_imagef(physics_particles_in, sam, pos);

    if(in.x != 0)
    {
        write_imagef(physics_particles_out, convert_int2(pos), in);
        return;
    }

    float4 is_bound = read_imagef(physics_boundaries, sam, pos);

    if(is_bound.x > 0)
        return;

    ///ok. We're a hole
    ///check around pixel neighbourhood
    ///if we find a pixel with a desire path, move them into us
    ///lots of other pixels will try to grab it
    ///so: we let them all do it, and whoever gets there last wins

    for(int y=-1; y <= 1; y++)
    {
        for(int x=-1; x <= 1; x++)
        {
            if(x == 0 && y == 0)
                continue;

            //if(abs(x) == abs(y))
            //    continue;

            float2 new_pos = pos + (float2){x, y};

            float4 val = read_imagef(physics_particles_in, sam, new_pos);

            int gid = val.x - 1;

            if(gid < 0)
                continue;

            if(read_imagef(physics_boundaries, sam, new_pos).x > 0)
                continue;

            float2 desire = val.yz;

            if(desire.x == 0 && desire.y == 0)
                continue;

            //printf("hello\n");

            particles[gid].pos = new_pos;

            write_imagef(physics_particles_out, convert_int2(new_pos), (float4)(gid + 1, 0,0,0));

            return;
        }
    }

    write_imagef(physics_particles_out, convert_int2(pos), (float4)(in.x, 0,0,0));
}
#endif // 0

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

    int2 block_size = (int2){4, 4};

    pos = pos * block_size + offset;

    bool broke = false;

    for(int y=0; y < block_size.y; y++)
    {
        for(int x=0; x < block_size.x; x++)
        {
            int2 global_pos = pos + (int2){x, y};

            if(read_imagef(physics_boundaries, sam, global_pos).x > 0)
                continue;

            float4 vals = read_imagef(physics_particles_in, sam, global_pos);

            int gid = vals.x - 1;

            if(gid < 0)
                continue;

            if(vals.y == 0 && vals.z == 0)
                continue;


            int found = 0;

            float2 nval = any_free_neighbour_pos(particles[gid].pos, physics_particles_in, physics_boundaries, &found);

            if(found)
            {
                particles[gid].pos = nval;

                particles[gid].col = (float4)(0, 1, 0, 1);

                broke = true;

                break;
            }
        }

        if(broke)
            break;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

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

    #if 0
    int local_pixels[4][4];
    int marked[4][4] = {{0}};

    for(int y=pos.y; y < pos.y + block_size.y; y++)
    {
        for(int x=pos.x; x < pos.x + block_size.x; x++)
        {
            int2 current_pixel = (int2){x, y};

            float4 ids = read_imagef(physics_particles_in, sam, current_pixel);

            local_pixels[y - pos.y][x - pos.x] = ids.x - 1;
        }
    }

    for(int y=pos.y; y < pos.y + block_size.y; y++)
    {
        for(int x=pos.x; x < pos.x + block_size.x; x++)
        {
            int2 current_pixel = (int2){x, y};

            if(any(current_pixel < 0) || any(current_pixel >= global_bounds))
                continue;

            if(marked[y - pos.y][x - pos.x])
                continue;

            float4 bound = read_imagef(physics_boundaries, sam, current_pixel).x;

            if(bound.x)
                continue;

            ///so take current particle
            ///if it wants to move, move it somewhere
            ///carry on

            float4 ids = read_imagef(physics_particles_in, sam, current_pixel);

            int found_id = local_pixels[y - pos.y][x - pos.x];

            if(found_id < 0)
                continue;

            ///wants to move
            if(ids.y == 0 && ids.z == 0)
                continue;

            ///just move it somewhere else for the moment

            bool term = false;

            //for(int oy=-1; oy <= 1; oy++)

            int oy = -1;
            {
                for(int ox=-1; ox <= 1; ox++)
                {
                    int2 lpix = (int2){ox + x - pos.x, oy + y - pos.y};
                    int2 gpix = (int2){ox + x, oy + y};

                    if(read_imagef(physics_boundaries, sam, gpix).x > 0)
                        continue;

                    if(ox == 0 && oy == 0)
                        continue;

                    if(any(lpix < 0) || any(lpix >= block_size))
                        continue;

                    int local_id = local_pixels[lpix.y][lpix.x];

                    if(local_id < 0)
                    {
                        //local_pixels[lpix.y][lpix.x] = local_pixels[y - pos.y][x - pos.x];
                        //local_pixels[y - pos.y][x - pos.x] = -1;
                        marked[y - pos.y][x - pos.x] = 1;
                        term = true;
                        break;
                    }
                }

                //if(term)
                //    break;
            }
        }
    }

    for(int y=pos.y; y < pos.y + block_size.y; y++)
    {
        for(int x=pos.x; x < pos.x + block_size.x; x++)
        {
            int2 current_pixel = (int2){x, y};
            int2 local_pixel = (int2){x - pos.x, y - pos.y};

            int l_id = local_pixels[local_pixel.y][local_pixel.x];

            if(l_id >= 0 && marked[local_pixel.y][local_pixel.x])
            {
                //particles[l_id].pos = (float2){x, y} + 0.5f;
                //particles[l_id].col = (float4)(1, 1, 1, 1);
            }

            //particles[l_id].col.x = get_global_id(0) / 100.f;
            //particles[l_id].col.y = get_global_id(1) / 100.f;

            write_imagef(physics_particles_out, current_pixel, (float4)(l_id + 1, 0, 0, 0));
        }
    }
    #endif

    #if 0
    float4 in = read_imagef(physics_particles_in, sam, pos);

    if(in.x != 0)
    {
        write_imagef(physics_particles_out, convert_int2(pos), in);
        return;
    }

    float4 is_bound = read_imagef(physics_boundaries, sam, pos);

    if(is_bound.x > 0)
        return;

    ///ok. We're a hole
    ///check around pixel neighbourhood
    ///if we find a pixel with a desire path, move them into us
    ///lots of other pixels will try to grab it
    ///so: we let them all do it, and whoever gets there last wins

    for(int y=-1; y <= 1; y++)
    {
        for(int x=-1; x <= 1; x++)
        {
            if(x == 0 && y == 0)
                continue;

            //if(abs(x) == abs(y))
            //    continue;

            float2 new_pos = pos + (float2){x, y};

            float4 val = read_imagef(physics_particles_in, sam, new_pos);

            int gid = val.x - 1;

            if(gid < 0)
                continue;

            if(read_imagef(physics_boundaries, sam, new_pos).x > 0)
                continue;

            float2 desire = val.yz;

            if(desire.x == 0 && desire.y == 0)
                continue;

            //printf("hello\n");

            particles[gid].pos = new_pos;

            write_imagef(physics_particles_out, convert_int2(new_pos), (float4)(gid + 1, 0,0,0));

            return;
        }
    }

    write_imagef(physics_particles_out, convert_int2(pos), (float4)(in.x, 0,0,0));
    #endif
}

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

    float4 col = particles[gid].col;

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
