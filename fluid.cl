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

#define GRID_SCALE 1.f

__kernel
void fluid_advection(__read_only image2d_t velocity, __read_only image2d_t advect_quantity_in, __write_only image2d_t advect_quantity_out, float timestep)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                    CLK_ADDRESS_CLAMP_TO_EDGE |
                    CLK_FILTER_LINEAR;

    float2 pos = (float2){get_global_id(0), get_global_id(1)};

    int gw = get_image_width(velocity);
    int gh = get_image_height(velocity);

    if(pos.x >= gw || pos.y >= gh)
        return;

    pos += 0.5f;

    float rdx = 1.f / GRID_SCALE;

    float2 new_pos = pos - timestep * rdx * read_imagef(velocity, sam, pos).xy;

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

    /*float dx = 0.1f;
    float dt = 0.5f;
    float n = 10.1; ///viscosity apparently

    //float dt = 16.f / 1000.f;

    float alpha = (dx * dx) / (n * dt);
    float beta = 1.f/(4 + alpha);*/

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

    float4 div = half_rdx * ((wR.x - wL.x) + (wT.y - wB.y));

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
void fluid_render(__read_only image2d_t field, __write_only image2d_t screen)
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

    position.y = gh - position.y;

    if(fast_length(pos - position) > max_len)
        return;

    direction = fast_normalize(direction);

    direction.y = -direction.y;

    float flen = 1.f - fast_length(pos - position) / max_len;

    float2 extra = force * direction * flen;

    float4 old_vel = read_imagef(velocity_in, sam, pos);

    float4 sum = old_vel + extra.xyxy;

    write_imagef(velocity_out, convert_int2(pos), sum);
}

struct fluid_particle
{
    float2 pos;
};

__kernel
void fluid_advect_particles(__read_only image2d_t velocity, __global struct fluid_particle* particles, int particles_num, float timestep)
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


    float2 new_pos = pos + timestep * rdx * read_imagef(velocity, sam, pos).xy;

    particles[gid].pos = new_pos - 0.5f;
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
