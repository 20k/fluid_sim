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

    //pos += 0.5f;

    float rdx = 1.f / GRID_SCALE;


    float2 new_pos = pos + timestep * rdx * read_imagef(velocity, sam, pos / scale).xy;

    particles[gid].pos = new_pos;// - 0.5f;
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

float2 get_y_of(float2 pos, __read_only image2d_t w_of_in, float imin, float imax, float2 dim)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_TRUE |
                    CLK_ADDRESS_REPEAT |
                    CLK_FILTER_NEAREST;
    float sum = 0;

    for(float i = imin; i < imax; i+=1.f)
    {
        float2 coord = pow(2, i) * pos;

        float w_of = read_imagef(w_of_in, sam, (coord + 0.5f) / dim).x;

        sum += w_of * pow(2, -(5.f/6) * (i - imin));
    }

    return sum;
}

__kernel
void wavelet_upscale(__read_only image2d_t w_of_in, __write_only image2d_t velocity_out)
{

}
