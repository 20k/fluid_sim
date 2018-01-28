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
                    CLK_FILTER_LINEAR;

    float4 val = read_imagef(test, sam, (int2){ix, iy});

    //printf("%f %f %f %f\n", val.x, val.y, val.z, val.w);

    write_imagef(screen, (int2){ix, iy}, val);
}
