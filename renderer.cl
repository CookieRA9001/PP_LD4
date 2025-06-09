const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void render_kernel(
	__read_only image2d_t in, __write_only image2d_t out,
	float p, float q
) {
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int2 pos = (int2)(gid_x, gid_y);
	uint4 pixel = read_imageui(in, smp, pos);
	float4 temp_pix = convert_float4(pixel);
	pixel = convert_uint4(temp_pix*p + q);
	// printf("Pixel at (%d, %d): %u %u %u %u\n", gid_x, gid_y, pixel.x, pixel.y, pixel.z, pixel.w);
	write_imageui(out, pos, pixel);
}