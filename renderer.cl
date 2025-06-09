__kernel void render_kernel(
	__global float4* image, float p, float q
) {
	const int gid = get_global_id(0);
	image[gid] =  image[gid]*p + q; 
}