#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef struct { float x; } v_t;

__kernel void column_writer(__global v_t *src, __global v_t *trg, int target_col, int stride)
{
	int i = get_global_id(0);
	trg[target_col + i*stride] = src[i];
}
