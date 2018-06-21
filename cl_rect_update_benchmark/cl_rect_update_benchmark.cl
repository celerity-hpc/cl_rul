#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void column_writer(__global float *src, __global float *trg, int target_col, int stride)
{
	int i = get_global_id(0);
	trg[target_col + i*stride] = src[i];
}

__kernel void column_reader(__global float *src, __global float *trg, int target_col, int stride)
{
	int i = get_global_id(0);
	trg[i] = src[target_col + i*stride];
}