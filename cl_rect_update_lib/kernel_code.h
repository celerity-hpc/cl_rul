
namespace cl_rul {
	namespace kernels {

		constexpr const char* upload_2D = R"(
			#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
			#pragma OPENCL EXTENSION cl_khr_fp64: enable

			typedef struct { T x; } v_t;

			__kernel void upload_2D(
				__global v_t *src, __global v_t *trg,
				uint pos_x, uint pos_y,
				uint size_x, uint size_y,
				uint stride)
			{
				int i = get_global_id(0);
				int line = i/size_x + pos_y;
				int col = i%size_x + pos_x;
				trg[col + line*stride] = src[i];
			}
		)";
	}
}
