
#include "../ext/cl_utils.h"
#include <string.h>
#include <algorithm>

#include "../cl_rect_update_lib/cl_rect_update_lib.h"

#define KERNEL_FILE "cl_rect_update_benchmark.cl"
constexpr int NUM_REPETITIONS = 10;
constexpr int NUM_INNER_REPETITIONS = 10;
constexpr int NUM_SIZES = 8;
constexpr int START_SIZE = 64;
constexpr int NUM_TYPES = 6;


int main(int argc, char **argv) {
	if(argc != 2) {
		printf("Usage: cl_rect_update_benchmark [DEVICE NUM]\n");
		exit(1);
	}
	size_t device_num = atoi(argv[1]);

	cl_int errcode;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device = cluInitDevice(device_num, &context, &queue);
	printf(cluGetDeviceDescription(device, (unsigned int)device_num));

	cl_rul::init_rect_update_lib(context, device);

	// data

	cl_mem device_buffers[NUM_SIZES];
	cl_mem staging_buffer;
	void *host_buffers[NUM_SIZES];
	int side_lengths[NUM_SIZES];
	int byte_sizes[NUM_SIZES];

	{
		int side_length = START_SIZE;
		for(int i = 0; i < NUM_SIZES; ++i) {
			side_lengths[i] = side_length;
			byte_sizes[i] = side_length * side_length * sizeof(cl_float);
			device_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_sizes[i], NULL, &errcode);
			host_buffers[i] = malloc(byte_sizes[i]);
			CLU_ERRCHECK(errcode, "Failed to acquire device memory");
			side_length *= 2;
		}

		staging_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, side_length * sizeof(cl_float), NULL, &errcode);
		CLU_ERRCHECK(errcode, "Failed to acquire device memory for staging buffer");
	}

	// kernel

	cl_program program = cluBuildProgramFromFile(context, device, KERNEL_FILE, "");
	cl_kernel kernel_column_writer = clCreateKernel(program, "column_writer", &errcode);
	CLU_ERRCHECK(errcode, "Failed to create column_writer kernel");
	cl_kernel kernel_column_reader = clCreateKernel(program, "column_reader", &errcode);
	CLU_ERRCHECK(errcode, "Failed to create column_reader kernel");

	const char* names[NUM_TYPES] = { "Complete", "Rect/Rect", "Rect/Linear", "Individual" ,"Col Kernel", "Lib Kernel" };

	{
		printf("\nUPLOAD (Host -> GPU) times in microseconds\n");
		for(int i = 0; i < NUM_TYPES + 1; ++i) {
			printf("%12s ", i == 0 ? "Side length" : names[i - 1]);
			printf(i == NUM_TYPES ? "\n" : ", ");
		}

		double results[NUM_SIZES][NUM_TYPES];
		std::fill(&results[0][0], &results[NUM_SIZES][0], 1000000.0f);

		cl_event cl_ev_before; // just for measurement
		auto start_bench = [&](int s) {
			clEnqueueWriteBuffer(queue, device_buffers[s], CL_FALSE, 0, 1, host_buffers[s], 0, NULL, &cl_ev_before);
		};
		auto end_bench = [&](int s, int id, cl_event cl_ev_transfer) {
			clFinish(queue);
			cl_ulong start, end;
			CLU_ERRCHECK(clGetEventProfilingInfo(cl_ev_before, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL), "Error reading start time");
			CLU_ERRCHECK(clGetEventProfilingInfo(cl_ev_transfer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL), "Error reading end time");
			results[s][id] = std::min(results[s][id], ((double)(end - start) / 1000.0));
		};

		for(int r = 0; r < NUM_REPETITIONS; ++r) {
			for(int s = 0; s < NUM_SIZES; ++s) {

				/// 1. complete transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_complete;
					errcode = clEnqueueWriteBuffer(queue, device_buffers[s], CL_FALSE, 0, byte_sizes[s], host_buffers[s], 0, NULL, &ev_complete);
					CLU_ERRCHECK(errcode, "Error enqueueing complete transfer");
					end_bench(s, 0, ev_complete);
				}

				/// 2. rect / rect column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_transfer;
					const size_t buffer_origin[3] = { 0, 0, 0 };
					const size_t host_origin[3] = { 0, 0, 0 };
					const size_t region[3] = { sizeof(float), (size_t)side_lengths[s], 1 };
					size_t buffer_row_pitch = side_lengths[s] * sizeof(float);
					size_t buffer_slice_pitch = byte_sizes[s];
					size_t host_row_pitch = side_lengths[s] * sizeof(float);
					size_t host_slice_pitch = byte_sizes[s];
					errcode = clEnqueueWriteBufferRect(queue, device_buffers[s], CL_FALSE,
						buffer_origin, host_origin, region,
						buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
						host_buffers[s], 0, NULL, &ev_transfer);
					if(errcode == CL_SUCCESS) {
						end_bench(s, 1, ev_transfer);
					}
				}

				/// 3. rect / linear column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_transfer = cl_rul::upload_rect<cl_float, cl_rul::ClRect>(queue, device_buffers[s],
					{ (size_t)side_lengths[s], (size_t)side_lengths[s], 1u }, { { 0u,0u,0u },{ 1u,(size_t)side_lengths[s],1u } }, (cl_float*)host_buffers[s]);
					end_bench(s, 2, ev_transfer);
				}

				/// 4. individual column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					// only test this on smaller sizes, kills some implementations
					if(side_lengths[s] <= 512) {
						start_bench(s);
						cl_event* evs = (cl_event*)alloca(sizeof(cl_event)*side_lengths[s]);
						for(int i = 0; i < side_lengths[s]; ++i) {
							errcode = clEnqueueWriteBuffer(queue, device_buffers[s], CL_FALSE, i * side_lengths[s] * sizeof(float), sizeof(cl_float), host_buffers[s], 0, NULL, &evs[i]);
							CLU_ERRCHECK(errcode, "Error enqueueing rect transfer");
						}
						end_bench(s, 3, evs[side_lengths[s] - 1]);
					}
				}

				/// 5. kernel-based transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);

					// transfer linearly to staging buffer
					cl_event ev_staging;
					errcode = clEnqueueWriteBuffer(queue, staging_buffer, CL_FALSE, side_lengths[s] * sizeof(float), sizeof(cl_float), host_buffers[s], 0, NULL, &ev_staging);
					CLU_ERRCHECK(errcode, "Error enqueueing staging transfer");

					cl_event ev_kernel;
					int zero = 0;
					cluSetKernelArguments(kernel_column_writer, 4, sizeof(cl_mem), (void *)&staging_buffer, sizeof(cl_mem), (void *)&device_buffers[s], sizeof(cl_int), &zero, sizeof(cl_int), &side_lengths[s]);
					size_t global_size = (size_t)side_lengths[s];
					errcode = clEnqueueNDRangeKernel(queue, kernel_column_writer, 1, NULL, &global_size, 0, 0, NULL, &ev_kernel);

					end_bench(s, 4, ev_kernel);
				}

				/// 6. library-based transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);

					cl_event cl_ev_transfer = cl_rul::upload_rect<cl_float>(queue, device_buffers[s],
					{ (size_t)side_lengths[s], (size_t)side_lengths[s], 1u }, { {0u,0u,0u}, {1u,(size_t)side_lengths[s],1u} }, (cl_float*)host_buffers[s]);

					end_bench(s, 5, cl_ev_transfer);
				}
			}
		}

		for(int s = 0; s < NUM_SIZES; ++s) {

			printf("%12d , ", side_lengths[s]);
			for(int i = 0; i < NUM_TYPES; ++i) {
				printf("%12.2lf", results[s][i]);
				printf(i == NUM_TYPES - 1 ? "\n" : " , ");
			}
		}
	}

	{
		printf("\nDOWNLOAD (GPU -> Host) times in microseconds\n");
		for(int i = 0; i < NUM_TYPES + 1; ++i) {
			printf("%12s ", i == 0 ? "Side length" : names[i - 1]);
			printf(i == NUM_TYPES ? "\n" : ", ");
		}

		double results[NUM_SIZES][NUM_TYPES];
		std::fill(&results[0][0], &results[NUM_SIZES][0], 1000000.0f);

		cl_event cl_ev_before; // just for measurement
		auto start_bench = [&](int s) {
			clEnqueueWriteBuffer(queue, device_buffers[s], CL_FALSE, 0, 1, host_buffers[s], 0, NULL, &cl_ev_before);
		};
		auto end_bench = [&](int s, int id, cl_event cl_ev_transfer) {
			clFinish(queue);
			cl_ulong start, end;
			CLU_ERRCHECK(clGetEventProfilingInfo(cl_ev_before, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL), "Error reading start time");
			CLU_ERRCHECK(clGetEventProfilingInfo(cl_ev_transfer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL), "Error reading end time");
			results[s][id] = std::min(results[s][id], ((double)(end - start) / 1000.0));
		};

		for(int r = 0; r < NUM_REPETITIONS; ++r) {
			for(int s = 0; s < NUM_SIZES; ++s) {

				/// 1. complete transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_complete;
					errcode = clEnqueueReadBuffer(queue, device_buffers[s], CL_FALSE, 0, byte_sizes[s], host_buffers[s], 0, NULL, &ev_complete);
					CLU_ERRCHECK(errcode, "Error enqueueing complete transfer");
					end_bench(s, 0, ev_complete);
				}

				/// 2. rect / rect column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_transfer;
					const size_t buffer_origin[3] = { 0, 0, 0 };
					const size_t host_origin[3] = { 0, 0, 0 };
					const size_t region[3] = { sizeof(float), (size_t)side_lengths[s], 1 };
					size_t buffer_row_pitch = side_lengths[s] * sizeof(float);
					size_t buffer_slice_pitch = byte_sizes[s];
					size_t host_row_pitch = side_lengths[s] * sizeof(float);
					size_t host_slice_pitch = byte_sizes[s];
					errcode = clEnqueueReadBufferRect(queue, device_buffers[s], CL_FALSE,
						buffer_origin, host_origin, region,
						buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
						host_buffers[s], 0, NULL, &ev_transfer);
					if(errcode == CL_SUCCESS) {
						end_bench(s, 1, ev_transfer);
					}
				}

				/// 3. rect / linear column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);
					cl_event ev_transfer = cl_rul::download_rect<cl_float, cl_rul::ClRect>(queue, device_buffers[s],
						{ (size_t)side_lengths[s], (size_t)side_lengths[s], 1u }, { { 0u,0u,0u },{ 1u,(size_t)side_lengths[s],1u } }, (cl_float*)host_buffers[s]);
					end_bench(s, 2, ev_transfer);
				}

				/// 4. individual column transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					// only test this on smaller sizes, kills some implementations
					if(side_lengths[s] <= 512) {
						start_bench(s);
						cl_event ev_transfer = cl_rul::download_rect<cl_float, cl_rul::Individual>(queue, device_buffers[s],
							{ (size_t)side_lengths[s], (size_t)side_lengths[s], 1u }, { { 0u,0u,0u },{ 1u,(size_t)side_lengths[s],1u } }, (cl_float*)host_buffers[s]);
						end_bench(s, 3, ev_transfer);
					}
				}

				/// 5. kernel-based transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);

					cl_event ev_kernel;
					int zero = 0;
					cluSetKernelArguments(kernel_column_reader, 4, sizeof(cl_mem), (void *)&device_buffers[s], sizeof(cl_mem), (void *)&staging_buffer, sizeof(cl_int), &zero, sizeof(cl_int), &side_lengths[s]);
					size_t global_size = (size_t)side_lengths[s];
					errcode = clEnqueueNDRangeKernel(queue, kernel_column_reader, 1, NULL, &global_size, 0, 0, NULL, &ev_kernel);

					// transfer linearly from staging buffer
					cl_event ev_staging;
					errcode = clEnqueueReadBuffer(queue, staging_buffer, CL_FALSE, side_lengths[s] * sizeof(float), sizeof(cl_float), host_buffers[s], 0, NULL, &ev_staging);
					CLU_ERRCHECK(errcode, "Error enqueueing staging transfer");

					end_bench(s, 4, ev_staging);
				}

				/// 6. library-based transfer
				for(int ri = 0; ri < NUM_INNER_REPETITIONS; ++ri) {
					start_bench(s);

					cl_event cl_ev_transfer = cl_rul::download_rect<cl_float>(queue, device_buffers[s],
						{ (size_t)side_lengths[s], (size_t)side_lengths[s], 1u }, { {0u,0u,0u}, {1u,(size_t)side_lengths[s],1u} }, (cl_float*)host_buffers[s]);

					end_bench(s, 5, cl_ev_transfer);
				}
			}
		}

		for(int s = 0; s < NUM_SIZES; ++s) {

			printf("%12d , ", side_lengths[s]);
			for(int i = 0; i < NUM_TYPES; ++i) {
				printf("%12.2lf", results[s][i]);
				printf(i == NUM_TYPES - 1 ? "\n" : " , ");
			}
		}
	}
}
