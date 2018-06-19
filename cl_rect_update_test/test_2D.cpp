#include "../ext/catch.hpp"

#include "../cl_rect_update_lib/cl_rect_update_lib.h"

/// /////////////////////////////////////////////////////////////////////// Float

template<typename T, size_t TEST_L>
void check_2D(T a[TEST_L][TEST_L], T b[TEST_L][TEST_L]) {
	for(size_t i = 0; i < TEST_L; ++i) {
		for(size_t j = 0; j < TEST_L; ++j) {
			REQUIRE(a[i][j] == b[i][j]);
		}
	}
}

template<typename T>
void print_1D(T* a, size_t count) {
	for(size_t i = 0; i < count; ++i) {
		printf("%u : %f\n", (unsigned)i, a[i]);
	}
}

template<typename Method, size_t TEST_L>
void partial_2D_float_upload_test(cl_command_queue queue, cl_mem device_buffer, cl_float host_buffer2[TEST_L][TEST_L]) {
	cl_float to_upload[3][2] = { {  42.f,  44.f }
							   , { -43.f, -45.f }
							   , {  44.f, -46.f } };

	cl_rul::upload_rect<cl_float, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 1u,2u,0u },{ 3u,2u,1u } }, (cl_float*)to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * TEST_L * sizeof(float), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);
	cl_float result[TEST_L][TEST_L] = { {  0.f,  1.f,  2.f,  3.f,  4.f }
	                                  , { 10.f, 11.f, 12.f, 13.f, 14.f }
	                                  , { 20.f, 21.f, 22.f, 23.f, 24.f }
	                                  , { 30.f, 31.f, 32.f, 33.f, 34.f }
	                                  , { 40.f, 41.f, 42.f, 43.f, 44.f } };
	check_2D<cl_float, TEST_L>(result, host_buffer2);
}

//template<typename Method, size_t TEST_L>
//void partial_1D_float_download_test(cl_command_queue queue, cl_mem device_buffer, cl_float* host_buffer2) {
//	const size_t DOWN_L = 3;
//	cl_rul::download_rect<cl_float, cl_rul::Individual>(queue, device_buffer, { TEST_L,1u,1u }, { { 5u,0u,0u },{ DOWN_L,1u,1u } }, host_buffer2);
//	clFinish(queue);
//	cl_float result[DOWN_L] = { 5.f, 6.f, 7.f };
//	check_1D(result, host_buffer2, DOWN_L);
//}

TEST_CASE("2D float buffers", "[2D]") {

	cl_context context;
	cl_command_queue queue;
	cl_device_id dev = cluInitDevice(0, &context, &queue);

	cl_rul::init_rect_update_lib(context, dev);

	constexpr size_t TEST_L = 5;

	cl_float host_buffer[TEST_L][TEST_L];
	cl_float host_buffer2[TEST_L][TEST_L];

	for(int i = 0; i < TEST_L; ++i) {
		for(int j = 0; j < TEST_L; ++j) {
			host_buffer[i][j] = (cl_float)(i + 10 * j);
			host_buffer2[i][j] = (cl_float)-(i + 10 * j);
		}
	}

	cl_int errcode;
	cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TEST_L * TEST_L * sizeof(cl_float), host_buffer, &errcode);
	REQUIRE(errcode == CL_SUCCESS);

	SECTION("partial upload [individual]") {
		partial_2D_float_upload_test<cl_rul::Individual, TEST_L>(queue, device_buffer, host_buffer2);
	}
	SECTION("partial upload [kernel]") {
		partial_2D_float_upload_test<cl_rul::Kernel, TEST_L>(queue, device_buffer, host_buffer2);
	}
	SECTION("partial upload [automatic]") {
		partial_2D_float_upload_test<cl_rul::Automatic, TEST_L>(queue, device_buffer, host_buffer2);
	}

	//SECTION("partial download [individual]") {
	//	partial_1D_float_download_test<cl_rul::Individual, TEST_L>(queue, device_buffer, host_buffer2);
	//}
	//SECTION("partial download [kernel]") {
	//	partial_1D_float_download_test<cl_rul::Kernel, TEST_L>(queue, device_buffer, host_buffer2);
	//}
	//SECTION("partial download [automatic]") {
	//	partial_1D_float_download_test<cl_rul::Automatic, TEST_L>(queue, device_buffer, host_buffer2);
	//}
}
