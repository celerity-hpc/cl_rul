#include "../ext/catch.hpp"

#include "global_cl.h"
#include "test_utils.h"

/// /////////////////////////////////////////////////////////////////////// Float

template<typename Method, size_t TEST_L>
void partial_1D_float_upload_test(cl_command_queue queue, cl_mem device_buffer, cl_float* host_buffer2) {
	cl_float to_upload[4] = { 42.f, 44.f, 46.f, 48.f };
	cl_rul::upload_rect<cl_float, Method>(queue, device_buffer, { TEST_L,1u,1u }, { { 3u,0u,0u },{ 4u,1u,1u } }, to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * sizeof(float), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);
	cl_float result[TEST_L] = { 0.f, 1.f, 2.f, 42.f, 44.f, 46.f, 48.f, 7.f, 8.f, 9.f };
	check_1D(result, host_buffer2, TEST_L);
}

template<typename Method, size_t TEST_L>
void partial_1D_float_download_test(cl_command_queue queue, cl_mem device_buffer, cl_float* host_buffer2) {
	const size_t DOWN_L = 3;
	cl_rul::download_rect<cl_float, cl_rul::Individual>(queue, device_buffer, { TEST_L,1u,1u }, { { 5u,0u,0u },{ DOWN_L,1u,1u } }, host_buffer2);
	clFinish(queue);
	cl_float result[DOWN_L] = { 5.f, 6.f, 7.f };
	check_1D(result, host_buffer2, DOWN_L);
}

TEST_CASE("1D float buffers", "[1D]") {

	constexpr size_t TEST_L = 10;

	cl_float host_buffer[TEST_L];
	cl_float host_buffer2[TEST_L];

	for(int i = 0; i < TEST_L; ++i) {
		host_buffer[i] = (cl_float)i;
		host_buffer2[i] = (cl_float)-i;
	}

	cl_int errcode;
	cl_mem device_buffer = clCreateBuffer(GlobalCl::context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TEST_L * sizeof(cl_float), host_buffer, &errcode);
	REQUIRE(errcode == CL_SUCCESS);

	SECTION("partial upload [individual]") {
		partial_1D_float_upload_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [kernel]") {
		partial_1D_float_upload_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [automatic]") {
		partial_1D_float_upload_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial download [individual]") {
		partial_1D_float_download_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [kernel]") {
		partial_1D_float_download_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [automatic]") {
		partial_1D_float_download_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
}

/// /////////////////////////////////////////////////////////////////////// Char4

template<typename Method, size_t TEST_L>
void partial_1D_char4_upload_test(cl_command_queue queue, cl_mem device_buffer, cl_char4* host_buffer2) {
	cl_char4 to_upload[2] = { { 42,42,40,40 },{ 17,18,19,20 } };
	cl_rul::upload_rect<cl_char4, Method>(queue, device_buffer, { TEST_L,1u,1u }, { { 1u,0u,0u },{ 2u,1u,1u } }, to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * sizeof(float), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);
	cl_char4 result[TEST_L] = { { 0,0,0,0 },{ 42,42,40,40 },{ 17,18,19,20 },{ 3,3,3,3 },{ 4,4,4,4 } };
	check_1D(result, host_buffer2, TEST_L);
}

template<typename Method, size_t TEST_L>
void partial_1D_char4_download_test(cl_command_queue queue, cl_mem device_buffer, cl_char4* host_buffer2) {
	const size_t DOWN_L = 2;
	cl_rul::download_rect<cl_char4, Method>(queue, device_buffer, { TEST_L,1u,1u }, { { 3u,0u,0u },{ DOWN_L,1u,1u } }, host_buffer2);
	clFinish(queue);
	cl_char4 result[DOWN_L] = { { 3,3,3,3 }, { 4,4,4,4 } };
	check_1D(result, host_buffer2, DOWN_L);
}

TEST_CASE("1D char4 buffers", "[1D]") {

	constexpr size_t TEST_L = 5;

	cl_char4 host_buffer[TEST_L];
	cl_char4 host_buffer2[TEST_L];

	for(cl_char i = 0; i < TEST_L; ++i) {
		cl_char4 c = { i,i,i,i };
		host_buffer[i] = c;
		cl_char4 nc = { (cl_char)-i, (cl_char)-i, (cl_char)-i, (cl_char)-i };
		host_buffer2[i] = nc;
	}

	cl_int errcode;
	cl_mem device_buffer = clCreateBuffer(GlobalCl::context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TEST_L * sizeof(cl_char4), host_buffer, &errcode);
	REQUIRE(errcode == CL_SUCCESS);

	SECTION("partial upload [individual]") {
		partial_1D_char4_upload_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial upload [kernel]") {
		partial_1D_char4_upload_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial upload [automatic]") {
		partial_1D_char4_upload_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial download [individual]") {
		partial_1D_char4_download_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [kernel]") {
		partial_1D_char4_download_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [automatic]") {
		partial_1D_char4_download_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
}
