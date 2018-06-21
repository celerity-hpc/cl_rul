#include "../ext/catch.hpp"

#include "global_cl.h"
#include "test_utils.h"

#include <iostream>

/// /////////////////////////////////////////////////////////////////////// Float

template<typename Method, size_t TEST_L>
void partial_2D_float_upload_test(cl_command_queue queue, cl_mem device_buffer, cl_float host_buffer2[TEST_L][TEST_L]) {
	cl_float to_upload[3][2] = { {  102.f,  104.f }
							   , { -203.f, -205.f }
							   , {  307.f,  308.f } };

	cl_rul::upload_rect<cl_float, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 1u,2u,0u },{ 2u,3u,1u } }, (cl_float*)to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * TEST_L * sizeof(float), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);

	cl_float result[TEST_L][TEST_L] = { { 0.f,     1.f,    2.f,   3.f,   4.f }
	                                  , { 10.f,   11.f,   12.f,  13.f,  14.f }
	                                  , { 20.f,  102.f,  104.f,  23.f,  24.f }
	                                  , { 30.f, -203.f, -205.f,  33.f,  34.f }
	                                  , { 40.f,  307.f,  308.f,  43.f,  44.f } };

	//print_2D<cl_float, TEST_L>(result, "result");
	//print_2D<cl_float, TEST_L>(host_buffer2, "host_buffer2 changed");
	check_2D<cl_float, TEST_L>(result, host_buffer2);
}

template<typename Method, size_t TEST_L>
void float_column_upload_test(cl_command_queue queue, cl_mem device_buffer, cl_float host_buffer2[TEST_L][TEST_L]) {
	cl_float to_upload[TEST_L] = { 100.f, 101.f, 102.f, 103.f, 104.f };

	cl_rul::upload_rect<cl_float, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 4u,0u,0u },{ 1u,TEST_L,1u } }, (cl_float*)to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * TEST_L * sizeof(float), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);

	cl_float result[TEST_L][TEST_L] = { {  0.f,   1.f,   2.f,   3.f, 100.f }
	                                  , { 10.f,  11.f,  12.f,  13.f, 101.f }
	                                  , { 20.f,  21.f,  22.f,  23.f, 102.f }
	                                  , { 30.f,  31.f,  32.f,  33.f, 103.f }
	                                  , { 40.f,  41.f,  42.f,  43.f, 104.f } };

	//print_2D<cl_float, TEST_L>(host_buffer2, "Result");
	//print_2D<cl_float, TEST_L>(result, "Expected");
	check_2D<cl_float, TEST_L>(result, host_buffer2);
}

template<typename Method, size_t TEST_L>
void partial_2D_float_download_test(cl_command_queue queue, cl_mem device_buffer, cl_float host_buffer2[TEST_L][TEST_L]) {
	cl_rul::download_rect<cl_float, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 2u,1u,0u },{ 2u,2u,1u } }, (cl_float*)host_buffer2);
	clFinish(queue);
	//print_1D((cl_float*)host_buffer2, 4);
	cl_float result[4] = { 12.f, 13.f, 22.f, 23.f };
	check_1D(result, (cl_float*)host_buffer2, 4);
}

template<typename Method, size_t TEST_L>
void float_column_download_test(cl_command_queue queue, cl_mem device_buffer, cl_float host_buffer2[TEST_L][TEST_L]) {
	cl_rul::download_rect<cl_float, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 4u,0u,0u },{ 1u,TEST_L,1u } }, (cl_float*)host_buffer2);
	clFinish(queue);
	//print_1D((cl_float*)host_buffer2, 4);
	cl_float result[TEST_L] = { 4.f, 14.f, 24.f, 34.f, 44.f };
	check_1D(result, (cl_float*)host_buffer2, TEST_L);
}

TEST_CASE("2D float buffers", "[2D]") {

	constexpr size_t TEST_L = 5;

	cl_float host_buffer[TEST_L][TEST_L];
	cl_float host_buffer2[TEST_L][TEST_L];

	for(int i = 0; i < TEST_L; ++i) {
		for(int j = 0; j < TEST_L; ++j) {
			host_buffer[i][j] = (cl_float)(j + 10 * i);
			host_buffer2[i][j] = (cl_float)-(j + 10 * i);
		}
	}

	cl_int errcode;
	cl_mem device_buffer = clCreateBuffer(GlobalCl::context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TEST_L * TEST_L * sizeof(cl_float), host_buffer, &errcode);
	REQUIRE(errcode == CL_SUCCESS);

	SECTION("partial upload [individual]") {
		partial_2D_float_upload_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [rect]") {
		partial_2D_float_upload_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [kernel]") {
		partial_2D_float_upload_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [automatic]") {
		partial_2D_float_upload_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("column upload [individual]") {
		float_column_upload_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column upload [rect]") {
		float_column_upload_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column upload [kernel]") {
		float_column_upload_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column upload [automatic]") {
		float_column_upload_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial download [individual]") {
		partial_2D_float_download_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [rect]") {
		partial_2D_float_download_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [kernel]") {
		partial_2D_float_download_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [automatic]") {
		partial_2D_float_download_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("column download [individual]") {
		float_column_download_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column download [rect]") {
		float_column_download_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column download [kernel]") {
		float_column_download_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("column download [automatic]") {
		float_column_download_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
}

/// /////////////////////////////////////////////////////////////////////// Custom type

struct CustomType {
	cl_float2 coord;
	cl_ushort id;
};

inline bool operator==(const CustomType& l, const CustomType& r) {
	return l.coord.x == r.coord.x && l.coord.y == r.coord.y && l.id == r.id;
}

std::ostream& operator<<(std::ostream& o, const CustomType& ct) {
	char buf[32];
	sprintf(buf, "{ { %4.0f.f, %4.0f.f }, %4uu }", ct.coord.x, ct.coord.y, ct.id);
	return o << buf;
}

template<size_t TEST_L>
void print_custom_matrix(CustomType host_buffer[TEST_L][TEST_L], const char* name = nullptr) {
	if(name) printf("%s:\n", name);
	for(size_t i = 0; i < TEST_L; ++i) {
		for(size_t j = 0; j < TEST_L; ++j) {
			std::cout << host_buffer[i][j] << ", ";
		}
		printf("\n");
	}
}

template<typename Method, size_t TEST_L>
void partial_2D_custom_upload_test(cl_command_queue queue, cl_mem device_buffer, CustomType host_buffer2[TEST_L][TEST_L]) {
	CustomType to_upload[2][2] = { { { { 100.f, 101.f }, 666u }, { { 110.f, 111.f }, 667u } }
								 , { { { 200.f, 201.f }, 777u }, { { 210.f, 211.f }, 778u } } };

	cl_rul::upload_rect<CustomType, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 1u,1u,0u },{ 2u,2u,1u } }, (CustomType*)to_upload);

	REQUIRE(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, TEST_L * TEST_L * sizeof(CustomType), host_buffer2, 0, nullptr, nullptr) == CL_SUCCESS);

	CustomType result[TEST_L][TEST_L] = { { { {   1.f,   1.f },   1u }, { {   2.f,   2.f },   2u }, { {   3.f,   3.f },   3u }, { {   4.f,   4.f },   4u } }
	                                    , { { {  11.f,  11.f },  11u }, { { 100.f, 101.f }, 666u }, { { 110.f, 111.f }, 667u }, { {  14.f,  14.f },  14u } }
	                                    , { { {  21.f,  21.f },  21u }, { { 200.f, 201.f }, 777u }, { { 210.f, 211.f }, 778u }, { {  24.f,  24.f },  24u } }
	                                    , { { {  31.f,  31.f },  31u }, { {  32.f,  32.f },  32u }, { {  33.f,  33.f },  33u }, { {  34.f,  34.f },  34u } } };

	//print_custom_matrix<TEST_L>(result, "result");
	//print_custom_matrix<TEST_L>(host_buffer2, "host_buffer2 changed");
	check_2D<CustomType, TEST_L>(result, host_buffer2);
}

template<typename Method, size_t TEST_L>
void partial_2D_custom_download_test(cl_command_queue queue, cl_mem device_buffer, CustomType host_buffer2[TEST_L][TEST_L]) {

	cl_rul::download_rect<CustomType, Method>(queue, device_buffer, { TEST_L,TEST_L,1u }, { { 2u,1u,0u },{ 2u,2u,1u } }, (CustomType*)host_buffer2);
	clFinish(queue);

	CustomType result[4] = { { {  13.f,  13.f },  13u }, { {  14.f,  14.f },  14u }
	                       , { {  23.f,  23.f },  23u }, { {  24.f,  24.f },  24u } };

	check_1D<CustomType>(result, (CustomType*)host_buffer2, 4);
}


TEST_CASE("2D custom type buffers", "[2D]") {

	constexpr size_t TEST_L = 4;

	CustomType host_buffer[TEST_L][TEST_L] = { { { {   1.f,   1.f },   1u }, { {   2.f,   2.f },   2u } , { {   3.f,   3.f },   3u }, { {   4.f,   4.f },   4u } }
											 , { { {  11.f,  11.f },  11u }, { {  12.f,  12.f },  12u } , { {  13.f,  13.f },  13u }, { {  14.f,  14.f },  14u } }
											 , { { {  21.f,  21.f },  21u }, { {  22.f,  22.f },  22u } , { {  23.f,  23.f },  23u }, { {  24.f,  24.f },  24u } }
											 , { { {  31.f,  31.f },  31u }, { {  32.f,  32.f },  32u } , { {  33.f,  33.f },  33u }, { {  34.f,  34.f },  34u } } };

	CustomType host_buffer2[TEST_L][TEST_L];
	memset(host_buffer2, 0, sizeof(CustomType) * TEST_L * TEST_L);

	cl_int errcode;
	cl_mem device_buffer = clCreateBuffer(GlobalCl::context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, TEST_L * TEST_L * sizeof(CustomType), host_buffer, &errcode);
	REQUIRE(errcode == CL_SUCCESS);

	SECTION("partial upload [individual]") {
		partial_2D_custom_upload_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [clrect]") {
		partial_2D_custom_upload_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [kernel]") {
		partial_2D_custom_upload_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial upload [automatic]") {
		partial_2D_custom_upload_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}

	SECTION("partial download [individual]") {
		partial_2D_custom_download_test<cl_rul::Individual, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [clrect]") {
		partial_2D_custom_download_test<cl_rul::ClRect, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [kernel]") {
		partial_2D_custom_download_test<cl_rul::Kernel, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
	SECTION("partial download [automatic]") {
		partial_2D_custom_download_test<cl_rul::Automatic, TEST_L>(GlobalCl::queue(), device_buffer, host_buffer2);
	}
}
