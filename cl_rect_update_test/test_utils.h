#pragma once

#include "../cl_rect_update_lib/cl_rect_update_lib.h"

inline bool operator==(const cl_char4& l, const cl_char4& r) {
	return l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w;
}

template<typename T>
inline void check_1D(T* a, T* b, size_t count) {
	for(size_t i = 0; i < count; ++i) {
		REQUIRE(a[i] == b[i]);
	}
}

template<typename T>
inline void print_1D(T* a, size_t count) {
	for(size_t i = 0; i < count; ++i) {
		printf("%u : %f\n", (unsigned)i, a[i]);
	}
}

template<typename T, size_t TEST_L>
inline void check_2D(T a[TEST_L][TEST_L], T b[TEST_L][TEST_L]) {
	for(size_t i = 0; i < TEST_L; ++i) {
		for(size_t j = 0; j < TEST_L; ++j) {
			REQUIRE(a[i][j] == b[i][j]);
		}
	}
}

template<typename T, size_t TEST_L>
inline void print_2D(T a[TEST_L][TEST_L], const char* name = nullptr) {
	if(name) printf("%s:\n", name);
	for(size_t i = 0; i < TEST_L; ++i) {
		for(size_t j = 0; j < TEST_L; ++j) {
			printf(" %6.f.f,", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

