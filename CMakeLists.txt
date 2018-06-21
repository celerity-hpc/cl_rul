cmake_minimum_required(VERSION 3.1)

project(cl_rul LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS /MP")
endif()

# OpenCL

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Library

add_library(cl_rect_update_lib INTERFACE)
target_include_directories(cl_rect_update_lib INTERFACE cl_rect_update_lib/)
target_link_libraries(cl_rect_update_lib INTERFACE ${OpenCL_LIBRARY})

if(MSVC) # Generate a project for editing the header-only interface library in the IDE  
  file(GLOB LIB_HEADER_FILES cl_rect_update_lib/*.h)
  add_custom_target(cl_rect_update_lib.headers SOURCES ${LIB_HEADER_FILES})
endif()

# Unit tests

file(GLOB TEST_SRC_FILES cl_rect_update_test/*.cpp cl_rect_update_test/*.h)
add_executable(cl_rect_update_test cl_rect_update_test/cl_rect_update_test.cc ${TEST_SRC_FILES})
target_link_libraries(cl_rect_update_test cl_rect_update_lib)

add_test(cl_rect_update_test ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cl_rect_update_test)

# Benchmark

file(GLOB BENCH_SRC_FILES cl_rect_update_benchmark/*.cpp cl_rect_update_benchmark/*.h)
add_executable(cl_rect_update_benchmark ${BENCH_SRC_FILES})
target_link_libraries(cl_rect_update_benchmark cl_rect_update_lib)

add_custom_command(TARGET cl_rect_update_benchmark
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${PROJECT_SOURCE_DIR}/cl_rect_update_benchmark/cl_rect_update_benchmark.cl
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
