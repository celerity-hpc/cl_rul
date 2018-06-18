#pragma once

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define _CRT_SECURE_NO_WARNINGS

#include <CL/cl.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
#include <alloca.h>
#else
#include <malloc.h>
#endif

#define MAX_KERNEL_SOURCE 1024*1024*4

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

// check __err for ocl success and print message in case of error
#define CLU_ERRCHECK(__err, __message, ...) \
if(__err != CL_SUCCESS) { \
	fprintf(stderr, "OpenCL Assertion failure in %s#%d:\n", __FILE__, __LINE__); \
	fprintf(stderr, "Error code: %s\n", cluErrorString(__err)); \
	fprintf(stderr, __message "\n", ##__VA_ARGS__); \
	exit(-1); \
}


// ------------------------------------------------------------------------------------------------ declarations

// initialize opencl device "num" -- devices are numbered sequentially across all platforms
// if supplied, "command_queue" and "context" are filled with an initialized context and command queue on the device
 cl_device_id cluInitDevice(size_t num, cl_context *out_context, cl_command_queue *out_queue);

// get string with basic information about the ocl device "device" with id "id"
const char* cluGetDeviceDescription(const cl_device_id device, unsigned id);

// loads and builds program from "fn" on the supplied context and device, with the options string "options"
// aborts and reports the build log in case of compiler errors
 cl_program cluBuildProgramFromFile(cl_context context, cl_device_id device_id, const char* fn, const char* options);

// sets "num_arg" arguments for kernel "kernel"
// additional arguments need to follow this order: arg0_size, arg0, arg1_size, arg1, ...
 void cluSetKernelArguments(const cl_kernel kernel, const cl_uint num_args, ...);

// return string representation of ocl error code "err"
 const char* cluErrorString(cl_int err);

// return string representation of ocl device type "type"
const char* cluDeviceTypeString(cl_device_type type);


// ------------------------------------------------------------------------------------------------ implementations

 cl_device_id cluInitDevice(size_t num, cl_context *out_context, cl_command_queue *out_queue) {
	// get platform ids
	cl_uint ret_num_platforms;
	CLU_ERRCHECK(clGetPlatformIDs(0, NULL, &ret_num_platforms), "Failed to query number of ocl platforms");
	cl_platform_id *ret_platforms = (cl_platform_id*)alloca(sizeof(cl_platform_id)*ret_num_platforms);
	CLU_ERRCHECK(clGetPlatformIDs(ret_num_platforms, ret_platforms, NULL), "Failed to retrieve ocl platforms");

	// get device id of desired device
	cl_device_id device_id = NULL;
	for(cl_uint i=0; i<ret_num_platforms; ++i) {
		cl_uint ret_num_devices;
		CLU_ERRCHECK(clGetDeviceIDs(ret_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices), "Failed to query number of ocl devices");
		if(num < ret_num_devices) {
			// desired device is on this platform, select
			cl_device_id *ret_devices = (cl_device_id*)alloca(sizeof(cl_device_id)*ret_num_devices);
			CLU_ERRCHECK(clGetDeviceIDs(ret_platforms[i], CL_DEVICE_TYPE_ALL, ret_num_devices, ret_devices, NULL), "Failed to retrieve ocl devices");
			device_id = ret_devices[num];
		}
		num -= ret_num_devices;
	}

	// create opencl context if requested
	if(out_context != NULL) {
		cl_int err;
		*out_context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
		CLU_ERRCHECK(err, "Failed to create ocl context");

		// create command queue if requested
		if(out_queue != NULL) {
			*out_queue = clCreateCommandQueue(*out_context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
			CLU_ERRCHECK(err, "Failed to create ocl command queue");
		}
	}
	return device_id;
}


 void cluLoadSource(const char* fn, size_t max_len, char* source_buffer) {
	FILE *fp;
	fp = fopen(fn, "r");
	if(!fp) {
		fprintf(stderr, "Failed to load kernel file %s\n", fn);
		exit(-1);
	}
	size_t len = fread(source_buffer, 1, max_len, fp);
	source_buffer[len] = '\0';
	assert(feof(fp) && "Kernel source buffer too small");
	fclose(fp);
}


 cl_program cluBuildProgramFromFile(cl_context context, cl_device_id device_id, const char* fn, const char* options) {
	cl_int err;

	// create kernel programs from source
	char *source_str = (char*)malloc(MAX_KERNEL_SOURCE * sizeof(char));
	cluLoadSource(fn, MAX_KERNEL_SOURCE, source_str);
	const char *sources[1] = { source_str };
	cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
	CLU_ERRCHECK(err, "Failed to create program from source file: %s", fn);

	// build kernel program
	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram() failed for source file: %s\n", fn);
		fprintf(stderr, "Error type: %s\n", cluErrorString(err));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, MAX_KERNEL_SOURCE, source_str, NULL);
		fprintf(stderr, "Build log:\n%s\n", source_str);
		exit(-1);
	}

	free(source_str);
	return program;
}

 cl_program cluBuildProgramFromString(cl_context context, cl_device_id device_id, const char* source_str, const char* options) {
	 cl_int err;

	 // create kernel programs from source
	 const char *sources[1] = { source_str };
	 cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
	 CLU_ERRCHECK(err, "Failed to create program from source code");

	 // build kernel program
	 err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	 if(err != CL_SUCCESS) {
		 fprintf(stderr, "clBuildProgram() failed\n");
		 fprintf(stderr, "Error type: %s\n", cluErrorString(err));
		 char log_str[2048];
		 clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 2048, log_str, NULL);
		 fprintf(stderr, "Build log:\n%s\n", log_str);
		 exit(-1);
	 }

	 return program;
 }


 void cluSetKernelArguments(const cl_kernel kernel, const cl_uint num_args, ...) {
	//loop through the arguments and call clSetKernelArg for each
	size_t arg_size;
	const void *arg_val;
	va_list arg_list;
	va_start(arg_list, num_args);
	for(cl_uint i=0; i<num_args; ++i) {
		arg_size = va_arg(arg_list, size_t);
		arg_val = va_arg(arg_list, void *);
		CLU_ERRCHECK(clSetKernelArg(kernel, i, arg_size, arg_val), "Error setting kernel argument %u", i);
	}
  	va_end(arg_list);
}


 void cluGetDeviceName(const cl_device_id device, const size_t buff_size, char *buffer) {
	CLU_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, buff_size, buffer, NULL), "Error getting \"device name\" info");
}
 void cluGetDeviceVendor(const cl_device_id device, const size_t buff_size, char *buffer) {
	CLU_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, buff_size, buffer, NULL), "Error getting \"device vendor\" info");
}
 cl_device_type cluGetDeviceType(cl_device_id device) {
	cl_device_type retval;
	CLU_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(retval), &retval, NULL), "Error getting \"device type\" info");
	return retval;
}

#define MAX_DEVICES 16
const char* cluGetDeviceDescription(const cl_device_id device, unsigned id) {
	static char descriptions[MAX_DEVICES][128];
	static cl_bool initialized[MAX_DEVICES];
	assert(id<MAX_DEVICES && "Device limit exceeded");
	if(!initialized[id]) {
		char name[255], vendor[255];
		cluGetDeviceName(device, 255, name);
		cluGetDeviceVendor(device, 255, vendor);
		sprintf(descriptions[id], "%32s  |  Vendor: %32s  |  Type: %4s", name, vendor, cluDeviceTypeString(cluGetDeviceType(device)));
	}
	return descriptions[id];
}


 const char* cluDeviceTypeString(cl_device_type type) {
	switch(type){
		case CL_DEVICE_TYPE_CPU: return "CPU";
		case CL_DEVICE_TYPE_GPU: return "GPU";
		case CL_DEVICE_TYPE_ACCELERATOR: return "ACC";
	}
	return "UNKNOWN";
}


 const char* cluErrorString(cl_int err) {
	static char buffer[1024];
	switch(err)
	{
		case CL_SUCCESS: return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
		case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
		case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
		case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
		case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
		case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
		case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
		case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
	}
	sprintf(buffer, "UNKNOWN_ERROR (%ld)", err);
	return buffer;
}
