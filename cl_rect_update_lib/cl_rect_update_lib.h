#pragma once

#include "../ext/cl_utils.h"

#include <algorithm>
#include <string>
#include <sstream>
#include <cassert>

#include "kernel_code.h"

namespace cl_rul {

	struct Point {
		size_t x;
		size_t y;
		size_t z;
	};

	struct Extent {
		size_t xs;
		size_t ys;
		size_t zs;

		Extent(size_t xs, size_t ys, size_t zs) : xs(xs), ys(ys), zs(zs) {};

		size_t size() const {
			return xs*ys*zs;
		}
		size_t slice_size() const {
			return xs*ys;
		}

		size_t slice_offset(size_t z) const {
			return z * slice_size();
		}
		size_t row_offset(size_t y, size_t z) const {
			return slice_offset(z) + y*xs;
		}
	};

	using Box = struct {
		Point origin;
		Extent extent;

		size_t size() const {
			return extent.size();
		};
	};

	/// Globals & Initialization ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {

		constexpr const char* UPLOAD_2D_KERNEL_NAME = "upload_2D";

		cl_context g_context;
		cl_device_id g_device;

		/// Kernel management

		template<typename T>
		cl_program g_upload_program_2D;
		template<typename T>
		cl_kernel g_upload_kernel_2D;

		template<typename T>
		const std::string get_type_string() {
			std::stringstream ss;
			ss << "char[" << sizeof(T) << "]" << std::flush;
			return ss.str();
		}

		#define BTYPE(_htype, _dtype) template<> const std::string get_type_string<_htype>() { return #_dtype; }
		#include "buffer_types.inc"
		#undef BTYPE

		template<typename T>
		cl_kernel get_upload_kernel_2D() {
			if(!detail::g_upload_kernel_2D<T>) {
				std::stringstream ss;
				ss << "-D T=" << get_type_string<T>() << std::flush;
				std::string options = ss.str();
				//printf("options: \"%s\"\n", options.c_str());
				detail::g_upload_program_2D<T> = cluBuildProgramFromString(g_context, g_device, kernels::upload_2D, options.c_str());
				cl_int errcode = CL_SUCCESS;
				detail::g_upload_kernel_2D<T> = clCreateKernel(detail::g_upload_program_2D<T>, UPLOAD_2D_KERNEL_NAME, &errcode);
				CLU_ERRCHECK(errcode, "cl_rect_update_lib - kernel loading error for type: %s", options.c_str());
			}
			return detail::g_upload_kernel_2D<T>;
		}

		/// Buffer management

		cl_mem get_staging_buffer(size_t size_in_bytes) {
			thread_local cl_mem buff = nullptr;
			thread_local size_t allocated_size = 0;
			//printf("cl_rect_update_lib - requesting staging buffer of size %u, available: %u at %p\n", (unsigned)size_in_bytes, (unsigned)allocated_size, buff);
			if(allocated_size < size_in_bytes) {
				if(buff) clReleaseMemObject(buff);
				cl_int errcode = CL_SUCCESS;
				buff = clCreateBuffer(g_context, CL_MEM_READ_WRITE, size_in_bytes, nullptr, &errcode);
				CLU_ERRCHECK(errcode, "cl_rect_update_lib - error allocating staging buffer of size %u", (unsigned)size_in_bytes);
				//printf("cl_rect_update_lib - (re-)allocated staging buffer of size %u\n", (unsigned)size_in_bytes);
			}
			return buff;
		}
	}

	void init_rect_update_lib(cl_context context, cl_device_id device) {
		detail::g_context = context;
		detail::g_device = device;

		// TODO make pre-compilation configurable? could take some time
		#define BTYPE(_htype, _dtype) detail::get_upload_kernel_2D<_htype>();
		#include "buffer_types.inc"
		#undef BTYPE
	}


	/// Upload functions ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Update methods (tag type dispatch)
	class Individual {};
	class ClRect {};
	class Kernel {};
	class Automatic {};
	class Runtime {};

	namespace detail {
		template<typename T, typename Method = Automatic>
		struct rect_uploader {
			cl_event operator()(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source);
		};

		template<typename T>
		struct rect_uploader<T, Individual> {
			cl_event operator()(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {
				cl_event ev_ret;

				size_t s = target_buffer_size.size();
				thread_local vector<T> staging_buffer;
				staging_buffer.(s);

				const Point& o = target_box.origin;
				const Extent& e = target_box.extent;
				const Extent& full_e = target_buffer_size;

				T* source_ptr = linearized_host_data_source;

				size_t zend = o.z + e.zs;
				size_t yend = o.y + e.ys;
				for(size_t z = o.z; z < zend; z++) {
					for(size_t y = o.y; y < yend; y++) {
						bool last = z == zend - 1 && y == yend - 1;
						size_t offset = full_e.row_offset(y, z);
						cl_int errcode = clEnqueueWriteBuffer(queue, target_buffer, CL_FALSE, offset * sizeof(T), e.xs * sizeof(T), source_ptr, 0, NULL, last ? ev_ret : NULL);
						CLU_ERRCHECK(errcode, "cl_rect_update_lib - upload_rect: error enqueueing individual transfer");
					}
				}

				return ev_ret;
			}
		};

		template<typename T>
		struct rect_uploader<T, ClRect> {
			cl_event operator()(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {
				cl_event ev_ret;

				const Point& o = target_box.origin;
				const Extent& e = target_box.extent;
				const Extent& full_e = target_buffer_size;

				const size_t buffer_origin[3] = { o.x, o.y, o.z };
				const size_t host_origin[3] = { 0, 0, 0 };
				const size_t region[3] = { e.x, e.y, e.z };
				size_t buffer_row_pitch = full_e.xs * sizeof(T);
				size_t buffer_slice_pitch = full_e.slice_size() * sizeof(T);
				size_t host_row_pitch = sizeof(T);
				size_t host_slice_pitch = 0;
				cl_int errcode = clEnqueueWriteBufferRect(queue, target_buffer, CL_FALSE,
					buffer_origin, host_origin, region,
					buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
					linearized_host_data_source, 0, NULL, &ev_ret);
				CLU_ERRCHECK(errcode, "cl_rect_upate_lib - upload_rect: error enqueueing clrect transfer");

				return ev_ret;
			}
		};

		template<typename T>
		cl_event upload_rect_kernel_2D(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {

			// transfer linearly to staging buffer

			size_t required_staging_size = target_box.size() * sizeof(T);
			cl_mem staging_buffer = get_staging_buffer(required_staging_size);
			cl_event ev_staging;
			cl_int errcode = clEnqueueWriteBuffer(queue, staging_buffer, CL_FALSE, 0, required_staging_size, linearized_host_data_source, 0, NULL, &ev_staging);
			CLU_ERRCHECK(errcode, "cl_rect_upate_lib - error enqueueing staging transfer");

			const Point& o = target_box.origin;
			const Extent& e = target_box.extent;
			const Extent& full_e = target_buffer_size;

			// use kernel to write to final destination
			// parameters:
			//		__global v_t *src, __global v_t *trg,
			//		uint pos_x, uint pos_y,
			//		uint size_x, uint size_y,
			//		uint stride

			cl_kernel kernel = get_upload_kernel_2D<T>();

			cl_event ev_kernel;
			cl_uint pos_x = static_cast<cl_uint>(o.x), pos_y = static_cast<cl_uint>(o.y);
			cl_uint size_x = static_cast<cl_uint>(e.xs), size_y = static_cast<cl_uint>(e.ys);
			cl_uint stride = static_cast<cl_uint>(full_e.xs);
			cluSetKernelArguments(kernel, 7,
				sizeof(cl_mem), &staging_buffer, sizeof(cl_mem), &target_buffer,
				sizeof(cl_uint), &pos_x, sizeof(cl_uint), &pos_y,
				sizeof(cl_uint), &size_x, sizeof(cl_uint), &size_y,
				sizeof(cl_uint), &stride);
			size_t global_size = target_box.size();
			errcode = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 0, 0, NULL, &ev_kernel);
			CLU_ERRCHECK(errcode, "cl_rect_upate_lib - error enqueueing upload kernel");

			return ev_kernel;
		}

		template<typename T>
		struct rect_uploader<T, Kernel> {
			cl_event operator()(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {
				cl_event ev_ret = nullptr;

				const Point& o = target_box.origin;
				const Extent& e = target_box.extent;
				const Extent& full_e = target_buffer_size;

				// if 1D, just use simple transfer
				if(e.ys == 1 && e.zs == 1) {
					cl_int errcode = clEnqueueWriteBuffer(queue, target_buffer, CL_FALSE, o.x * sizeof(T), e.xs * sizeof(T), linearized_host_data_source, 0, NULL, &ev_ret);
					CLU_ERRCHECK(errcode, "cl_rect_upate_lib - upload_rect: error enqueueing transfer");
					return ev_ret;
				}

				// if 2D or 3D use linearized transfer and specialized kernel
				if(e.zs == 1) return upload_rect_kernel_2D<T>(queue, target_buffer, target_buffer_size, target_box, linearized_host_data_source);

				assert(false && "3D kernel transfer not implemented yet");
				return ev_ret;
			}
		};

		template<typename T>
		struct rect_uploader<T, Automatic> {
			cl_event operator()(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {
				// try being smarter later
				return rect_uploader<T, Kernel>()(queue, target_buffer, target_buffer_size, target_box, linearized_host_data_source);
			}
		};
	}

	template<typename T, typename Method = Automatic>
	cl_event upload_rect(cl_command_queue queue, cl_mem target_buffer, const Extent& target_buffer_size, const Box& target_box, const T *linearized_host_data_source) {
		return detail::rect_uploader<T, Method>{}(queue, target_buffer, target_buffer_size, target_box, linearized_host_data_source);
	}


	template<typename T, typename Method = Automatic>
	cl_event download_rect(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target);

} // namespace cl_rul
