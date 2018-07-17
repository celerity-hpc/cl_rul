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

	struct Box {
		Point origin;
		Extent extent;

		size_t size() const {
			return extent.size();
		};
	};

	/// Globals & Initialization ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {

// On Windows, CL_RUL_GLOBAL_STORAGE can be used to enable sharing of the
// global context across DLLs, by setting __declspec(dllexport / dllimport).
#ifndef CL_RUL_GLOBAL_STORAGE
#define CL_RUL_GLOBAL_STORAGE
#endif

		extern CL_RUL_GLOBAL_STORAGE cl_context g_context;
		extern CL_RUL_GLOBAL_STORAGE cl_device_id g_device;

#ifdef CL_RUL_IMPL
		cl_context g_context;
		cl_device_id g_device;
#endif

		constexpr const char* UPLOAD_2D_KERNEL_NAME = "upload_2D";
		constexpr const char* DOWNLOAD_2D_KERNEL_NAME = "download_2D";

		inline cl_context get_global_context(cl_context c_in = nullptr) {
			if(c_in) g_context = c_in;
			assert(g_context && "cl_rect_upate_lib - request global context before setting it -- did you call init_rect_update_lib?");
			return g_context;
		}

		inline cl_device_id get_global_device(cl_device_id dev_in = nullptr) {
			if(dev_in) g_device = dev_in;
			assert(g_device && "cl_rect_upate_lib - request global device before setting it -- did you call init_rect_update_lib?");
			return g_device;
		}

		inline void check_global_state_validity(cl_command_queue queue) {
			cl_context local_ctx;
			clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &local_ctx, nullptr);
			assert(local_ctx == get_global_context());
			cl_device_id local_dev;
			clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &local_dev, nullptr);
			assert(local_dev == get_global_device());
		}

		/// Kernel management

		template<typename T>
		cl_program g_upload_program_2D;
		template<typename T>
		cl_kernel g_upload_kernel_2D;
		template<typename T>
		cl_program g_download_program_2D;
		template<typename T>
		cl_kernel g_download_kernel_2D;

		struct type_info {
			std::string name;
			int num;
		};

		template<typename T>
		inline const type_info get_type_info() {
			int s = sizeof(T);
			if(s % 16 == 0) return { "float4", s / 16 };
			if(s %  8 == 0) return { "float2", s /  8 };
			if(s %  4 == 0) return {  "float", s /  4 };
			return { "char", s };
		}

		#define BUF_TYPE(_htype, _dtype) template<> inline const type_info get_type_info<_htype>() { return {#_dtype, 1}; }
		#include "buffer_types.inc"
		#undef BUF_TYPE

		template<typename T>
		inline void build_transfer_kernel(const char* source, const char* kernel_name, cl_program& out_prog, cl_kernel& out_kernel) {
			auto ti = get_type_info<T>();
			std::stringstream ss;
			ss << "-D T=" << ti.name << " " << "-D NUM=" << ti.num << std::flush;
			std::string options = ss.str();
			//printf("options: \"%s\"\n", options.c_str());
			out_prog = cluBuildProgramFromString(get_global_context(), get_global_device(), source, options.c_str());
			cl_int errcode = CL_SUCCESS;
			out_kernel = clCreateKernel(out_prog, kernel_name, &errcode);
			CLU_ERRCHECK(errcode, "cl_rect_update_lib - kernel loading error for options: %s", options.c_str());
		}

		template<typename T>
		inline cl_kernel get_upload_kernel_2D() {
			if(!detail::g_upload_kernel_2D<T>) {
				build_transfer_kernel<T>(kernels::upload_2D, UPLOAD_2D_KERNEL_NAME, detail::g_upload_program_2D<T>, detail::g_upload_kernel_2D<T>);
			}
			return detail::g_upload_kernel_2D<T>;
		}

		template<typename T>
		inline cl_kernel get_download_kernel_2D() {
			if(!detail::g_download_kernel_2D<T>) {
				build_transfer_kernel<T>(kernels::download_2D, DOWNLOAD_2D_KERNEL_NAME, detail::g_download_program_2D<T>, detail::g_download_kernel_2D<T>);
			}
			return detail::g_download_kernel_2D<T>;
		}

		/// Buffer management

		inline cl_mem get_staging_buffer(size_t size_in_bytes) {
			thread_local cl_mem buff = nullptr;
			thread_local size_t allocated_size = 0;
			//printf("cl_rect_update_lib - requesting staging buffer of size %u, available: %u at %p\n", (unsigned)size_in_bytes, (unsigned)allocated_size, buff);
			if(allocated_size < size_in_bytes) {
				if(buff) clReleaseMemObject(buff);
				cl_int errcode = CL_SUCCESS;
				buff = clCreateBuffer(get_global_context(), CL_MEM_READ_WRITE, size_in_bytes, nullptr, &errcode);
				CLU_ERRCHECK(errcode, "cl_rect_update_lib - error allocating staging buffer of size %u", (unsigned)size_in_bytes);
				//printf("cl_rect_update_lib - (re-)allocated staging buffer of size %u at %p\n", (unsigned)size_in_bytes, buff);
				allocated_size = size_in_bytes;
			}
			return buff;
		}
	}

	inline void init_rect_update_lib(cl_context context, cl_device_id device) {
		// *set* the globals
		detail::get_global_context(context);
		detail::get_global_device(device);

		// TODO make pre-compilation configurable? could take some time
		#define BUF_TYPE(_htype, _dtype) detail::get_upload_kernel_2D<_htype>();
		#include "buffer_types.inc"
		#undef BUF_TYPE
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
				cl_event ev_ret = nullptr;

				const Point& o = target_box.origin;
				const Extent& e = target_box.extent;
				const Extent& full_e = target_buffer_size;

				const T* source_ptr = linearized_host_data_source;

				size_t zend = o.z + e.zs;
				size_t yend = o.y + e.ys;
				for(size_t z = o.z; z < zend; z++) {
					for(size_t y = o.y; y < yend; y++) {
						bool last = z == zend - 1 && y == yend - 1;
						size_t offset = full_e.row_offset(y, z) + o.x;
						//printf("cl_rect_update_lib - individual upload offset: %8u ; range: %8u\n", (unsigned)(offset * sizeof(T)), (unsigned)(e.xs * sizeof(T)));
						cl_int errcode = clEnqueueWriteBuffer(queue, target_buffer, CL_FALSE, offset * sizeof(T), e.xs * sizeof(T), source_ptr, 0, NULL, last ? &ev_ret : NULL);
						CLU_ERRCHECK(errcode, "cl_rect_update_lib - upload_rect: error enqueueing individual transfer");
						source_ptr += e.xs;
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

				const size_t buffer_origin[3] = { o.x * sizeof(T), o.y, o.z };
				const size_t host_origin[3] = { 0, 0, 0 };
				const size_t region[3] = { e.xs * sizeof(T), e.ys, e.zs };
				size_t buffer_row_pitch = full_e.xs * sizeof(T);
				size_t buffer_slice_pitch = full_e.slice_size() * sizeof(T);
				size_t host_row_pitch = e.xs * sizeof(T);
				size_t host_slice_pitch = e.slice_size() * sizeof(T);
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

				// if 1D, just use simple transfer
				if(e.ys == 1 && e.zs == 1) {
					cl_int errcode = clEnqueueWriteBuffer(queue, target_buffer, CL_FALSE, o.x * sizeof(T), e.xs * sizeof(T), linearized_host_data_source, 0, NULL, &ev_ret);
					CLU_ERRCHECK(errcode, "cl_rect_upate_lib - upload_rect: error enqueueing transfer");
					return ev_ret;
				}

				// if 2D or 3D use linearized transfer and specialized kernel
				if(e.zs == 1) return upload_rect_kernel_2D<T>(queue, target_buffer, target_buffer_size, target_box, linearized_host_data_source);

				assert(false && "cl_rect_upate_lib - 3D kernel transfer not implemented yet");
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


	/// Download functions ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {
		template<typename T, typename Method = Automatic>
		struct rect_downloader {
			cl_event operator()(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target);
		};

		template<typename T>
		struct rect_downloader<T, Individual> {
			cl_event operator()(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {
				cl_event ev_ret = nullptr;

				const Point& o = source_box.origin;
				const Extent& e = source_box.extent;
				const Extent& full_e = source_buffer_size;

				T* trg_ptr = linearized_host_data_target;

				size_t zend = o.z + e.zs;
				size_t yend = o.y + e.ys;
				for(size_t z = o.z; z < zend; z++) {
					for(size_t y = o.y; y < yend; y++) {
						bool last = z == zend - 1 && y == yend - 1;
						size_t offset = full_e.row_offset(y, z) + o.x;
						//printf("cl_rect_update_lib - individual download  offset: %8u ; range: %8u\n", (unsigned)(offset * sizeof(T)), (unsigned)(e.xs * sizeof(T)));
						cl_int errcode = clEnqueueReadBuffer(queue, source_buffer, CL_FALSE, offset * sizeof(T), e.xs * sizeof(T), trg_ptr, 0, NULL, last ? &ev_ret : NULL);
						CLU_ERRCHECK(errcode, "cl_rect_update_lib - download_rect: error enqueueing individual transfer");
						trg_ptr += e.xs;
					}
				}

				return ev_ret;
			}
		};

		template<typename T>
		struct rect_downloader<T, ClRect> {
			cl_event operator()(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {
				cl_event ev_ret;

				const Point& o = source_box.origin;
				const Extent& e = source_box.extent;
				const Extent& full_e = source_buffer_size;

				const size_t buffer_origin[3] = { o.x * sizeof(T), o.y, o.z };
				const size_t host_origin[3] = { 0, 0, 0 };
				const size_t region[3] = { e.xs * sizeof(T), e.ys, e.zs };
				size_t buffer_row_pitch = full_e.xs * sizeof(T);
				size_t buffer_slice_pitch = full_e.slice_size() * sizeof(T);
				size_t host_row_pitch = e.xs * sizeof(T);
				size_t host_slice_pitch = e.slice_size() * sizeof(T);
				cl_int errcode = clEnqueueReadBufferRect(queue, source_buffer, CL_FALSE,
					buffer_origin, host_origin, region,
					buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
					linearized_host_data_target, 0, NULL, &ev_ret);
				CLU_ERRCHECK(errcode, "cl_rect_upate_lib - download_rect: error enqueueing clrect transfer");

				return ev_ret;
			}
		};

		template<typename T>
		cl_event download_rect_kernel_2D(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {

			const Point& o = source_box.origin;
			const Extent& e = source_box.extent;
			const Extent& full_e = source_buffer_size;

			// get staging buffer

			size_t required_staging_size = e.size() * sizeof(T);
			cl_mem staging_buffer = get_staging_buffer(required_staging_size);

			// use kernel to write to staging buffer
			// parameters:
			//		__global v_t *src, __global v_t *trg,
			//		uint pos_x, uint pos_y,
			//		uint size_x, uint size_y,
			//		uint stride

			cl_kernel kernel = get_download_kernel_2D<T>();

			cl_event ev_kernel;
			cl_uint pos_x = static_cast<cl_uint>(o.x), pos_y = static_cast<cl_uint>(o.y);
			cl_uint size_x = static_cast<cl_uint>(e.xs), size_y = static_cast<cl_uint>(e.ys);
			cl_uint stride = static_cast<cl_uint>(full_e.xs);
			cluSetKernelArguments(kernel, 7,
				sizeof(cl_mem), &source_buffer, sizeof(cl_mem), &staging_buffer,
				sizeof(cl_uint), &pos_x, sizeof(cl_uint), &pos_y,
				sizeof(cl_uint), &size_x, sizeof(cl_uint), &size_y,
				sizeof(cl_uint), &stride);
			size_t global_size = e.size();
			cl_int errcode = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 0, 0, NULL, &ev_kernel);
			CLU_ERRCHECK(errcode, "cl_rect_upate_lib - error enqueueing download kernel");

			// transfer from staging buffer to host

			cl_event ev_staging;
			errcode = clEnqueueReadBuffer(queue, staging_buffer, CL_FALSE, 0, required_staging_size, linearized_host_data_target, 0, NULL, &ev_staging);
			CLU_ERRCHECK(errcode, "cl_rect_upate_lib - error enqueueing staging transfer");

			return ev_staging;
		}

		template<typename T>
		struct rect_downloader<T, Kernel> {
			cl_event operator()(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {
				cl_event ev_ret = nullptr;

				const Point& o = source_box.origin;
				const Extent& e = source_box.extent;

				// if 1D, just use simple transfer
				if(e.ys == 1 && e.zs == 1) {
					cl_int errcode = clEnqueueReadBuffer(queue, source_buffer, CL_FALSE, o.x * sizeof(T), e.xs * sizeof(T), linearized_host_data_target, 0, NULL, &ev_ret);
					CLU_ERRCHECK(errcode, "cl_rect_update_lib - download_rect: error enqueueing transfer");
					return ev_ret;
				}

				// if 2D or 3D use linearized transfer and specialized kernel
				if(e.zs == 1) return download_rect_kernel_2D<T>(queue, source_buffer, source_buffer_size, source_box, linearized_host_data_target);

				assert(false && "cl_rect_upate_lib - 3D kernel transfer not implemented yet");
				return ev_ret;
			}
		};

		template<typename T>
		struct rect_downloader<T, Automatic> {
			cl_event operator()(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {
				// try being smarter later
				return rect_downloader<T, Kernel>()(queue, source_buffer, source_buffer_size, source_box, linearized_host_data_target);
			}
		};
	}

	template<typename T, typename Method = Automatic>
	cl_event download_rect(cl_command_queue queue, cl_mem source_buffer, const Extent& source_buffer_size, const Box& source_box, T *linearized_host_data_target) {
		return detail::rect_downloader<T, Method>{}(queue, source_buffer, source_buffer_size, source_box, linearized_host_data_target);
	}

} // namespace cl_rul
