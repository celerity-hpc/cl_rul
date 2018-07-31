#pragma once

// On Windows, CL_RUL_GLOBAL_STORAGE can be used to enable sharing of the
// global context across DLLs, by setting __declspec(dllexport / dllimport).
#ifndef CL_RUL_GLOBAL_STORAGE
#define CL_RUL_GLOBAL_STORAGE
#endif

#include "../ext/cl_utils.h"

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

		class cl_rul_context {
		public:
			void initialize(cl_context ctx, cl_device_id device) {
				assert(cl_ctx == nullptr && cl_device == nullptr && "cl_rect_update_lib - already initialized.");
				cl_ctx = ctx;
				cl_device = device;
			}

			void reset() {
				cl_ctx = nullptr;
				cl_device = nullptr;
				if(staging_buffer != nullptr) {
					clReleaseMemObject(staging_buffer);
					staging_buffer = nullptr;
					staging_buffer_size = 0;
				}

				// FIXME: This doesn't reset user provided types!
				#define BUF_TYPE(_htype, _dtype) reset_kernel<_htype>();
				#include "buffer_types.inc"
				#undef BUF_TYPE
			}

			cl_context get_cl_context() const {
				assert(cl_ctx && "cl_rect_update_lib - request global CL context before setting it -- did you call init_rect_update_lib?");
				return cl_ctx;
			}
			cl_device_id get_cl_device_id() const {
				assert(cl_device && "cl_rect_update_lib - request global CL device before setting it -- did you call init_rect_update_lib?");
				return cl_device;
			}

			cl_mem get_staging_buffer(size_t size_in_bytes) {
				//printf("cl_rect_update_lib - requesting staging buffer of size %u, available: %u at %p\n", (unsigned)size_in_bytes, (unsigned)staging_buffer_size, staging_buffer);
				if(staging_buffer_size < size_in_bytes) {
					if(staging_buffer != nullptr) clReleaseMemObject(staging_buffer);
					cl_int errcode = CL_SUCCESS;
					staging_buffer = clCreateBuffer(get_cl_context(), CL_MEM_READ_WRITE, size_in_bytes, nullptr, &errcode);
					CLU_ERRCHECK(errcode, "cl_rect_update_lib - error allocating staging buffer of size %u", (unsigned)size_in_bytes);
					//printf("cl_rect_update_lib - (re-)allocated staging buffer of size %u at %p\n", (unsigned)size_in_bytes, staging_buffer);
					staging_buffer_size = size_in_bytes;
				}
				return staging_buffer;
			}

			template<typename T>
			cl_program& upload_program_2D();

			template<typename T>
			cl_kernel& upload_kernel_2D();

			template<typename T>
			cl_program& download_program_2D();

			template<typename T>
			cl_kernel& download_kernel_2D();

		private:
			cl_context cl_ctx = nullptr;
			cl_device_id cl_device = nullptr;
			cl_mem staging_buffer = nullptr;
			size_t staging_buffer_size = 0;

			template<typename T>
			void reset_kernel() {
				if(upload_program_2D<T>() != nullptr) clReleaseProgram(upload_program_2D<T>());
				if(upload_kernel_2D<T>() != nullptr) clReleaseKernel(upload_kernel_2D<T>());
				if(download_program_2D<T>() != nullptr) clReleaseProgram(download_program_2D<T>());
				if(download_kernel_2D<T>() != nullptr) clReleaseKernel(download_kernel_2D<T>());
				upload_program_2D<T>() = nullptr;
				upload_kernel_2D<T>() = nullptr;
				download_program_2D<T>() = nullptr;
				download_kernel_2D<T>() = nullptr;
			}
		};

		#define BODY_PROGRAM { static cl_program prog = nullptr; return prog; }
		#define BODY_KERNEL { static cl_kernel kernel = nullptr; return kernel; }

		// Provide default implementation for user-defined types.
		// This means that kernels for user-defined types are local to each translation unit.
		// Unfortunately there currently is no way (afaik) around this.
		template<typename T> cl_program& cl_rul_context::upload_program_2D() BODY_PROGRAM;
		template<typename T> cl_kernel& cl_rul_context::upload_kernel_2D() BODY_KERNEL;
		template<typename T> cl_program& cl_rul_context::download_program_2D() BODY_PROGRAM;
		template<typename T> cl_kernel& cl_rul_context::download_kernel_2D() BODY_KERNEL;

		// Mark all predefined types as external, so the kernels can be shared across translation units.
		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_program& cl_rul_context::upload_program_2D<_htype>();
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_kernel& cl_rul_context::upload_kernel_2D<_htype>();
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_program& cl_rul_context::download_program_2D<_htype>();
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_kernel& cl_rul_context::download_kernel_2D<_htype>();
		#include "buffer_types.inc"
		#undef BUF_TYPE

		extern CL_RUL_GLOBAL_STORAGE cl_rul_context g_context;

#ifdef CL_RUL_IMPL
		cl_rul_context g_context;

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_program& cl_rul_context::upload_program_2D<_htype>() BODY_PROGRAM;
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_kernel& cl_rul_context::upload_kernel_2D<_htype>() BODY_KERNEL;
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_program& cl_rul_context::download_program_2D<_htype>() BODY_PROGRAM;
		#include "buffer_types.inc"
		#undef BUF_TYPE

		#define BUF_TYPE(_htype, _dtype) template<> CL_RUL_GLOBAL_STORAGE cl_kernel& cl_rul_context::download_kernel_2D<_htype>() BODY_KERNEL;
		#include "buffer_types.inc"
		#undef BUF_TYPE

#endif

		#undef BODY_PROGRAM
		#undef BODY_KERNEL

		constexpr const char* UPLOAD_2D_KERNEL_NAME = "upload_2D";
		constexpr const char* DOWNLOAD_2D_KERNEL_NAME = "download_2D";

		inline void check_global_state_validity(cl_command_queue queue) {
			cl_context local_ctx;
			clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &local_ctx, nullptr);
			assert(local_ctx == g_context.get_cl_context());
			cl_device_id local_dev;
			clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &local_dev, nullptr);
			assert(local_dev == g_context.get_cl_device_id());
		}

		struct type_info {
			std::string name;
			int num;
		};

		template<typename T>
		const type_info get_type_info() {
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
		void build_transfer_kernel(const char* source, const char* kernel_name, cl_program& out_prog, cl_kernel& out_kernel) {
			auto ti = get_type_info<T>();
			std::stringstream ss;
			ss << "-D T=" << ti.name << " " << "-D NUM=" << ti.num << std::flush;
			std::string options = ss.str();
			//printf("options: \"%s\"\n", options.c_str());
			out_prog = cluBuildProgramFromString(g_context.get_cl_context(), g_context.get_cl_device_id(), source, options.c_str());
			cl_int errcode = CL_SUCCESS;
			out_kernel = clCreateKernel(out_prog, kernel_name, &errcode);
			CLU_ERRCHECK(errcode, "cl_rect_update_lib - kernel loading error for options: %s", options.c_str());
		}

		template<typename T>
		cl_kernel get_upload_kernel_2D() {
			if(!g_context.upload_kernel_2D<T>()) {
				build_transfer_kernel<T>(kernels::upload_2D, UPLOAD_2D_KERNEL_NAME, g_context.upload_program_2D<T>(), g_context.upload_kernel_2D<T>());
			}
			return g_context.upload_kernel_2D<T>();
		}

		template<typename T>
		cl_kernel get_download_kernel_2D() {
			if(!g_context.download_kernel_2D<T>()) {
				build_transfer_kernel<T>(kernels::download_2D, DOWNLOAD_2D_KERNEL_NAME, g_context.download_program_2D<T>(), g_context.download_kernel_2D<T>());
			}
			return g_context.download_kernel_2D<T>();
		}

	} // namespace detail

	/**
	 * @brief Initializes the cl_rect_update library. This must be called before using any of the other methods.
	 *
	 * @param eager If true, transfer kernels for all predefined types will be compiled immediately, instead of when they're first required.
	 */
	inline void init_rect_update_lib(cl_context context, cl_device_id device, bool eager = false) {
		detail::g_context.initialize(context, device);

		if (eager) {
			// TODO make pre-compilation configurable? could take some time
			#define BUF_TYPE(_htype, _dtype) detail::get_upload_kernel_2D<_htype>();
			#include "buffer_types.inc"
			#undef BUF_TYPE

			#define BUF_TYPE(_htype, _dtype) detail::get_download_kernel_2D<_htype>();
			#include "buffer_types.inc"
			#undef BUF_TYPE
		}
	}

	inline void reset_rect_update_lib() {
		detail::g_context.reset();
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
			cl_mem staging_buffer = g_context.get_staging_buffer(required_staging_size);
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

				const Extent& s = target_buffer_size;
				const Point& o = target_box.origin;
				const Extent& e = target_box.extent;

				// if 1D, just use simple transfer
				if(e.ys == 1 && e.zs == 1) {
					const size_t linear_offset = o.z * s.xs * s.ys + o.y * s.xs + o.x;
					cl_int errcode = clEnqueueWriteBuffer(queue, target_buffer, CL_FALSE, linear_offset * sizeof(T), e.xs * sizeof(T), linearized_host_data_source, 0, NULL, &ev_ret);
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
			cl_mem staging_buffer = g_context.get_staging_buffer(required_staging_size);

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

				const Extent& s = source_buffer_size;
				const Point& o = source_box.origin;
				const Extent& e = source_box.extent;

				// if 1D, just use simple transfer
				if(e.ys == 1 && e.zs == 1) {
					const size_t linear_offset = o.z * s.xs * s.ys + o.y * s.xs + o.x;
					cl_int errcode = clEnqueueReadBuffer(queue, source_buffer, CL_FALSE, linear_offset * sizeof(T), e.xs * sizeof(T), linearized_host_data_target, 0, NULL, &ev_ret);
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
#ifndef NDEBUG
		detail::check_global_state_validity(queue);
#endif
		return detail::rect_downloader<T, Method>{}(queue, source_buffer, source_buffer_size, source_box, linearized_host_data_target);
	}

} // namespace cl_rul

