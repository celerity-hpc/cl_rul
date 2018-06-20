#pragma once

#include "../cl_rect_update_lib/cl_rect_update_lib.h"
#include <memory>

class GlobalCl {

	cl_context _context;
	cl_command_queue _queue;
	cl_device_id _dev;

	GlobalCl() {
		cl_device_id _dev = cluInitDevice(0, &_context, &_queue);
		cl_rul::init_rect_update_lib(_context, _dev);
	}

	static std::unique_ptr<GlobalCl> state;

public:
	static GlobalCl& get() {
		if(!state) state.reset(new GlobalCl());
		return *state;
	}

	static cl_context context() { return get()._context; }
	static cl_command_queue queue() { return get()._queue; }
	static cl_device_id dev() { return get()._dev; }
};
