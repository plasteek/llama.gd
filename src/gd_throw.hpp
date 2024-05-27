#ifndef LLAMA_GD_UTILS
#define LLAMA_GD_UTILS

#include <stdexcept>

namespace godot
{
   void gd_throw_runtime_err(std::runtime_error err);
}

#endif