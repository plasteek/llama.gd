#ifndef LLAMA_GD_THROW
#define LLAMA_GD_THROW

#include <stdexcept>

namespace godot
{
   void gd_throw_err(std::runtime_error err);
}

#endif