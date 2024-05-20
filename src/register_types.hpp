#ifndef REGISTER_LIB_TYPES
#define REGISTER_LIB_TYPES

#include <godot_cpp/core/class_db.hpp>

namespace godot
{

   void initialize_llm_module(ModuleInitializationLevel p_level);
   void cleanup_llm_module(ModuleInitializationLevel p_level);
}

#endif
