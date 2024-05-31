#include "conversion.hpp"
#include "gd_throw.hpp"

#include <stdexcept>

#include <godot_cpp/variant/utility_functions.hpp>

namespace godot
{
   void gd_throw_runtime_err(std::runtime_error err)
   {
      std::string msg(err.what());
      String normalized_msg = string_std_to_gd(msg);
      UtilityFunctions::push_error(normalized_msg);
   }
}