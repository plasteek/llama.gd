#include "llamagd.hpp"
#include "llama_state.hpp"
#include "register_types.hpp"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

namespace godot
{

   void initialize_llm_module(ModuleInitializationLevel p_level)
   {
      if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
      {
         return;
      }

      ClassDB::register_class<LlamaGD>();
      ClassDB::register_class<LlamaState>();
   }

   void cleanup_llm_module(ModuleInitializationLevel p_level)
   {
      if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
      {
         return;
      }
   }

   extern "C"
   {
      GDExtensionBool GDE_EXPORT llm_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)
      {
         godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

         init_obj.register_initializer(initialize_llm_module);
         init_obj.register_terminator(cleanup_llm_module);
         init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

         return init_obj.init();
      }
   }
}