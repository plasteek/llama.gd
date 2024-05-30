#ifndef LLAMA_GD_STATE_FILE
#define LLAMA_GD_STATE_FILE

#include "llama_state.hpp"
#include "llamagd.hpp"

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>

namespace godot
{
   // Class for reading and writing state files
   class LlamaStateFile : public RefCounted
   {
      GDCLASS(LlamaStateFile, RefCounted)

   protected:
      static void _bind_methods();

   public:
      static void write_to_file(String destination, Ref<LlamaState> state);
      static Ref<LlamaState> read_from_file(String source, LlamaGD *llama);
   };
}

#endif