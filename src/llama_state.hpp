#ifndef GD_MODEL_STATE
#define GD_MODEL_STATE

#include "llama_worker.hpp"

#include <llama.h>
#include <common.h>

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/ref.hpp>

// Wrapper class for LlamaWorkerState

namespace godot
{
   class LlamaState : public RefCounted
   {
      GDCLASS(LlamaState, RefCounted)

   public:
      LlamaState();
      ~LlamaState();

      LlamaWorkerState *worker_state;

      static void write_to_file(String destination, Ref<LlamaState> state);
      static Ref<LlamaState> read_from_file(String source);

      Array get_tokens() const;
      void set_tokens(const Array tokens);

      static Ref<LlamaState> create_state(LlamaWorkerState *state);
      static Ref<LlamaState> create_state(llama_model *model, gpt_params *params);

   protected:
      static void _bind_methods();
   };
}

#endif