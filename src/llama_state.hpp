#ifndef GD_MODEL_STATE
#define GD_MODEL_STATE

#include "llama_worker.hpp"

#include <llama.h>
#include <common.h>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/ref.hpp>

namespace godot
{
   // Wrapper class for LlamaWorkerState
   class LlamaState : public RefCounted
   {
      GDCLASS(LlamaState, RefCounted)

   public:
      LlamaState();
      ~LlamaState();

      LlamaWorkerState *worker_state;

      Array get_tokens() const;
      void set_tokens(const Array tokens);

      static Ref<LlamaState> create(LlamaWorkerState *state);
      static Ref<LlamaState> create(llama_model *model, gpt_params *params);

   protected:
      static void _bind_methods();
   };
}

#endif