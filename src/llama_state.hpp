#ifndef GD_MODEL_STATE
#define GD_MODEL_STATE

#include "llama_worker.hpp"
#include <godot_cpp/classes/ref_counted.hpp>
#include <llama.h>
#include <common.h>

// Wrapper class for LlamaWorkerState

namespace godot
{
   class LlamaState : public RefCounted
   {
      GDCLASS(LlamaState, RefCounted)

   public:
      LlamaState(LlamaWorkerState *state);
      LlamaState(llama_model *model, gpt_params *params);
      ~LlamaState();

      LlamaWorkerState *state;

      static void write(String destination, LlamaState *state);
      static LlamaState *read(String source);

      Array get_tokens() const;
      void set_tokens(const Array tokens);

   protected:
      static void _bind_methods();
   };
}

#endif