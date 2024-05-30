#include "gd_throw.hpp"
#include "conversion.hpp"
#include "llama_state.hpp"
#include "llama_worker.hpp"
#include "llamagd.hpp"

#include <llama.h>
#include <common.h>
#include <stdexcept>

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/memory.hpp>

namespace godot
{
   void LlamaState::_bind_methods()
   {
      ClassDB::bind_method(D_METHOD("get_tokens"), &LlamaState::get_tokens);
      ClassDB::bind_method(D_METHOD("set_tokens"), &LlamaState::set_tokens);
      ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "tokens", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "set_tokens", "get_tokens");
   }
   Ref<LlamaState> LlamaState::create_state(LlamaWorkerState *initial_state)
   {
      Ref<LlamaState> new_state(memnew(LlamaState));
      new_state->worker_state = initial_state;
      return new_state;
   }
   Ref<LlamaState> LlamaState::create_state(llama_model *model, gpt_params *params)
   {
      Ref<LlamaState> new_state(memnew(LlamaState));
      auto worker_state = new LlamaWorkerState(model, params);
      new_state->worker_state = worker_state;
      return new_state;
   }
   LlamaState::LlamaState()
   {
      worker_state = nullptr;
   }
   LlamaState::~LlamaState()
   {
      delete worker_state;
   }

   Array LlamaState::get_tokens() const
   {
      if (worker_state == nullptr)
         return Array();
      return int_vec_to_gd_arr(worker_state->tokens);
   }
   void LlamaState::set_tokens(const Array tokens)
   {
      UtilityFunctions::push_error("You may not set token in a state object");
   }
}