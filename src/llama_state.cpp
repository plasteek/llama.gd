#include "llama_worker.hpp"
#include "gd_throw.cpp"
#include "conversion.hpp"
#include "llama_state.hpp"
#include <llama.h>
#include <common.h>
#include <stdexcept>

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot
{
   void LlamaState::_bind_methods()
   {
      // Export only functions because n_consumed and n_past should only be from
      // the cpp
      ClassDB::bind_static_method("read", &LlamaState::read, "source");
      ClassDB::bind_static_method("write", &LlamaState::write, "destination", "state");

      ClassDB::bind_method(D_METHOD("get_tokens"), &LlamaState::get_tokens);
      ClassDB::bind_method(D_METHOD("set_tokens"), &LlamaState::set_tokens);
      ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "tokens", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "set_tokens", "get_tokens");
   }
   LlamaState::LlamaState(LlamaWorkerState *initial_state)
   {
      state = initial_state;
   }
   LlamaState::LlamaState(llama_model *model, gpt_params *params)
   {
      state = new LlamaWorkerState(model, params);
   }
   LlamaState::~LlamaState()
   {
      delete state;
   }

   void LlamaState::write(String dest, LlamaState *target)
   {
      LlamaWorkerState *state = target->state;
      llama_state_save_file(
          state->ctx,
          string_gd_to_std(dest).c_str(),
          state->tokens.data(),
          state->tokens.size());
   }
   LlamaState *LlamaState::read(String src)
   {
      LlamaWorkerState *state = new LlamaWorkerState();
      llama_token *token_ptr;
      size_t *token_count;

      llama_state_load_file(
          state->ctx,
          string_gd_to_std(src).c_str(),
          token_ptr,
          // Just read as much as possible because we don't really have a limit
          INT_MAX,
          token_count);

      // Transform the token pointer to array
      auto gd_tokens = Array();
      llama_token *t = token_ptr;
      for (int i = 0; i < *token_count; i++)
      {
         auto curr_token = *t;
         gd_tokens.append(t);
         state->tokens.emplace_back(t);
         t += 1;
      }

      state->n_past = state->n_consumed = *token_count;
      std::free(token_ptr);
      std::free(token_count);

      return new LlamaState(state);
   }

   Array LlamaState::get_tokens() const
   {
      return int_vec_to_gd_arr(state->tokens);
   }
   void LlamaState::set_tokens(const Array tokens)
   {
      UtilityFunctions::push_error("You may not set token in a state object");
   }
}