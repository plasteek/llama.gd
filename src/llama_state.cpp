#include "llama_worker.hpp"
#include "gd_throw.hpp"
#include "conversion.hpp"
#include "llama_state.hpp"

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
      // TODO: this may not register properly, need to check
      ClassDB::bind_static_method(get_class_static(), D_METHOD("read_from_file", "source"), &LlamaState::read_from_file);
      ClassDB::bind_static_method(get_class_static(), D_METHOD("write_state_to_file", "destination", "target"), &LlamaState::write_to_file);

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

   void LlamaState::write_to_file(String dest, Ref<LlamaState> target_state)
   {
      auto *state = target_state->worker_state;
      llama_state_save_file(
          state->ctx,
          string_gd_to_std(dest).c_str(),
          state->tokens.data(),
          state->tokens.size());
   }
   Ref<LlamaState> LlamaState::read_from_file(String src)
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
         gd_tokens.append(curr_token);
         state->tokens.emplace_back(curr_token);
         t += 1;
      }

      state->last_evaluated_token_index = *token_count;
      std::free(token_ptr);
      std::free(token_count);

      return LlamaState::create_state(state);
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