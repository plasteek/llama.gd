#include "llama_state_file.hpp"
#include "llama_worker.hpp"
#include "conversion.hpp"
#include "llamagd.hpp"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/ref.hpp>

namespace godot
{
   void LlamaStateFile::_bind_methods()
   {
      ClassDB::bind_static_method(get_class_static(), D_METHOD("read_from_file", "source", "llama"), &LlamaStateFile::read_from_file);
      ClassDB::bind_static_method(get_class_static(), D_METHOD("write_to_file", "destination", "target"), &LlamaStateFile::write_to_file);
   }

   void LlamaStateFile::write_to_file(String dest, Ref<LlamaState> target_state)
   {
      auto *state = target_state->worker_state;
      llama_state_save_file(
          state->ctx,
          string_gd_to_std(dest).c_str(),
          state->tokens.data(),
          state->tokens.size());
   }
   Ref<LlamaState> LlamaStateFile::read_from_file(String src, LlamaGD *llama_node)
   {
      LlamaWorkerState *state = new LlamaWorkerState();
      llama_token *tokens;
      size_t token_count;

      if (!llama_node->is_model_loaded())
      {
         UtilityFunctions::push_error("Model has not been loaded");
         return LlamaState::create_state(state);
      }

      state->init(llama_node->get_model(), llama_node->get_params());
      llama_state_load_file(
          state->ctx,
          string_gd_to_std(src).c_str(),
          tokens,
          // Just read as much as possible because we don't really have a limit
          INT_MAX,
          &token_count);

      // Transform the token pointer to array
      auto gd_tokens = Array();
      for (int i = 0; i < token_count; i++)
      {
         auto token = tokens[i];
         gd_tokens.append(token);
         state->tokens.emplace_back(token);
      }

      state->last_decoded_token_index = token_count;
      std::free(tokens);

      return LlamaState::create_state(state);
   }

}