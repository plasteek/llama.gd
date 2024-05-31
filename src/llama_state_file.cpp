#include "llama_state_file.hpp"
#include "llamagd.hpp"
#include "llama_worker.hpp"
#include "llama_utils.hpp"
#include "conversion.hpp"
#include "gd_throw.hpp"

#include <stdexcept>
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

      std::string src_path = string_gd_to_std(src);
      if (!file_exists(src_path))
      {
         UtilityFunctions::push_error("State file does not exist");
         return LlamaState::create_state(state);
      }

      state->init_ctx(llama_node->get_model(), llama_node->get_params());
      llama_context *ctx = state->ctx;

      // Just read as much as possible because we don't really have a limit
      if (!llama_state_load_file(ctx, src_path.c_str(), tokens, INT_MAX, &token_count))
      {
         UtilityFunctions::push_error("Cannot load state_file");
         return LlamaState::create_state(state);
      }

      // Append read tokens into the state
      for (int i = 0; i < token_count; i++)
      {
         auto token = tokens[i];
         state->tokens.emplace_back(token);
      }

      state->last_decoded_token_index = token_count;
      std::free(tokens);

      return LlamaState::create_state(state);
   }

}