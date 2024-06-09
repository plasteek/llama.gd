#include "gd_throw.hpp"
#include "conversion.hpp"
#include "llamagd.hpp"
#include "llama_worker.hpp"
#include "llama_state.hpp"

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/global_constants.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <llama.h>
#include <string>
#include <common.h>
#include <vector>
#include <stdexcept>

namespace godot
{
   void LlamaGD::_bind_methods()
   {
      ADD_SIGNAL(MethodInfo("model_loaded"));
      ADD_SIGNAL(MethodInfo("model_load_failed"));
      ADD_SIGNAL(MethodInfo("new_token_generated", PropertyInfo(Variant::STRING, "token")));
      ADD_SIGNAL(MethodInfo("generation_completed", PropertyInfo(Variant::STRING, "result")));
      ADD_SIGNAL(MethodInfo("generation_failed", PropertyInfo(Variant::STRING, "msg")));
      ADD_SIGNAL(MethodInfo("state_created", PropertyInfo(Variant::OBJECT, "state")));

      // Primary generation method
      ClassDB::bind_method(D_METHOD("tokenize", "prompt"), &LlamaGD::tokenize);
      ClassDB::bind_method(D_METHOD("decode", "tokens"), &LlamaGD::decode);

      ClassDB::bind_method(D_METHOD("get_model_eos"), &LlamaGD::get_model_eos);
      ClassDB::bind_method(D_METHOD("get_model_bos"), &LlamaGD::get_model_bos);
      ClassDB::bind_method(D_METHOD("get_model_eos_id"), &LlamaGD::get_model_eos_id);
      ClassDB::bind_method(D_METHOD("get_model_bos_id"), &LlamaGD::get_model_bos_id);

      ClassDB::bind_method(D_METHOD("create_completion", "prompt"), &LlamaGD::create_completion);
      ClassDB::bind_method(D_METHOD("create_completion_async", "prompt"), &LlamaGD::create_completion_async);
      ClassDB::bind_method(D_METHOD("stop_generation"), &LlamaGD::stop_generation);

      ClassDB::bind_method(D_METHOD("predict_sequence", "tokens"), &LlamaGD::predict_sequence);
      ClassDB::bind_method(D_METHOD("predict_sequence_async", "tokens"), &LlamaGD::predict_sequence_async);

      ClassDB::bind_method(D_METHOD("load_model"), &LlamaGD::load_model);
      ClassDB::bind_method(D_METHOD("unload_model"), &LlamaGD::unload_model);
      ClassDB::bind_method(D_METHOD("is_model_loaded"), &LlamaGD::is_model_loaded);
      ClassDB::bind_method(D_METHOD("is_loading_model"), &LlamaGD::is_loading_model);

      ClassDB::bind_method(D_METHOD("create_state", "prompt"), &LlamaGD::create_state);
      ClassDB::bind_method(D_METHOD("create_state_async", "prompt"), &LlamaGD::create_state_async);
      ClassDB::bind_method(D_METHOD("use_state", "state"), &LlamaGD::use_state);
      ClassDB::bind_method(D_METHOD("clear_state"), &LlamaGD::clear_state);

      _bind_panel_attributes();

      // Internal attributes
      ClassDB::bind_method(D_METHOD("get_busy"), &LlamaGD::get_busy);
      ClassDB::bind_method(D_METHOD("set_busy"), &LlamaGD::set_busy);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "busy", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "set_busy", "get_busy");
   }

   LlamaGD::LlamaGD()
   {
      params = gpt_params();
      lparams = lookahead_params();

      append_bos = true;
      output_bos = false;
      output_eos = false;

      verbose = false;
      lookahead = false;
      backend_initialized = false;

      model = nullptr;
      worker = nullptr;

      generation_mutex.instantiate();
      function_call_mutex.instantiate();

      text_generation_thread.instantiate();
      model_loader_thread.instantiate();

      init_backend();
   }
   void LlamaGD::log(std::string msg)
   {
      if (verbose)
         UtilityFunctions::print(("[LLAMA_GD] " + msg).c_str());
   }
   void LlamaGD::init_backend()
   {
      log("Initializing Llama.cpp backend");
      if (backend_initialized)
      {
         log("Backend has been initialized. Aborting.");
         return;
      }
      backend_initialized = true;
      llama_backend_init();
      llama_numa_init(params.numa);
   }

   LlamaGD::~LlamaGD()
   {
      // This is when the object is released
      // We want to fully cleanup
      log("LlamaGD node deconstructor");
      log("Cleaning up LlamaGD module");

      // We lock but do not wait if mutex waits for another call
      // we force the cleanup
      function_call_mutex->try_lock();
      cleanup_threads();

      // Because this means complete release, we want to completely
      // cleanup backend and assume we will never use it again
      cleanup_backend();
      unload_model();

      log("Cleanup Completed");
   }
   void LlamaGD::_exit_tree()
   {
      // This is when the node is removed, we assume that the model
      // or the backend will be re-used again unless released
      log("LlamaGD node left the tree");

      function_call_mutex->lock();
      cleanup_threads();
      function_call_mutex->unlock();
   }
   void LlamaGD::cleanup_threads()
   {
      log("Shutting down threads");
      if (worker != nullptr || !generation_mutex->try_lock())
      {
         log("A running worker detected. Stopping work forcefully");
         worker->stop();
         generation_mutex->lock();
      }
      // Properly cleanup the threads
      log("Waiting model loader thread to finish running");
      if (model_loader_thread->is_started())
         model_loader_thread->wait_to_finish();

      log("Waiting text generation thread to finish running");
      if (text_generation_thread->is_started())
         text_generation_thread->wait_to_finish();
   }
   void LlamaGD::cleanup_backend()
   {
      log("Freeing backend");
      if (!backend_initialized)
         return;
      llama_backend_free();
   }

   void LlamaGD::load_model()
   {
      function_call_mutex->lock();
      log("Attempting to load model");

      if (!backend_initialized)
      {
         log("Backend has not been initialized. Initializing");
         init_backend();
      }

      // Is a model is loaded, don't do anything
      if (model != nullptr)
      {
         log("Another model has been loaded. Aborting");
         UtilityFunctions::push_error("Model is already loaded, please unload before using");
         function_call_mutex->unlock();
         return;
      }
      // Unlocked when loading failed or success
      log("Beginning to load model in another thread");
      model_loader_thread->start(callable_mp(this, &LlamaGD::load_model_impl));
   }
   void LlamaGD::load_model_impl()
   {
      loading = true;
      // NOTE: the lock is in the main thread call
      // What the hell?
      // dedicate one sequence to the system prompt
      params.n_parallel += 1;

      auto mparams = llama_model_default_params();
      model = llama_load_model_from_file(params.model.c_str(), mparams);

      // but be sneaky about it
      params.n_parallel -= 1;

      if (model == nullptr)
      {
         log("Unable to load model");
         UtilityFunctions::push_error("Cannot load model");
         call_deferred("emit_signal", "model_load_failed");
         function_call_mutex->unlock();
         return;
      }

      // We don't want to use the model default
      // should_output_bos = llama_should_add_bos_token(model);
      // should_output_eos = (bool)llama_add_eos_token(model);
      // Might be important but LOL
      // params.n_ctx = llama_n_ctx(ctx);

      log("Model loaded successfully");

      GGML_ASSERT(llama_add_eos_token(model) != 1);
      call_deferred("emit_signal", "model_loaded");
      loading = false;

      function_call_mutex->unlock();
   }
   void LlamaGD::unload_model()
   {
      function_call_mutex->lock();

      log("Freeing model and context");
      llama_free_model(model);
      model = nullptr;

      function_call_mutex->unlock();
   }
   bool LlamaGD::is_model_loaded()
   {
      return model != nullptr;
   }
   bool LlamaGD::is_loading_model()
   {
      return loading;
   }
   bool LlamaGD::is_params_locked()
   {
      // Might add more constraint to when locking params
      return is_model_loaded();
   }
   bool LlamaGD::should_block_setting_param()
   {
      if (is_params_locked())
      {
         UtilityFunctions::push_error("Model has been loaded. Cannot set parameters until released");
         return true;
      }
      return false;
   }

   void LlamaGD::stop_generation()
   {
      if (worker == nullptr)
      {
         UtilityFunctions::push_error("No generation process has been started");
         return;
      }
      // Let the calling function handle the cleanup
      worker->stop();
   }

   void LlamaGD::create_completion_async(const String prompt)
   {
      await_generation_thread();
      text_generation_thread->start(callable_mp(this, &LlamaGD::create_completion).bind(prompt));
   }

   String LlamaGD::create_completion(const String prompt)
   {

      log("Pre-processing prompt");
      std::string normalized_prompt = string_gd_to_std(prompt);
      std::string prompt_payload = normalized_prompt;
      prompt_payload = params.input_prefix + normalized_prompt + params.input_suffix;

      auto tokens = ::llama_tokenize(model, prompt_payload, true, true);

      std::string result = predict_sequence_internal(tokens);
      String normalized = string_std_to_gd(result);
      call_deferred("emit_signal", "generation_completed", normalized);

      return normalized;
   }
   // More direct token based approach
   void LlamaGD::predict_sequence_async(const Array tokens)
   {
      await_generation_thread();
      text_generation_thread->start(callable_mp(this, &LlamaGD::predict_sequence).bind(tokens));
   }
   String LlamaGD::predict_sequence(const Array tokens)
   {
      try
      {
         log("Converting GDArray to vector");
         std::vector<llama_token> payload = gd_arr_to_int_vec(tokens);
         std::string result = predict_sequence_internal(payload);

         String normalized = string_std_to_gd(result);
         call_deferred("emit_signal", "generation_completed", normalized);
         return normalized;
      }
      catch (std::runtime_error err)
      {
         log("Error while predicting tokens. Aborting");
         gd_throw_err(err);
         call_deferred("emit_signal", "generation_failed", "Error predicting tokens");
      }
      // If the program throws return nothing
      return "";
   }

   std::string LlamaGD::predict_sequence_internal(const std::vector<llama_token> tokens)
   {
      if (!is_model_loaded())
      {
         log("Model not loaded. Aborting");
         UtilityFunctions::push_error("Model is not loaded, cannot predict next sequence");
         return "";
      }

      prepare_worker();
      std::string prediction_result = "";
      try
      {
         if (lookahead)
         {
            log("Starting prediction (lookahead decoding)");
            prediction_result = worker->run_with_lookahead(tokens, &lparams);
         }
         else
         {
            log("Starting prediction (autoregressive decoding - normal)");
            prediction_result = worker->run(tokens);
         }
      }
      catch (std::runtime_error err)
      {
         log("Error while predicting tokens. Aborting");
         gd_throw_err(err);
         call_deferred("emit_signal", "generation_failed", "Error predicting tokens");
      }

      log("Prediction Completed");
      cleanup_worker();

      // Ensure we cleanup mutex if we used a thread
      generation_mutex->unlock();
      return prediction_result;
   }

   void LlamaGD::await_generation_thread()
   {
      // This guaranteed that one function uses one thread at a time
      // We treat the generation mutex as a representation of the thread work
      // Hence as a shared resource that has to be protected
      function_call_mutex->lock();
      if (!generation_mutex->try_lock())
      {
         // Unlocked by the sync process of the caller
         log("Another generation running. Waiting.");
         generation_mutex->lock();
      }
      function_call_mutex->unlock();
      // Make sure that the thread is on idle
      if (text_generation_thread->is_started())
         text_generation_thread->wait_to_finish();
   }
   LlamaWorker *LlamaGD::prepare_worker()
   {
      log("Setting up new worker");
      worker = new LlamaWorker(
          model,
          &params);

      worker->output_eos = output_eos;
      worker->output_bos = output_bos;
      // Attach the signal listener
      worker->on_new_token = [this](std::string new_token)
      {
         call_deferred("emit_signal", "new_token_generated", string_std_to_gd(new_token));
      };

      if (state != nullptr)
      {
         log("Detected a state. Using custom initial state");
         auto worker_state = state->worker_state;
         worker->use_state(worker_state);
      }

      log("Worker initialized");
      return worker;
   }
   void LlamaGD::cleanup_worker()
   {
      log("Releasing worker from memory");
      delete worker;
      worker = nullptr;
      log("Release completed");
   }

   // Llama methods we extend to godot
   // We use Array because godot typed array usually not great
   Array LlamaGD::tokenize(const String prompt)
   {
      if (!is_model_loaded())
      {
         /// Return empty array if model is not loaded
         UtilityFunctions::push_error("Cannot tokenize. Model has not been loaded");
         return Array();
      }

      std::string payload = string_gd_to_std(prompt);
      auto tokens = ::llama_tokenize(model, payload, true, true);

      // To ensure tokenization is accurate, because of append_bos
      // remove BOS too if needed
      bool first_token_bos = tokens[0] == llama_token_bos(model);
      if (!append_bos && first_token_bos)
         tokens.erase(tokens.begin());

      return int_vec_to_gd_arr(tokens);
   }
   String LlamaGD::decode(const Array tokens)
   {
      if (!is_model_loaded())
      {
         UtilityFunctions::push_error("Cannot decode tokens. Model has not been loaded");
         return "";
      }

      String result = "";
      auto cparams = llama_context_params_from_gpt_params(params);
      llama_context *ctx = llama_new_context_with_model(model, cparams);

      for (int i = 0; i < tokens.size(); i++)
      {
         auto token = tokens[i];
         if (token.get_type() != Variant::INT)
         {
            UtilityFunctions::push_error("Invalid token", token);
            break;
         };

         std::string decoded = llama_token_to_piece(ctx, token, !params.conversation);
         result += string_std_to_gd(decoded);
      }

      llama_free(ctx);
      return result;
   }

   void LlamaGD::create_state_async(String prompt)
   {
      await_generation_thread();
      text_generation_thread->start(callable_mp(this, &LlamaGD::create_state).bind(prompt));
   }
   Ref<LlamaState> LlamaGD::create_state(String prompt)
   {
      prepare_worker();

      log("Creating state from prompt");
      auto state = worker->create_state_from_prompt(string_gd_to_std(prompt));

      log("State created. Releasing worker.");
      cleanup_worker();
      generation_mutex->unlock();

      auto new_state = LlamaState::create(state);
      call_deferred("emit_signal", "state_created", new_state);
      return new_state;
   }

   void LlamaGD::use_state(Ref<LlamaState> llama_state)
   {
      auto worker_state = llama_state->worker_state;
      state = llama_state;
   }
   void LlamaGD::clear_state()
   {
      state.unref();
   }

   String LlamaGD::get_model_bos()
   {
      auto token_id = get_model_bos_id();
      if (token_id == -1)
         return "";

      auto cparams = llama_context_params_from_gpt_params(params);
      llama_context *ctx = llama_new_context_with_model(model, cparams);

      String bos = string_std_to_gd(llama_token_to_piece(ctx, token_id, !params.conversation));
      llama_free(ctx);
      return bos;
   }
   String LlamaGD::get_model_eos()
   {
      auto token_id = get_model_eos_id();
      if (token_id == -1)
         return "";

      auto cparams = llama_context_params_from_gpt_params(params);
      llama_context *ctx = llama_new_context_with_model(model, cparams);

      String eos = string_std_to_gd(llama_token_to_piece(ctx, token_id, !params.conversation));
      llama_free(ctx);
      return eos;
   }

   int LlamaGD::get_model_bos_id()
   {
      if (!is_model_loaded())
      {
         UtilityFunctions::push_error("Cannot get BOS id. Model has not been loaded");
         return -1;
      }
      return llama_token_bos(model);
   }
   int LlamaGD::get_model_eos_id()
   {
      if (!is_model_loaded())
      {

         UtilityFunctions::push_error("Cannot get EOS Id. Model has not been loaded");
         // Assume that token cannot be negative
         return -1;
      }
      return llama_token_eos(model);
   }

   // Below here are godot getter and setters
   void LlamaGD::_bind_panel_attributes()
   {
      // This describes the attributes panel (and the attributes itself)
      ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaGD::get_model_path);
      ClassDB::bind_method(D_METHOD("set_model_path"), &LlamaGD::set_model_path);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"), "set_model_path", "get_model_path");

      ADD_GROUP("Preprocessing", "");
      ClassDB::bind_method(D_METHOD("get_input_prefix"), &LlamaGD::get_input_prefix);
      ClassDB::bind_method(D_METHOD("set_input_prefix", "prefix"), &LlamaGD::set_input_prefix);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_prefix", PROPERTY_HINT_MULTILINE_TEXT, "Append to the beginning of prompt"), "set_input_prefix", "get_input_prefix");

      ClassDB::bind_method(D_METHOD("get_input_suffix"), &LlamaGD::get_input_suffix);
      ClassDB::bind_method(D_METHOD("set_input_suffix", "suffix"), &LlamaGD::set_input_suffix);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_suffix", PROPERTY_HINT_MULTILINE_TEXT, "Append to the end of prompt"), "set_input_suffix", "get_input_suffix");

      ADD_GROUP("Lookahead Decoding (Experimental)", "");
      ClassDB::bind_method(D_METHOD("get_lookahead"), &LlamaGD::get_lookahead);
      ClassDB::bind_method(D_METHOD("set_lookahead", "enabled"), &LlamaGD::set_lookahead);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_lookahead", PROPERTY_HINT_NONE), "set_lookahead", "get_lookahead");

      ClassDB::bind_method(D_METHOD("get_window_size"), &LlamaGD::get_window_size);
      ClassDB::bind_method(D_METHOD("set_window_size", "size"), &LlamaGD::set_window_size);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "window_size", PROPERTY_HINT_NONE), "set_window_size", "get_window_size");

      ClassDB::bind_method(D_METHOD("get_ngram_size"), &LlamaGD::get_ngram_size);
      ClassDB::bind_method(D_METHOD("set_ngram_size", "size"), &LlamaGD::set_ngram_size);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "ngram_size", PROPERTY_HINT_NONE), "set_ngram_size", "get_ngram_size");

      ClassDB::bind_method(D_METHOD("get_max_verify"), &LlamaGD::get_max_verify);
      ClassDB::bind_method(D_METHOD("set_max_verify", "max"), &LlamaGD::set_max_verify);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "max_ngram_verification", PROPERTY_HINT_NONE, "-1 for windows size"), "set_max_verify", "get_max_verify");

      ADD_GROUP("Classifier-Free Guidance (CFG)", "");
      ClassDB::bind_method(D_METHOD("get_cfg_scale"), &LlamaGD::get_cfg_scale);
      ClassDB::bind_method(D_METHOD("set_cfg_scale", "scale"), &LlamaGD::set_cfg_scale);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cfg_scale", PROPERTY_HINT_NONE, "How strong is guidance"), "set_cfg_scale", "get_cfg_scale");

      ClassDB::bind_method(D_METHOD("get_negative_prompt"), &LlamaGD::get_cfg_scale);
      ClassDB::bind_method(D_METHOD("set_negative_prompt", "prompt"), &LlamaGD::set_cfg_scale);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "negative_prompt", PROPERTY_HINT_MULTILINE_TEXT, "Guidance prompt"), "set_negative_prompt", "get_negative_prompt");

      ADD_GROUP("Generation", "");
      ClassDB::bind_method(D_METHOD("get_n_ctx"), &LlamaGD::get_n_ctx);
      ClassDB::bind_method(D_METHOD("set_n_ctx", "context_size"), &LlamaGD::set_n_ctx);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "context_size", PROPERTY_HINT_NONE, "Set context size"), "set_n_ctx", "get_n_ctx");

      ClassDB::bind_method(D_METHOD("get_n_predict"), &LlamaGD::get_n_predict);
      ClassDB::bind_method(D_METHOD("set_n_predict", "n"), &LlamaGD::set_n_predict);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "max_token", PROPERTY_HINT_NONE, "Maximum token to predict. -1 for infinity"), "set_n_predict", "get_n_predict");

      ClassDB::bind_method(D_METHOD("get_n_keep"), &LlamaGD::get_n_keep);
      ClassDB::bind_method(D_METHOD("set_n_keep", "new_n_keep"), &LlamaGD::set_n_keep);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "persisted_token", PROPERTY_HINT_NONE, "If the prompt is very long, keep the first n token when shifting context"), "set_n_keep", "get_n_keep");

      ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaGD::get_temperature);
      ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &LlamaGD::set_temperature);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature", PROPERTY_HINT_NONE, "Set model randomness/temperature. <=0 for greedy sampling (best probability)"), "set_temperature", "get_temperature");

      ClassDB::bind_method(D_METHOD("get_escape"), &LlamaGD::get_escape);
      ClassDB::bind_method(D_METHOD("set_escape", "new_escape"), &LlamaGD::set_escape);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "process_escape", PROPERTY_HINT_NONE, "Escape special characters such as \\n, \\r, \\t, ', \", and \\ to make the model process them"), "set_escape", "get_escape");

      ADD_SUBGROUP("Output", "");
      ClassDB::bind_method(D_METHOD("get_append_bos"), &LlamaGD::get_append_bos);
      ClassDB::bind_method(D_METHOD("set_append_bos", "enabled"), &LlamaGD::set_append_bos);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "append_bos", PROPERTY_HINT_NONE, "Append BOS automatically"), "set_append_bos", "get_append_bos");

      ClassDB::bind_method(D_METHOD("get_output_bos"), &LlamaGD::get_output_bos);
      ClassDB::bind_method(D_METHOD("set_output_bos", "enabled"), &LlamaGD::set_output_bos);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "output_bos", PROPERTY_HINT_NONE, "Output the model BOS if enabled"), "set_output_bos", "get_output_bos");

      ClassDB::bind_method(D_METHOD("get_output_eos"), &LlamaGD::get_output_eos);
      ClassDB::bind_method(D_METHOD("set_output_eos", "enabled"), &LlamaGD::set_output_eos);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "output_eos", PROPERTY_HINT_NONE, "Output the model EOS if enabled"), "set_output_eos", "get_output_eos");

      ADD_SUBGROUP("Penalty", "");
      ClassDB::bind_method(D_METHOD("get_penalty_repeat"), &LlamaGD::get_penalty_repeat);
      ClassDB::bind_method(D_METHOD("set_penalty_repeat", "penalty_value"), &LlamaGD::set_penalty_repeat);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "repeat_penalty", PROPERTY_HINT_NONE, "Penalty for repeating similar semantic"), "set_penalty_repeat", "get_penalty_repeat");

      ClassDB::bind_method(D_METHOD("get_penalty_last_n"), &LlamaGD::get_penalty_last_n);
      ClassDB::bind_method(D_METHOD("set_penalty_last_n", "n"), &LlamaGD::set_penalty_last_n);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "penalize_last_n", PROPERTY_HINT_NONE, "Penalize the last n tokens. 0 to disable, -1 for context size"), "set_penalty_last_n", "get_penalty_last_n");

      ClassDB::bind_method(D_METHOD("get_penalize_nl"), &LlamaGD::get_penalize_nl);
      ClassDB::bind_method(D_METHOD("set_penalize_nl", "enabled"), &LlamaGD::set_penalize_nl);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "penalize_new_line", PROPERTY_HINT_NONE, "Penalize new line"), "set_penalize_nl", "get_penalize_nl");

      ClassDB::bind_method(D_METHOD("get_penalize_freq"), &LlamaGD::get_penalty_freq);
      ClassDB::bind_method(D_METHOD("set_penalize_freq", "enabled"), &LlamaGD::set_penalty_freq);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frequency_penalty", PROPERTY_HINT_NONE, "Penalize frequently occurring token. 0 to disable"), "set_penalize_freq", "get_penalize_freq");

      ClassDB::bind_method(D_METHOD("get_penalize_present"), &LlamaGD::get_penalty_present);
      ClassDB::bind_method(D_METHOD("set_penalize_present", "enabled"), &LlamaGD::set_penalty_present);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "presence_penalty", PROPERTY_HINT_NONE, "Penalize present tokens. 0 to disable"), "set_penalize_present", "get_penalize_present");

      ADD_SUBGROUP("Sampling", "");
      ClassDB::bind_method(D_METHOD("get_top_k"), &LlamaGD::get_top_k);
      ClassDB::bind_method(D_METHOD("set_top_k", "k"), &LlamaGD::set_top_k);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k", PROPERTY_HINT_NONE, "Sample only k most probable tokens. <=0 to use all"), "set_top_k", "get_top_k");

      ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaGD::get_top_p);
      ClassDB::bind_method(D_METHOD("set_top_p", "p"), &LlamaGD::set_top_p);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p", PROPERTY_HINT_NONE, "Sample only tokens with cumulative probability of p. 1.0 to disable"), "set_top_p", "get_top_p");

      ClassDB::bind_method(D_METHOD("get_min_p"), &LlamaGD::get_min_p);
      ClassDB::bind_method(D_METHOD("set_min_p", "p"), &LlamaGD::set_min_p);
      ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_p", PROPERTY_HINT_NONE, "Sample tokens with at least p probability, 0.0 tot disable"), "set_min_p", "get_min_p");

      ADD_GROUP("Performance", "");
      ClassDB::bind_method(D_METHOD("get_flash_attn"), &LlamaGD::get_flash_attn);
      ClassDB::bind_method(D_METHOD("set_flash_attn"), &LlamaGD::set_flash_attn);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_flash_attention", PROPERTY_HINT_NONE, "Enable flash attention (capable model only)"), "set_flash_attn", "get_flash_attn");

      ClassDB::bind_method(D_METHOD("get_n_threads"), &LlamaGD::get_n_threads);
      ClassDB::bind_method(D_METHOD("set_n_threads", "new_n_threads"), &LlamaGD::set_n_threads);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads", PROPERTY_HINT_NONE, "CPU threads to use"), "set_n_threads", "get_n_threads");

      ClassDB::bind_method(D_METHOD("get_n_gpu_layer"), &LlamaGD::get_n_gpu_layer);
      ClassDB::bind_method(D_METHOD("set_n_gpu_layer", "n"), &LlamaGD::set_n_gpu_layer);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "n_gpu_layer", PROPERTY_HINT_NONE, "Offload N layers to the GPU. -1 to use GPU fully"), "set_n_gpu_layer", "get_n_gpu_layer");

      ClassDB::bind_method(D_METHOD("get_n_batch"), &LlamaGD::get_n_batch);
      ClassDB::bind_method(D_METHOD("set_n_batch", "new_n_batch"), &LlamaGD::set_n_batch);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "n_batch", PROPERTY_HINT_NONE, "Logical batch size for prompt processing"), "set_n_batch", "get_n_batch");

      ClassDB::bind_method(D_METHOD("get_n_ubatch"), &LlamaGD::get_n_ubatch);
      ClassDB::bind_method(D_METHOD("set_n_ubatch", "new_n_ubatch"), &LlamaGD::set_n_ubatch);
      ADD_PROPERTY(PropertyInfo(Variant::INT, "n_ubatch", PROPERTY_HINT_NONE, "Physical batch size for prompt processing"), "set_n_ubatch", "get_n_ubatch");

      ADD_GROUP("Debugging", "");
      ClassDB::bind_method(D_METHOD("get_verbose"), &LlamaGD::get_verbose);
      ClassDB::bind_method(D_METHOD("set_verbose", "new_n_ubatch"), &LlamaGD::set_verbose);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "verbose", PROPERTY_HINT_NONE, "Logs internal plugin processes"), "set_verbose", "get_verbose");
   }

   llama_model *LlamaGD::get_model()
   {
      return model;
   }
   gpt_params *LlamaGD::get_params()
   {
      return &params;
   }

   String LlamaGD::get_model_path() const
   {
      return string_std_to_gd(params.model);
   }
   void LlamaGD::set_model_path(const String path)
   {
      if (should_block_setting_param())
         return;
      params.model = string_gd_to_std(path.trim_prefix(String("res://")));
   }

   bool LlamaGD::get_flash_attn() const
   {
      return params.flash_attn;
   }
   void LlamaGD::set_flash_attn(const bool enabled)
   {
      params.flash_attn = enabled;
   }

   String LlamaGD::get_input_prefix() const
   {
      return string_std_to_gd(params.input_prefix);
   };
   void LlamaGD::set_input_prefix(const String prefix)
   {
      if (should_block_setting_param())
         return;
      params.input_prefix = string_gd_to_std(prefix);
   };

   String LlamaGD::get_input_suffix() const
   {
      return string_std_to_gd(params.input_suffix);
   };
   void LlamaGD::set_input_suffix(const String suffix)
   {
      if (should_block_setting_param())
         return;
      params.input_suffix = string_gd_to_std(suffix);
   };

   bool LlamaGD::get_append_bos() const
   {
      return append_bos;
   }
   void LlamaGD::set_append_bos(const bool enabled)
   {
      append_bos = enabled;
   }

   bool LlamaGD::get_output_bos() const
   {
      return output_bos;
   };
   void LlamaGD::set_output_bos(const bool enabled)
   {
      if (should_block_setting_param())
         return;
      output_bos = enabled;
   };

   bool LlamaGD::get_output_eos() const
   {
      return output_eos;
   };
   void LlamaGD::set_output_eos(const bool enabled)
   {
      if (should_block_setting_param())
         return;
      output_eos = enabled;
   };

   int32_t LlamaGD::get_n_ctx() const
   {
      return params.n_ctx;
   }
   void LlamaGD::set_n_ctx(const int32_t context_size)
   {
      if (should_block_setting_param())
         return;
      params.n_ctx = context_size;
   }

   int32_t LlamaGD::get_n_predict() const
   {
      return params.n_predict;
   }
   void LlamaGD::set_n_predict(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_predict = n;
   }

   int32_t LlamaGD::get_n_keep() const
   {
      return params.n_keep;
   }
   void LlamaGD::set_n_keep(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_keep = n;
   }

   float LlamaGD::get_temperature() const
   {
      return params.sparams.temp;
   }
   void LlamaGD::set_temperature(const float temperature)
   {
      if (should_block_setting_param())
         return;
      params.sparams.temp = temperature;
   }

   float LlamaGD::get_penalty_repeat() const
   {
      return params.sparams.penalty_repeat;
   }
   void LlamaGD::set_penalty_repeat(const float penalty_value)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_repeat = penalty_value;
   }

   int32_t LlamaGD::get_penalty_last_n() const
   {
      return params.sparams.penalty_last_n;
   }
   void LlamaGD::set_penalty_last_n(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_last_n = n;
   }

   bool LlamaGD::get_penalize_nl() const
   {
      return params.sparams.penalize_nl;
   }
   void LlamaGD::set_penalize_nl(const bool penalty)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalize_nl = penalty;
   }

   float LlamaGD::get_penalty_freq() const
   {
      return params.sparams.penalty_freq;
   }
   void LlamaGD::set_penalty_freq(const float penalty_value)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_freq = penalty_value;
   }

   float LlamaGD::get_penalty_present() const
   {
      return params.sparams.penalty_present;
   }
   void LlamaGD::set_penalty_present(const float penalty_value)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_present = penalty_value;
   }

   int32_t LlamaGD::get_top_k() const
   {
      return params.sparams.top_k;
   }
   void LlamaGD::set_top_k(const int32_t k)
   {
      if (should_block_setting_param())
         return;
      params.sparams.top_k = k;
   }

   float LlamaGD::get_top_p() const
   {
      return params.sparams.top_p;
   }
   void LlamaGD::set_top_p(const float p)
   {
      if (should_block_setting_param())
         return;
      params.sparams.top_p = p;
   }

   float LlamaGD::get_min_p() const
   {
      return params.sparams.min_p;
   }
   void LlamaGD::set_min_p(const float p)
   {
      if (should_block_setting_param())
         return;
      params.sparams.min_p = p;
   }

   int32_t LlamaGD::get_n_threads() const
   {
      return params.n_threads;
   }
   void LlamaGD::set_n_threads(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_threads = n;
   }

   int32_t LlamaGD::get_n_gpu_layer() const
   {
      return params.n_gpu_layers;
   }
   void LlamaGD::set_n_gpu_layer(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_gpu_layers = n;
   }

   bool LlamaGD::get_escape() const
   {
      return params.escape;
   }
   void LlamaGD::set_escape(const bool enabled)
   {
      if (should_block_setting_param())
         return;
      params.escape = enabled;
   }

   int32_t LlamaGD::get_n_batch() const
   {
      return params.n_batch;
   }
   void LlamaGD::set_n_batch(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_batch = n;
   }

   int32_t LlamaGD::get_n_ubatch() const
   {
      return params.n_ubatch;
   }
   void LlamaGD::set_n_ubatch(const int32_t n)
   {
      if (should_block_setting_param())
         return;
      params.n_ubatch = n;
   }

   bool LlamaGD::get_verbose() const
   {
      return verbose;
   }
   void LlamaGD::set_verbose(bool enabled)
   {
      verbose = enabled;
   }

   bool LlamaGD::get_busy() const
   {
      return worker != nullptr;
   }
   void LlamaGD::set_busy(const bool is_busy)
   {
      UtilityFunctions::push_error("Cannot set read only property of `busy`");
   }

   float LlamaGD::get_cfg_scale() const
   {
      return params.sparams.cfg_scale;
   }
   void LlamaGD::set_cfg_scale(const float scale)
   {
      params.sparams.cfg_scale = scale;
   }

   String LlamaGD::get_negative_prompt() const
   {
      return string_std_to_gd(params.sparams.cfg_negative_prompt);
   }
   void LlamaGD::set_negative_prompt(const String prompt)
   {
      params.sparams.cfg_negative_prompt = string_gd_to_std(prompt);
   }

   bool LlamaGD::get_lookahead() const
   {
      return lookahead;
   }
   void LlamaGD::set_lookahead(const bool enabled)
   {
      lookahead = enabled;
   }

   int LlamaGD::get_window_size() const
   {
      return lparams.window_size;
   }
   void LlamaGD::set_window_size(const int size)
   {
      lparams.window_size = size;
   }

   int LlamaGD::get_ngram_size() const
   {
      return lparams.ngram_size;
   }
   void LlamaGD::set_ngram_size(const int size)
   {
      lparams.ngram_size = size;
   }

   int LlamaGD::get_max_verify() const
   {
      return lparams.max_ngram_verify;
   }
   void LlamaGD::set_max_verify(const int max_verify)
   {
      lparams.max_ngram_verify = max_verify;
   }
}