#include "conversion.hpp"
#include "llamagd.hpp"
#include <stdexcept>

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

namespace godot
{
   void LlamaGD::_bind_methods()
   {
      ADD_SIGNAL(MethodInfo("model_loaded"));
      ADD_SIGNAL(MethodInfo("model_load_failed"));
      ADD_SIGNAL(MethodInfo("new_token_generated", PropertyInfo(Variant::STRING, "token")));
      ADD_SIGNAL(MethodInfo("generation_completed", PropertyInfo(Variant::STRING, "result")));
      ADD_SIGNAL(MethodInfo("generation_failed", PropertyInfo(Variant::STRING, "msg")));

      // Primary generation method
      ClassDB::bind_method(D_METHOD("create_completion", "prompt"), &LlamaGD::create_completion);
      ClassDB::bind_method(D_METHOD("create_completion_async", "prompt"), &LlamaGD::create_completion_async);
      ClassDB::bind_method(D_METHOD("tokenize", "prompt"), &LlamaGD::tokenize);
      ClassDB::bind_method(D_METHOD("predict_sequence", "tokens"), &LlamaGD::predict_sequence);
      ClassDB::bind_method(D_METHOD("predict_sequence_async", "tokens"), &LlamaGD::predict_sequence_async);
      ClassDB::bind_method(D_METHOD("load_model"), &LlamaGD::load_model);
      ClassDB::bind_method(D_METHOD("unload_model"), &LlamaGD::unload_model);
      ClassDB::bind_method(D_METHOD("is_model_loaded"), &LlamaGD::is_model_loaded);

      _bind_panel_attributes();
   }

   LlamaGD::LlamaGD()
   {
      params = gpt_params();
      should_output_bos = true;
      should_output_eos = true;
      backend_initialized = false;
      verbose = false;

      ctx = nullptr;
      model = nullptr;
      worker = nullptr;

      generation_mutex.instantiate();
      function_call_mutex.instantiate();

      text_generation_thread.instantiate();
      model_loader_thread.instantiate();
   }
   void LlamaGD::log(std::string msg)
   {
      if (verbose)
         UtilityFunctions::print(msg.c_str());
   }
   void LlamaGD::init_backend()
   {
      log("Initializing Llama.cpp Backend");
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
      log("LlamaGD node deconstructor");
      cleanup();
   }
   void LlamaGD::_exit_tree()
   {
      log("LlamaGD node left the tree");
      cleanup();
   }
   void LlamaGD::cleanup()
   {
      log("Cleaning up LlamaGD module");
      // Make sure no other function runs (if possible)
      function_call_mutex->try_lock();

      // If there is a running worker (meaning working is not null pointer)
      // (Note that this is kind of an overkill but assume there can be multiple
      // worker in the future or something). We stop it and wait.
      if (worker != nullptr || !generation_mutex->try_lock())
      {
         log("A running worker detected. Stopping work forcefully");
         // Stop the worker and let the calling function clean it up
         worker->stop();
      }

      // Properly cleanup the threads
      log("Waiting threads to finish running");
      if (model_loader_thread->is_started())
         model_loader_thread->wait_to_finish();
      if (text_generation_thread->is_started())
         text_generation_thread->wait_to_finish();

      log("Freeing llama backends and model");
      unload_model();
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
      // NOTE: the lock is in the main thread call
      // What the hell?
      // dedicate one sequence to the system prompt
      params.n_parallel += 1;
      std::tie(model, ctx) = llama_init_from_gpt_params(params);
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
      function_call_mutex->unlock();
   }
   void LlamaGD::unload_model()
   {
      function_call_mutex->lock();

      log("Freeing model and context");
      llama_free_model(model);
      llama_free(ctx);

      function_call_mutex->unlock();
   }
   bool LlamaGD::is_model_loaded()
   {
      return model != nullptr;
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

   void LlamaGD::create_completion_async(String prompt)
   {
      await_generation_thread();
      text_generation_thread->start(callable_mp(this, &LlamaGD::create_completion).bind(prompt));
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
   String LlamaGD::create_completion(String prompt)
   {

      if (!is_model_loaded())
      {
         log("Model not loaded before generating completion. Aborting");
         UtilityFunctions::push_error("Please load the model before creating completion");
         return "";
      }

      log("Start generation process");
      log("Pre-processing prompt");
      std::string normalized_prompt = string_gd_to_std(prompt);
      std::string prompt_payload = normalized_prompt;
      prompt_payload = params.input_prefix + normalized_prompt + params.input_suffix;

      log("Initializing llama worker");
      prepare_worker();
      String completion_result = "";
      try
      {
         log("Running completion");
         auto completion = worker->run(prompt_payload);
         completion_result = string_std_to_gd(completion);
      }
      catch (std::runtime_error err)
      {
         log("Error while creating completion. Aborting");
         std::string msg(err.what());
         String normalized_msg = string_std_to_gd(msg);
         UtilityFunctions::push_error(normalized_msg);
         call_deferred("emit_signal", "generation_failed", normalized_msg);
      }

      // Free after use
      log("Completion successful. Releasing worker");
      delete worker;

      // Free up generation if this uses mutex
      generation_mutex->unlock();

      call_deferred("emit_signal", "generation_completed", completion_result);
      return completion_result;
   }
   LlamaWorker *LlamaGD::prepare_worker()
   {
      worker = new LlamaWorker(
          model,
          ctx,
          &params);

      worker->output_eos = should_output_eos;
      worker->output_bos = should_output_bos;
      // Attach the signal listener
      worker->on_new_token = [this](std::string new_token)
      {
         call_deferred("emit_signal", "new_token_generated", string_std_to_gd(new_token));
      };

      return worker;
   }

   // Llama methods we extend to godot
   // We use Array because godot typed array usually not great
   Array LlamaGD::tokenize(const String prompt)
   {
      if (!is_model_loaded())
      {

         std::string payload = string_gd_to_std(prompt);
         auto tokens = ::llama_tokenize(model, payload, true, true);
         return int_vec_to_gd_arr(tokens);
      }

      /// Return empty array if model is not loaded
      UtilityFunctions::push_error("Cannot tokenize. Model has not been loaded");
      return Array();
   }
   // More direct token based approach
   void LlamaGD::predict_sequence_async(Array tokens)
   {
      await_generation_thread();
      text_generation_thread->start(callable_mp(this, &LlamaGD::predict_sequence).bind(tokens));
   }
   String LlamaGD::predict_sequence(Array tokens)
   {
      log("Starting token prediction");

      if (!is_model_loaded())
      {
         log("Model not loaded. Aborting");
         UtilityFunctions::push_error("Model is not loaded, cannot predict next sequence");
         return "";
      }

      String prediction_result = "";
      try
      {
         log("Converting GD Array to Vector");
         std::vector<llama_token> payload = gd_arr_to_int_vec(tokens);

         log("Preparing worker");
         prepare_worker();
         std::string result = worker->predict(payload);
         prediction_result = string_std_to_gd(result);
      }
      catch (std::runtime_error err)
      {
         log("Error while predicting tokens. Aborting");
         std::string msg(err.what());
         String normalized_msg = string_std_to_gd(msg);
         UtilityFunctions::push_error(normalized_msg);
         call_deferred("emit_signal", "generation_failed", normalized_msg);
      }
      log("Cleaning up worker");
      delete worker;

      // Ensure we cleanup mutex if we used a thread
      generation_mutex->unlock();
      return prediction_result;
   }

   // Below here are godot getter and setters
   void LlamaGD::_bind_panel_attributes()
   {
      // This describes the attributes panel (and the attributes itself)
      ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaGD::get_model_path);
      ClassDB::bind_method(D_METHOD("set_model_path"), &LlamaGD::set_model_path);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "`.gguf` path to model"), "set_model_path", "get_model_path");

      ADD_GROUP("Prompt Preprocessing", "");
      ClassDB::bind_method(D_METHOD("get_input_prefix"), &LlamaGD::get_input_prefix);
      ClassDB::bind_method(D_METHOD("set_input_prefix", "prefix"), &LlamaGD::set_input_prefix);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_prefix", PROPERTY_HINT_MULTILINE_TEXT, "Append to the beginning of prompt"), "set_input_prefix", "get_input_prefix");

      ClassDB::bind_method(D_METHOD("get_input_suffix"), &LlamaGD::get_input_suffix);
      ClassDB::bind_method(D_METHOD("set_input_suffix", "suffix"), &LlamaGD::set_input_suffix);
      ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_suffix", PROPERTY_HINT_MULTILINE_TEXT, "Append to the end of prompt"), "set_input_suffix", "get_input_suffix");

      ADD_GROUP("Output Tokens", "");
      ClassDB::bind_method(D_METHOD("get_output_bos"), &LlamaGD::get_output_bos);
      ClassDB::bind_method(D_METHOD("set_output_bos", "enabled"), &LlamaGD::set_output_bos);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "output_bos", PROPERTY_HINT_NONE, "Output the model BOS if enabled"), "set_should_output_bos", "get_should_output_bos");

      ClassDB::bind_method(D_METHOD("get_output_eos"), &LlamaGD::get_output_eos);
      ClassDB::bind_method(D_METHOD("set_output_eos", "enabled"), &LlamaGD::set_output_eos);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "output_eos", PROPERTY_HINT_NONE, "Output the model EOS if enabled"), "set_should_output_eos", "get_should_output_eos");

      ADD_GROUP("Generation Config", "");
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
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "escape", PROPERTY_HINT_NONE, "Escape special characters such as \\n, \\r, \\t, ', \", and \\ to make the model process them"), "set_escape", "get_escape");

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

      ClassDB::bind_method(D_METHOD("get_penalize_freq"), &LlamaGD::get_penalty_present);
      ClassDB::bind_method(D_METHOD("set_penalize_freq", "enabled"), &LlamaGD::set_penalty_freq);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "frequency_penalty", PROPERTY_HINT_NONE, "Penalize frequently occurring token. 0 to disable"), "set_penalize_nl", "get_penalize_nl");

      ClassDB::bind_method(D_METHOD("get_penalize_present"), &LlamaGD::get_penalty_present);
      ClassDB::bind_method(D_METHOD("set_penalize_present", "enabled"), &LlamaGD::set_penalty_present);
      ADD_PROPERTY(PropertyInfo(Variant::BOOL, "presence_penalty", PROPERTY_HINT_NONE, "Penalize present tokens. 0 to disable"), "set_penalize_nl", "get_penalize_nl");

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

   bool LlamaGD::get_output_bos() const
   {
      return should_output_bos;
   };
   void LlamaGD::set_output_bos(const bool enabled)
   {
      if (should_block_setting_param())
         return;
      should_output_bos = enabled;
   };

   bool LlamaGD::get_output_eos() const
   {
      return should_output_eos;
   };
   void LlamaGD::set_output_eos(const bool enabled)
   {
      if (should_block_setting_param())
         return;
      should_output_eos = enabled;
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
}