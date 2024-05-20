#include "conversion.hpp"
#include "llamagd.hpp"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/mutex.hpp>

#include <string>
#include <llama.h>
#include <common/common.h>

namespace godot
{
   void LlamaGD::_bind_methods()
   {

      ADD_SIGNAL(MethodInfo("model_loaded"));
      ADD_SIGNAL(MethodInfo("model_load_failed"));
      ADD_SIGNAL(MethodInfo("new_token_generated", PropertyInfo(Variant::STRING, "token")));

      // Below here are just godot getter and setters
      ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaGD::get_model_path);
      ClassDB::bind_method(D_METHOD("set_model_path"), &LlamaGD::set_model_path);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model_path", "get_model_path");

      ClassDB::bind_method(D_METHOD("get_flash_attn"), &LlamaGD::get_flash_attn);
      ClassDB::bind_method(D_METHOD("set_flash_attn"), &LlamaGD::set_flash_attn);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::STRING, "enable_flash_attn", PROPERTY_HINT_FILE), "set_flash_attn", "get_flash_attn");

      ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaGD::get_model_path);
      ClassDB::bind_method(D_METHOD("set_model_path"), &LlamaGD::set_model_path);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model_path", "get_model_path");

      ClassDB::bind_method(D_METHOD("get_input_prefix"), &LlamaGD::get_input_prefix);
      ClassDB::bind_method(D_METHOD("set_input_prefix", "p_input_prefix"), &LlamaGD::set_input_prefix);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::STRING, "input_prefix", PROPERTY_HINT_NONE), "set_input_prefix", "get_input_prefix");

      ClassDB::bind_method(D_METHOD("get_input_suffix"), &LlamaGD::get_input_suffix);
      ClassDB::bind_method(D_METHOD("set_input_suffix", "p_input_suffix"), &LlamaGD::set_input_suffix);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::STRING, "input_suffix", PROPERTY_HINT_NONE), "set_input_suffix", "get_input_suffix");

      ClassDB::bind_method(D_METHOD("get_should_output_bos"), &LlamaGD::get_should_output_bos);
      ClassDB::bind_method(D_METHOD("set_should_output_bos", "p_should_output_bos"), &LlamaGD::set_should_output_bos);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::BOOL, "should_output_bos", PROPERTY_HINT_NONE), "set_should_output_bos", "get_should_output_bos");

      ClassDB::bind_method(D_METHOD("get_should_output_eos"), &LlamaGD::get_should_output_eos);
      ClassDB::bind_method(D_METHOD("set_should_output_eos", "p_should_output_eos"), &LlamaGD::set_should_output_eos);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::BOOL, "should_output_eos", PROPERTY_HINT_NONE), "set_should_output_eos", "get_should_output_eos");

      ClassDB::bind_method(D_METHOD("get_n_ctx"), &LlamaGD::get_n_ctx);
      ClassDB::bind_method(D_METHOD("set_n_ctx", "p_n_ctx"), &LlamaGD::set_n_ctx);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "context_size", PROPERTY_HINT_NONE), "set_n_ctx", "get_n_ctx");

      ClassDB::bind_method(D_METHOD("get_n_predict"), &LlamaGD::get_n_predict);
      ClassDB::bind_method(D_METHOD("set_n_predict", "p_n_predict"), &LlamaGD::set_n_predict);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_predict", PROPERTY_HINT_NONE), "set_n_predict", "get_n_predict");

      ClassDB::bind_method(D_METHOD("get_n_keep"), &LlamaGD::get_n_keep);
      ClassDB::bind_method(D_METHOD("set_n_keep", "p_n_keep"), &LlamaGD::set_n_keep);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_keep", PROPERTY_HINT_NONE), "set_n_keep", "get_n_keep");

      ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaGD::get_temperature);
      ClassDB::bind_method(D_METHOD("set_temperature", "p_temperature"), &LlamaGD::set_temperature);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::FLOAT, "temperature", PROPERTY_HINT_NONE), "set_temperature", "get_temperature");

      ClassDB::bind_method(D_METHOD("get_penalty_repeat"), &LlamaGD::get_penalty_repeat);
      ClassDB::bind_method(D_METHOD("set_penalty_repeat", "p_penalty_repeat"), &LlamaGD::set_penalty_repeat);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::FLOAT, "penalty_repeat", PROPERTY_HINT_NONE), "set_penalty_repeat", "get_penalty_repeat");

      ClassDB::bind_method(D_METHOD("get_penalty_last_n"), &LlamaGD::get_penalty_last_n);
      ClassDB::bind_method(D_METHOD("set_penalty_last_n", "p_penalty_last_n"), &LlamaGD::set_penalty_last_n);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "penalty_last_n", PROPERTY_HINT_NONE), "set_penalty_last_n", "get_penalty_last_n");

      ClassDB::bind_method(D_METHOD("get_penalize_nl"), &LlamaGD::get_penalize_nl);
      ClassDB::bind_method(D_METHOD("set_penalize_nl", "p_penalize_nl"), &LlamaGD::set_penalize_nl);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::BOOL, "penalize_nl", PROPERTY_HINT_NONE), "set_penalize_nl", "get_penalize_nl");

      ClassDB::bind_method(D_METHOD("get_top_k"), &LlamaGD::get_top_k);
      ClassDB::bind_method(D_METHOD("set_top_k", "p_top_k"), &LlamaGD::set_top_k);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "top_k", PROPERTY_HINT_NONE), "set_top_k", "get_top_k");

      ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaGD::get_top_p);
      ClassDB::bind_method(D_METHOD("set_top_p", "p_top_p"), &LlamaGD::set_top_p);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::FLOAT, "top_p", PROPERTY_HINT_NONE), "set_top_p", "get_top_p");

      ClassDB::bind_method(D_METHOD("get_min_p"), &LlamaGD::get_min_p);
      ClassDB::bind_method(D_METHOD("set_min_p", "p_min_p"), &LlamaGD::set_min_p);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::FLOAT, "min_p", PROPERTY_HINT_NONE), "set_min_p", "get_min_p");

      ClassDB::bind_method(D_METHOD("get_n_threads"), &LlamaGD::get_n_threads);
      ClassDB::bind_method(D_METHOD("set_n_threads", "p_n_threads"), &LlamaGD::set_n_threads);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_threads", PROPERTY_HINT_NONE), "set_n_threads", "get_n_threads");

      ClassDB::bind_method(D_METHOD("get_n_gpu_layer"), &LlamaGD::get_n_gpu_layer);
      ClassDB::bind_method(D_METHOD("set_n_gpu_layer", "p_n_gpu_layer"), &LlamaGD::set_n_gpu_layer);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_gpu_layer", PROPERTY_HINT_NONE), "set_n_gpu_layer", "get_n_gpu_layer");

      ClassDB::bind_method(D_METHOD("get_escape"), &LlamaGD::get_escape);
      ClassDB::bind_method(D_METHOD("set_escape", "p_escape"), &LlamaGD::set_escape);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::BOOL, "escape", PROPERTY_HINT_NONE), "set_escape", "get_escape");

      ClassDB::bind_method(D_METHOD("get_n_batch"), &LlamaGD::get_n_batch);
      ClassDB::bind_method(D_METHOD("set_n_batch", "p_n_batch"), &LlamaGD::set_n_batch);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_batch", PROPERTY_HINT_NONE), "set_n_batch", "get_n_batch");

      ClassDB::bind_method(D_METHOD("get_n_ubatch"), &LlamaGD::get_n_ubatch);
      ClassDB::bind_method(D_METHOD("set_n_ubatch", "p_n_ubatch"), &LlamaGD::set_n_ubatch);
      ClassDB::add_property("LlamaGD", PropertyInfo(Variant::INT, "n_ubatch", PROPERTY_HINT_NONE), "set_n_ubatch", "get_n_ubatch");
   }
   LlamaGD::LlamaGD()
   {
      params = gpt_params();
      should_output_bos = true;
      should_output_eos = true;

      generation_mutex.instantiate();
      function_call_mutex.instantiate();

      text_generation_thread.instantiate();
      model_loader_thread.instantiate();

      // Todo: might not wanna do this
      // To instantiate the llama backend
      llama_backend_init();
      llama_numa_init(params.numa);
   }
   LlamaGD::~LlamaGD()
   {
      // Just in case not unloaded
      unload_model();
      // Free everything tthat is not freed
      llama_backend_free();
   }
   void LlamaGD::load_model()
   {
      // TODO: maybe lock the params when the model is finally loaded
      // Unlocked when loading failed or success
      function_call_mutex->lock();
      model_loader_thread->start(callable_mp(this, &LlamaGD::load_model_impl));
   }
   void LlamaGD::load_model_impl()
   {
      // What the hell?
      // dedicate one sequence to the system prompt
      params.n_parallel += 1;
      std::tie(model, ctx) = llama_init_from_gpt_params(params);
      // but be sneaky about it
      params.n_parallel -= 1;

      if (model == nullptr)
      {
         LOG("Unable to load model");
         emit_signal("model_load_failed");
         function_call_mutex->unlock();
      }

      // We don't want to use the model default
      // should_output_bos = llama_should_add_bos_token(model);
      // should_output_eos = (bool)llama_add_eos_token(model);
      // Might be important but LOL
      // params.n_ctx = llama_n_ctx(ctx);

      GGML_ASSERT(llama_add_eos_token(model) != 1);
      emit_signal("model_loaded");
      function_call_mutex->unlock();
   }
   void LlamaGD::unload_model()
   {
      llama_free_model(model);
      llama_free(ctx);
   }
   bool LlamaGD::is_params_locked()
   {
      return model == nullptr;
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
   String LlamaGD::create_completion(String prompt)
   {
      function_call_mutex->lock();

      std::string normalized_prompt = string_gd_to_std(prompt);
      std::string prompt_payload = normalized_prompt;
      prompt_payload = params.input_prefix + normalized_prompt + params.input_suffix;

      // Prompt has been prepared, run the generation!

      function_call_mutex->unlock();
   }

   // Below here are godot getter and setters
   String LlamaGD::get_model_path() const
   {
      return string_std_to_gd(params.model);
   }
   void LlamaGD::set_model_path(const String p_model_path)
   {
      if (should_block_setting_param())
         return;
      params.model = string_gd_to_std(p_model_path.trim_prefix(String("res://")));
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

   void LlamaGD::set_input_prefix(const String p_input_prefix)
   {
      if (should_block_setting_param())
         return;
      params.input_prefix = string_gd_to_std(p_input_prefix);
   };

   String LlamaGD::get_input_suffix() const
   {
      return string_std_to_gd(params.input_suffix);
   };

   void LlamaGD::set_input_suffix(const String p_input_suffix)
   {
      if (should_block_setting_param())
         return;
      params.input_suffix = string_gd_to_std(p_input_suffix);
   };

   bool LlamaGD::get_should_output_bos() const
   {
      return should_output_bos;
   };

   void LlamaGD::set_should_output_bos(const bool p_should_output_bos)
   {
      if (should_block_setting_param())
         return;
      should_output_bos = p_should_output_bos;
   };

   bool LlamaGD::get_should_output_eos() const
   {
      return should_output_eos;
   };

   void LlamaGD::set_should_output_eos(const bool p_should_output_eos)
   {
      if (should_block_setting_param())
         return;
      should_output_eos = p_should_output_eos;
   };

   int32_t LlamaGD::get_n_ctx() const
   {
      return params.n_ctx;
   }

   void LlamaGD::set_n_ctx(const int32_t p_n_ctx)
   {
      if (should_block_setting_param())
         return;
      params.n_ctx = p_n_ctx;
   }

   int32_t LlamaGD::get_n_predict() const
   {
      return params.n_predict;
   }

   void LlamaGD::set_n_predict(const int32_t p_n_predict)
   {
      if (should_block_setting_param())
         return;
      params.n_predict = p_n_predict;
   }

   int32_t LlamaGD::get_n_keep() const
   {
      return params.n_keep;
   }

   void LlamaGD::set_n_keep(const int32_t p_n_keep)
   {
      if (should_block_setting_param())
         return;
      params.n_keep = p_n_keep;
   }

   float LlamaGD::get_temperature() const
   {
      return params.sparams.temp;
   }

   void LlamaGD::set_temperature(const float p_temperature)
   {
      if (should_block_setting_param())
         return;
      params.sparams.temp = p_temperature;
   }

   float LlamaGD::get_penalty_repeat() const
   {
      return params.sparams.penalty_repeat;
   }

   void LlamaGD::set_penalty_repeat(const float p_penalty_repeat)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_repeat = p_penalty_repeat;
   }

   int32_t LlamaGD::get_penalty_last_n() const
   {
      return params.sparams.penalty_last_n;
   }

   void LlamaGD::set_penalty_last_n(const int32_t p_penalty_last_n)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalty_last_n = p_penalty_last_n;
   }

   bool LlamaGD::get_penalize_nl() const
   {
      return params.sparams.penalize_nl;
   }

   void LlamaGD::set_penalize_nl(const bool p_penalize_nl)
   {
      if (should_block_setting_param())
         return;
      params.sparams.penalize_nl = p_penalize_nl;
   }

   int32_t LlamaGD::get_top_k() const
   {
      return params.sparams.top_k;
   }

   void LlamaGD::set_top_k(const int32_t p_top_k)
   {
      if (should_block_setting_param())
         return;
      params.sparams.top_k = p_top_k;
   }

   float LlamaGD::get_top_p() const
   {
      return params.sparams.top_p;
   }

   void LlamaGD::set_top_p(const float p_top_p)
   {
      if (should_block_setting_param())
         return;
      params.sparams.top_p = p_top_p;
   }

   float LlamaGD::get_min_p() const
   {
      return params.sparams.min_p;
   }

   void LlamaGD::set_min_p(const float p_min_p)
   {
      if (should_block_setting_param())
         return;
      params.sparams.min_p = p_min_p;
   }

   int32_t LlamaGD::get_n_threads() const
   {
      return params.n_threads;
   }

   void LlamaGD::set_n_threads(const int32_t p_n_threads)
   {
      if (should_block_setting_param())
         return;
      params.n_threads = p_n_threads;
   }

   int32_t LlamaGD::get_n_gpu_layer() const
   {
      return params.n_gpu_layers;
   }

   void LlamaGD::set_n_gpu_layer(const int32_t p_n_gpu_layers)
   {
      if (should_block_setting_param())
         return;
      params.n_gpu_layers = p_n_gpu_layers;
   }

   bool LlamaGD::get_escape() const
   {
      return params.escape;
   }

   void LlamaGD::set_escape(const bool p_escape)
   {
      if (should_block_setting_param())
         return;
      params.escape = p_escape;
   }

   int32_t LlamaGD::get_n_batch() const
   {
      return params.n_batch;
   }

   void LlamaGD::set_n_batch(const int32_t p_n_batch)
   {
      if (should_block_setting_param())
         return;
      params.n_batch = p_n_batch;
   }

   int32_t LlamaGD::get_n_ubatch() const
   {
      return params.n_ubatch;
   }

   void LlamaGD::set_n_ubatch(const int32_t p_n_ubatch)
   {
      if (should_block_setting_param())
         return;
      params.n_ubatch = p_n_ubatch;
   }
}