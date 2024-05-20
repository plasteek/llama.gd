#ifndef LLAMA_TYPES
#define LLAMA_TYPES

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <common/common.h>

namespace godot
{
   class LlamaGD : public Node
   {
      GDCLASS(LlamaGD, Node)
   private:
      // Generation
      Ref<Mutex> generation_mutex;
      // Calling a function should not be able to be
      // done concurrently (generating text and load model)
      Ref<Mutex> function_call_mutex;
      Ref<Mutex> model_loader_mutex;
      Ref<Thread> text_generation_thread;
      Ref<Thread> model_loader_thread;
      struct gpt_params params;
      struct llama_context *ctx;
      struct llama_model *model;
      bool should_output_bos;
      bool should_output_eos;
      std::string antiprompt;

      // Implementation for loading the model
      // and notifying godot through signal
      void load_model_impl();
      bool is_params_locked();
      bool should_block_setting_param();
      void reset_antiprompt();

   protected:
      static void _bind_methods();

   public:
      LlamaGD();
      ~LlamaGD();
      // Starts a thread to load the model async
      void load_model();
      void unload_model();
      String create_completion(String prompt);

      // Getter-setters for godot attributes
      // Error run_create_completion(String prompt);
      String get_model_path() const;
      void set_model_path(const String p_model_path);

      bool get_flash_attn() const;
      void set_flash_attn(const bool enabled);

      bool get_instruct() const;
      void set_instruct(const bool p_instruct);

      bool get_interactive() const;
      void set_interactive(const bool p_interactive);

      String get_antiprompt() const;
      void set_antiprompt(const String neg_prompt);

      bool get_should_output_bos() const;
      void set_should_output_bos(const bool p_should_output_bos);

      bool get_should_output_eos() const;
      void set_should_output_eos(const bool p_should_output_eos);

      String get_input_prefix() const;
      void set_input_prefix(const String p_input_prefix);

      String get_input_suffix() const;
      void set_input_suffix(const String p_input_suffix);

      int32_t get_n_ctx() const;
      void set_n_ctx(const int32_t p_n_ctx);

      int32_t get_n_predict() const;
      void set_n_predict(const int32_t p_n_predict);

      int32_t get_n_keep() const;
      void set_n_keep(const int32_t p_n_keep);

      float get_temperature() const;
      void set_temperature(const float p_temperature);

      float get_penalty_repeat() const;
      void set_penalty_repeat(const float p_penalty_repeat);

      int32_t get_penalty_last_n() const;
      void set_penalty_last_n(const int32_t p_penalty_last_n);

      bool get_penalize_nl() const;
      void set_penalize_nl(const bool p_penalize_nl);

      int32_t get_top_k() const;
      void set_top_k(const int32_t p_top_k);

      float get_top_p() const;
      void set_top_p(const float p_top_p);

      float get_min_p() const;
      void set_min_p(const float p_min_p);

      int32_t get_n_threads() const;
      void set_n_threads(const int32_t n_threads);

      int32_t get_n_gpu_layer() const;
      void set_n_gpu_layer(const int32_t p_n_gpu_layers);

      bool get_escape() const;
      void set_escape(const bool p_escape);

      int32_t get_n_batch() const;
      void set_n_batch(const int32_t p_n_batch);

      int32_t get_n_ubatch() const;
      void set_n_ubatch(const int32_t p_n_ubatch);
   };
}

#endif