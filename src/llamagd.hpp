#ifndef LLAMA_TYPES
#define LLAMA_TYPES

#include "llama_state.hpp"
#include "llama_worker.hpp"

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <common.h>

namespace godot
{
   class LlamaGD : public Node
   {
      GDCLASS(LlamaGD, Node)
   private:
      // Calling a function should not be able to be
      // done concurrently (generating text and load model)
      bool verbose;
      Ref<Mutex> function_call_mutex;
      Ref<Mutex> generation_mutex;
      Ref<Thread> text_generation_thread;
      Ref<Thread> model_loader_thread;
      gpt_params params;

      llama_model *model;
      LlamaWorker *worker;
      Ref<LlamaState> state;

      bool output_bos;
      bool output_eos;
      bool backend_initialized;

      void log(const std::string msg);

      void init_backend();
      void cleanup_backend();
      void cleanup_threads();
      void load_model_impl();

      bool is_params_locked();
      bool should_block_setting_param();

      void await_generation_thread();
      LlamaWorker *prepare_worker();
      void cleanup_worker();

      std::string predict_sequence_internal(const std::vector<llama_token> tokens);

   protected:
      static void _bind_methods();
      static void _bind_panel_attributes();

   public:
      LlamaGD();
      ~LlamaGD();
      // Starts a thread to load the model async
      void load_model();
      void unload_model();
      bool is_model_loaded();

      llama_model *get_model();
      gpt_params *get_params();

      int get_model_eos_id();
      int get_model_bos_id();
      String get_model_eos();
      String get_model_bos();

      String create_completion(const String prompt);
      void create_completion_async(const String prompt);
      String predict_sequence(const Array tokens);
      void predict_sequence_async(const Array tokens);
      void stop_generation();

      Ref<LlamaState> create_state(const String prompt);
      void create_state_async(const String prompt);

      void use_state(Ref<LlamaState> llama_state);
      void clear_state();

      Array tokenize(const String prompt);
      String decode(const Array tokens);

      void _exit_tree() override;

      // Getter-setters for godot attributes
      // Error run_create_completion(String prompt);
      String get_model_path() const;
      void set_model_path(const String p_model_path);

      bool get_flash_attn() const;
      void set_flash_attn(const bool enabled);

      bool get_output_bos() const;
      void set_output_bos(const bool enabled);

      bool get_output_eos() const;
      void set_output_eos(const bool enabled);

      String get_input_prefix() const;
      void set_input_prefix(const String new_prefix);

      String get_input_suffix() const;
      void set_input_suffix(const String new_suffix);

      int32_t get_n_ctx() const;
      void set_n_ctx(const int32_t n);

      int32_t get_n_predict() const;
      void set_n_predict(const int32_t n);

      int32_t get_n_keep() const;
      void set_n_keep(const int32_t n);

      float get_temperature() const;
      void set_temperature(const float temperature);

      float get_penalty_repeat() const;
      void set_penalty_repeat(const float penalty_value);

      float get_penalty_freq() const;
      void set_penalty_freq(const float penalty_value);

      float get_penalty_present() const;
      void set_penalty_present(const float penalty_value);

      int32_t get_penalty_last_n() const;
      void set_penalty_last_n(const int32_t total);

      bool get_penalize_nl() const;
      void set_penalize_nl(const bool penalty_value);

      int32_t get_top_k() const;
      void set_top_k(const int32_t k);

      float get_top_p() const;
      void set_top_p(const float p);

      float get_min_p() const;
      void set_min_p(const float p);

      int32_t get_n_threads() const;
      void set_n_threads(const int32_t n);

      int32_t get_n_gpu_layer() const;
      void set_n_gpu_layer(const int32_t n);

      bool get_escape() const;
      void set_escape(const bool enabled);

      int32_t get_n_batch() const;
      void set_n_batch(const int32_t n);

      int32_t get_n_ubatch() const;
      void set_n_ubatch(const int32_t n);

      bool get_verbose() const;
      void set_verbose(const bool enabled);

      bool get_busy() const;
      void set_busy(const bool is_busy);

      float get_cfg_scale() const;
      void set_cfg_scale(const float scale);

      String get_negative_prompt() const;
      void set_negative_prompt(const String prompt);
   };
}

#endif