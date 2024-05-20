#include <string>
#include <llama.h>
#include <functional>

#ifndef LLAMA_WORKER_TYPES
#define LLAMA_WORKER_TYPES

// Runner steps:
// 1. Setup the runner parameters
// 2. Call the `.run` function to being generation
// 3. Cleanup non-parameter variables

class LlamaWorker
{
private:
   bool should_yield;
   gpt_params params;
   llama_context *ctx;
   llama_model *model;

   bool output_bos; // TODO: export this and token new somehow
   bool output_eos;

   bool file_exists(const std::string path);
   bool file_is_empty(const std::string path);

   std::function<void(std::string)> on_new_token;

public:
   // Only allow the params to be the exact same as the params
   LlamaWorker(
       llama_model *loaded_model,
       llama_context *loaded_ctx,
       gpt_params *locked_params);
   void set_output_bos(bool enabled);
   void set_output_eos(bool enabled);
   void listen_for_new_token(std::function<void(std::string)>);
   // NOTICE: might throw error
   std::string run(std::string prompt);
   void stop();
};

#endif