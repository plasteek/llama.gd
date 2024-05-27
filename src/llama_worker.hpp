#ifndef LLAMA_WORKER_TYPES
#define LLAMA_WORKER_TYPES

#include <string>
#include <functional>
#include <common.h>

// Runner steps:
// 1. Setup the runner parameters
// 2. Call the `.run` function to being generation
// 3. Cleanup non-parameter variables

class LlamaWorker
{
private:
   bool should_yield;
   gpt_params *params;
   llama_context *ctx;
   llama_model *model;

   bool file_exists(const std::string path);
   bool file_is_empty(const std::string path);

public:
   bool output_bos;
   bool output_eos;
   std::function<void(std::string)> on_new_token;

public:
   // Only allow the params to be the exact same as the params
   LlamaWorker(
       llama_model *loaded_model,
       gpt_params *locked_params);
   ~LlamaWorker();
   // NOTICE: might throw error
   std::string run(std::string prompt);
   // More direct token prediction
   std::string predict(std::vector<llama_token> tokens);
   void stop();
   // TODO: maybe have a use_state(state here or something)
   // Not sure what to do with the guidance context tho.
   // TODO: maybe have a make_state(prompt here or something)
   // which should return the state object?
};

#endif