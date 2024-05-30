#ifndef LLAMA_WORKER_TYPES
#define LLAMA_WORKER_TYPES

#include <string>
#include <functional>
#include <common.h>

// Runner steps:
// 1. Setup the runner parameters
// 2. Call the `.run` function to being generation
// 3. Cleanup non-parameter variables

// C++ level LlamaWorker State object
class LlamaWorkerState
{
public:
   LlamaWorkerState();
   LlamaWorkerState(llama_model *model, gpt_params *params);
   ~LlamaWorkerState();

   // We can assume that n_consumed is already the same as the embeddings
   // And the initial past is also the same as n_consumed
   llama_context *ctx;
   int last_decoded_token_index;
   std::vector<llama_token> tokens;
};

class LlamaWorker
{
private:
   bool should_yield;
   gpt_params *params;
   llama_model *model;

   // It looks weird but we want so that godot can access
   // the object to be used
   LlamaWorkerState *state;

   bool file_exists(const std::string path);
   bool file_is_empty(const std::string path);
   void insert_without_bos(std::vector<llama_token> *embd, std::vector<llama_token> *tokens, llama_token bos);

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
   // More direct token prediction
   std::string run(std::vector<llama_token> tokens);
   void stop();
   void use_state(const LlamaWorkerState *state);
   // Create a state with initial prompt
   LlamaWorkerState *make_state(const std::string prompt);
};

#endif