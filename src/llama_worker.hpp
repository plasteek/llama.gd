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
   bool initialized;
   llama_context *ctx;
   int last_decoded_token_index;
   std::vector<llama_token> tokens;

   void init_ctx(llama_context *ctx);
   void init_ctx(llama_model *model, gpt_params *locked_params);
   static LlamaWorkerState *clone(const LlamaWorkerState *worker, llama_model *model, gpt_params *params);
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
   void insert_without_bos(std::vector<llama_token> *embd, std::vector<llama_token> *tokens, llama_token bos);
   void ensure_state_initialized();

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
   std::string run_with_lookahead(std::vector<llama_token> tokens, lookahead_params *params);
   void stop();

   // Create a state with initial prompt
   LlamaWorkerState *create_state_from_prompt(const std::string prompt);
   void use_state(const LlamaWorkerState *state);
};

struct ngram_data
{
   bool active = false;
   llama_seq_id seq_id = -1;
   std::vector<int> batch_index;
   std::vector<llama_token> tokens;
};
struct ngram_pool
{
   ngram_pool(int n_vocab, int N, int G)
   {
      // This is basically the logits
      head.resize(n_vocab);
      count.resize(n_vocab);
      tokens.resize(n_vocab * G * (N - 1));
   }

   int n_total = 0;
   std::vector<int> count;
   // Head toe the actual tokens?
   std::vector<int> head;
   // [n_vocab][G][N - 1]
   // for each token of the vocab, keep a ring-buffer of capacity G of n-grams of size N - 1
   // NOTE: this means this is a "3d" array where each "head" has the size of G (for "lookahead")
   // and the count of n - 1 or so
   std::vector<llama_token> tokens;
};
struct lookahead_params
{
   int ngram_size;
   int window_size;
   int max_ngram_verify;
   lookahead_params();
};

#endif
