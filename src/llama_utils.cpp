#include "llama_utils.hpp"
#include <llama.h>
#include <common.h>
#include <fstream>

void batch_decode_tokens(
    int batch_size,
    llama_context *ctx,
    std::vector<llama_token> tokens,
    int start_index)
{
   int token_size = tokens.size();
   // Edge case if all the batch "has been" evaluated
   if (start_index > token_size - 1)
      return;

   llama_batch batch = llama_batch_init(batch_size, 0, 1);
   for (int batch_start = start_index; batch_start < token_size; batch_start += batch_size)
   {
      int remaining = token_size - batch_start;
      int batch_count = std::min(remaining, batch_size);

      // Begin decoding
      llama_batch_clear(batch);
      for (int token_index = 0; token_index < batch_count; token_index++)
      {
         auto token_pos = batch_start + token_index;
         auto token = tokens[token_pos];
         llama_batch_add(batch, token, token_pos, {0}, false);

         LOG_TEE(
             "batch_decode: token: '%s', token_pos: %d\n",
             llama_token_to_piece(ctx, token).c_str(),
             token_pos);
      }

      // Output the logit for the very last element
      bool is_last = batch_start + batch_count == token_size;
      if (is_last)
         batch.logits[batch.n_tokens - 1] = true;

      if (llama_decode(ctx, batch) != 0)
      {
         LOG_TEE("%s : failed to eval\n", __func__);
         throw std::runtime_error(std::string(__func__) + ": failed to eval");
      }
   }
}

void insert_without_bos(std::vector<llama_token> *embd, std::vector<llama_token> *tokens, llama_token bos)
{
   auto new_token_start = tokens->begin();
   if (tokens->front() == bos)
      ++new_token_start;
   embd->insert(embd->end(), new_token_start, tokens->end());
}

std::vector<llama_token> merge_token_list(
    std::vector<llama_token> *state_token_list,
    std::vector<llama_token> *input_tokens,
    llama_token bos_token,
    bool beginning_bos)
{
   std::vector<llama_token> token_list;
   if (beginning_bos)
      // If the prompt is empty, add starting token
      token_list.emplace_back(bos_token);
   // append the state tokens if exist
   if (!state_token_list->empty())
   {
      LOG("Detected state token. Embedding into the prompt\n");
      insert_without_bos(&token_list, state_token_list, bos_token);
   }
   // append the actual user tokens
   insert_without_bos(&token_list, &*input_tokens, bos_token);

   return token_list;
}

bool file_exists(const std::string path)
{
   std::ifstream f(path.c_str());
   return f.good();
}

bool file_is_empty(const std::string path)
{
   std::ifstream f;
   f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
   f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
   return f.tellg() == 0;
}

void llama_log_timings(llama_context *ctx)
{
   const llama_timings timings = llama_get_timings(ctx);

   LOG("\n");
   LOG("%s:        load time = %10.2f ms\n", __func__, timings.t_load_ms);
   LOG("%s:      sample time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
       __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
   LOG("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
       __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
   LOG("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
       __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
   LOG("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (timings.t_end_ms - timings.t_start_ms), (timings.n_p_eval + timings.n_eval));
}