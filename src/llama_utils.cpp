#include "llama_utils.hpp"
#include <llama.h>
#include <common.h>

void batch_decode_tokens(
    int batch_size,
    llama_context *ctx,
    std::vector<llama_token> tokens,
    int start_index)
{
   int token_size = tokens.size();
   // Edge case if all the batch "has been" evaluated
   if (start_index > token_size)
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

      batch.logits[0] = true; // sampling later asserts unless first value of the logits array is valid

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