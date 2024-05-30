#ifndef LLAMA_GD_UTILS
#define LLAMA_GD_UTILS

#include <llama.h>
#include <common.h>

void batch_decode_tokens(int batch_size, llama_context *ctx, std::vector<llama_token> tokens, int start_index = 0);

#endif