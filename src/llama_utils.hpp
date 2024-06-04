#ifndef LLAMA_GD_UTILS
#define LLAMA_GD_UTILS

#include <llama.h>
#include <common.h>

void batch_decode_tokens(int batch_size, llama_context *ctx, std::vector<llama_token> tokens, int start_index = 0, bool sample_last_logit = true);
void insert_without_bos(std::vector<llama_token> *embd, std::vector<llama_token> *tokens, llama_token bos);
std::vector<llama_token> construct_token_list(std::vector<llama_token> *state_token_list, std::vector<llama_token> *input_tokens, llama_token bos_token);

bool file_exists(const std::string path);
bool file_is_empty(const std::string path);

void llama_log_timings(llama_context *ctx);

#endif