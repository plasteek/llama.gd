#include "llama_worker.hpp"
#include "llama_utils.hpp"

#include <llama.h>
#include <common.h>
#include <fstream>
#include <string>
#include <cmath>
#include <stdexcept>

// Black magic from the llama.cpp main app
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

LlamaWorkerState::LlamaWorkerState()
{
    n_past = 0;
    n_consumed = 0;
    ctx = nullptr;
}
LlamaWorkerState::LlamaWorkerState(llama_model *model, gpt_params *params) : LlamaWorkerState()
{
    // Initialize default context
    auto cparams = llama_context_params_from_gpt_params(*params);
    ctx = llama_new_context_with_model(model, cparams);
}
LlamaWorkerState::~LlamaWorkerState()
{
    llama_free(ctx);
}

LlamaWorker::LlamaWorker(
    llama_model *loaded_model,
    gpt_params *locked_params)
{
    // We want to load or create our own context
    model = loaded_model;
    params = locked_params;
    state = new LlamaWorkerState(model, params);

    output_eos = true;
    output_bos = false;
    should_yield = false;
    // Default on_new_token that does absolutely nothing
    on_new_token = [this](std::string token) {
    };
}
LlamaWorker::~LlamaWorker()
{
    if (state != nullptr)
        delete state;
}

bool LlamaWorker::file_exists(const std::string path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

bool LlamaWorker::file_is_empty(const std::string path)
{
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

void LlamaWorker::stop()
{
    should_yield = true;
}

void LlamaWorker::use_state(const LlamaWorkerState *new_state)
{
    if (state != nullptr)
        delete state;

    // Copy the state to ensure immutability
    state = new LlamaWorkerState(*new_state);
}

void LlamaWorker::insert_without_bos(std::vector<llama_token> *embd, std::vector<llama_token> *tokens, llama_token bos)
{
    auto new_token_start = tokens->begin();
    if (tokens->front() == llama_token_bos(model))
        ++new_token_start;
    embd->insert(embd->end(), new_token_start, tokens->end());
}

// This long function is direct implementation from the main.cpp
std::string LlamaWorker::run(std::vector<llama_token> input_tokens)
{
#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log start\n");
#endif // LOG_DISABLE_LOGS

    // NOTE: the comments contains my version of what the hell is going on
    // Append the prompt
    std::string generated_text = "";

    // Just in case if state is cleared and wanted to be reused
    if (state == nullptr)
    {
        LOG("No initial state provided, creating a blank");
        state = new LlamaWorkerState(model, params);
    }

    if (state->ctx == nullptr)
    {
        LOG("State does not have a context. Aborting.");
        throw std::runtime_error("State does not have a context initialized");
        return "";
    }

    // Needed llama_context
    llama_sampling_params &sparams = (*params).sparams;
    llama_context *ctx_main = state->ctx;
    llama_context *ctx_guidance = NULL;

    // If some parameters are not supposed to be defined
    if ((*params).logits_all)
        throw std::runtime_error(std::string(__func__) + ": please use the 'perplexity' tool for perplexity calculations");
    if ((*params).embedding)
        throw std::runtime_error(std::string(__func__) + ": please use the 'embedding' tool for embedding calculations");

    // Parameter checks
    if ((*params).n_ctx != 0 && (*params).n_ctx < 8)
    {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        (*params).n_ctx = 8;
    }
    if ((*params).rope_freq_base != 0.0)
    {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, (*params).rope_freq_base);
    }
    if ((*params).rope_freq_scale != 0.0)
    {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, (*params).rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    if (sparams.cfg_scale > 1.f)
    {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(*params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx_main);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train)
        LOG_TEE(
            "%s: warning: model was trained on only %d context tokens (%d specified)\n",
            __func__, n_ctx_train, n_ctx);

    // print system information
    LOG_TEE("\n");
    LOG_TEE("%s\n", gpt_params_get_system_info((*params)).c_str());

    // does the model require a bos_token for starting generation?
    const bool add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG("add_bos: %d\n", add_bos);

    // construct the prompt tokens
    std::vector<llama_token> token_list;
    auto cached_tokens = state->tokens;

    // If the prompt is empty, add starting token
    if (token_list.empty())
    {
        token_list.emplace_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());
    }
    // append the state tokens if exist
    if (!cached_tokens.empty())
    {
        LOG("Detected state token. Embedding into the prompt\n");
        auto state_tokens = state->tokens;
        insert_without_bos(&token_list, &state_tokens, llama_token_bos(model));
    }
    // append the actual user tokens
    insert_without_bos(&token_list, &input_tokens, llama_token_bos(model));

    // Note: (n_ctx - 4) here is to match the logic for command line prompt handling via
    // --prompt or --file which uses the same value.
    int max_token_size = n_ctx - 4;
    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if (token_list.size() > max_token_size)
    {
        const int skipped_tokens = token_list.size() - max_token_size;
        token_list.resize(max_token_size);
        LOG("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
    }

    // LOG("prompt: \"%s\"\n", log_tostr((*params).prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());

    if ((int)token_list.size() > n_ctx - 4)
    {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)token_list.size(), n_ctx - 4);
        throw std::runtime_error(std::string(__func__) + ": error: prompt is too long (" + std::to_string((int)token_list.size()) + " tokens, max " + std::to_string(n_ctx - 4) + ")");
    }

    // number of tokens to keep when resetting context
    int n_keep = (*params).n_keep;
    if (n_keep < 0 || n_keep > (int)token_list.size())
    {
        n_keep = (int)token_list.size();
    }
    else
    {
        // always keep the BOS token
        n_keep += add_bos;
    }

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, (*params).n_batch, (*params).n_predict, (*params).n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if (*params).grp_attn_n > 1)
    int ga_i = 0;
    const int ga_n = (*params).grp_attn_n;
    const int ga_w = (*params).grp_attn_w;
    if (ga_n != 1)
    {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");                         // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
                                                                                        // GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        // GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    // evaluate initial prompt
    int consumed_index = state->n_consumed;
    LOG("embd_inp.size(): %d, n_consumed: %d\n", (int)token_list.size(), consumed_index);

    int n_batch = (*params).n_batch;
    batch_decode_tokens(
        n_batch,
        ctx_main,
        token_list);

    struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
    if (!ctx_sampling)
    {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        throw std::runtime_error(std::string(__func__) + ": failed to initialize sampling subsystem");
    }
    // push the prompt in the sampling context in order to apply repetition penalties later
    // for the prompt, we don't apply grammar rules
    for (int token_index = 0; token_index < token_list.size(); token_index++)
    {
        auto token = token_list[token_index];
        // should accept from the context. But we're not applying grammar so it's fine
        llama_sampling_accept(ctx_sampling, ctx_main, token, false);
        LOG_TEE(
            "build: sampling context accept '%s' at %d\n",
            llama_token_to_piece(ctx_main, token).c_str(),
            token_index);
    }

    // prepare for Guidance (if enabled)
    int guidance_offset; // Needed for shifting context
    if (ctx_guidance)
    {
        int prompt_size = token_list.size();
        std::vector<llama_token> guidance_tokens;
        guidance_tokens = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
        guidance_offset = guidance_tokens.size() - prompt_size;

        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_tokens).c_str());
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, input_tokens).c_str());
        LOG("original_prompt_len: %s", log_tostr(prompt_size));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));

        int input_size = 0;
        llama_token *input_buf = NULL;

        // Guidance context should have the same data with these modifications:
        //
        // * Replace the initial prompt
        // * Shift everything by guidance_offset
        if (token_list.begin() + prompt_size < token_list.end())
        {
            guidance_tokens.insert(
                guidance_tokens.end(),
                token_list.begin() + prompt_size,
                token_list.end());
        }

        LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, guidance_tokens).c_str());

        batch_decode_tokens(
            n_batch,
            ctx_guidance,
            guidance_tokens);
        guidance_tokens.clear();
    }

    int last_evaluated_token = token_list.size() - 1;
    int last_token_pos = last_evaluated_token;
    int remaining = (*params).n_predict;
    int guidance_token_pos = 0;

    // prediction start
    llama_batch predict_batch = llama_batch_init((*params).n_batch, 0, 1); // Should only have 1 at a time
    while (!should_yield && (remaining > 0))
    {
        // clear for next prediction
        llama_batch_clear(predict_batch);

        const llama_token sampled_id = llama_sampling_sample(ctx_sampling, ctx_main, ctx_guidance);
        llama_sampling_accept(ctx_sampling, ctx_main, sampled_id, true);
        LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, ctx_sampling->prev).c_str());

        --remaining;
        LOG("n_remain: %d\n", remaining);

        // "decode" and process the sampled token
        const std::string token_str = llama_token_to_piece(ctx_main, sampled_id, !(*params).conversation);
        bool is_bos = (sampled_id == llama_token_bos(model));
        bool is_eos = (sampled_id == llama_token_eos(model));
        if ((!is_bos || output_bos) && (!is_eos || output_eos))
        {
            generated_text.append(token_str);
            on_new_token(token_str);
        }

        // if generation finished, no need fo further decode for next iteration
        if (llama_token_is_eog(model, sampled_id))
        {
            LOG(" [end of text]\n");
            break;
        }

        // prepare the next logit for sampling
        int token_pos = last_token_pos + 1; // one more than last token
        last_token_pos = token_pos;
        // Decode logit for next sampling
        llama_batch_add(predict_batch, sampled_id, token_pos, {0}, true);
        if (llama_decode(ctx_main, predict_batch))
        {
            LOG_TEE("%s : failed to eval\n", __func__);
            throw std::runtime_error(std::string(__func__) + ": failed to eval");
        }

        LOG("n_past = %d\n", token_pos);
        if ((*params).n_print > 0 && token_pos % (*params).n_print == 0)
        {
            LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", token_pos, n_ctx);
        }

        // Handle context extension here
        if (ga_n == 1)
        {
            // infinite text generation via context shifting
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (token_pos + token_list.size() + std::max<int>(0, guidance_offset) >= n_ctx)
            {
                if ((*params).n_predict == -2)
                {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, (*params).n_predict);
                    break;
                }

                const int n_left = token_pos - n_keep;
                const int n_discard = n_left / 2;

                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    token_pos, n_left, n_ctx, n_keep, n_discard);

                llama_kv_cache_seq_rm(ctx_main, 0, n_keep, n_keep + n_discard);
                llama_kv_cache_seq_add(ctx_main, 0, n_keep + n_discard, token_pos, -n_discard);

                token_pos -= n_discard; // NOTE: guidance offset used to be affected

                // LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);
                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());
            }
        }
        else
        {
            // context extension via Self-Extend
            while (token_pos >= ga_i + ga_w)
            {
                const int ib = (ga_n * ga_i) / ga_w;
                const int bd = (ga_w / ga_n) * (ga_n - 1);
                const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                LOG("\n");
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, token_pos, ib * bd, ga_i + ib * bd, token_pos + ib * bd);
                LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, token_pos + ib * bd, dd, ga_i + ib * bd + ga_w + dd, token_pos + ib * bd + dd);

                llama_kv_cache_seq_add(ctx_main, 0, ga_i, token_pos, ib * bd);
                llama_kv_cache_seq_div(ctx_main, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                llama_kv_cache_seq_add(ctx_main, 0, ga_i + ib * bd + ga_w, token_pos + ib * bd, dd);

                token_pos -= bd;
                ga_i += ga_w / ga_n;
                LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", token_pos + bd, token_pos, ga_i);
            }
        }
    }

    LOG("prediction completed with %d tokens remaining\n", remaining);
    llama_print_timings(ctx_main);

    // Free all the used context here
    // Assume that state is handled immutably
    delete state;
    state = nullptr;

    llama_sampling_free(ctx_sampling);
    if (ctx_guidance)
        llama_free(ctx_guidance);

    return generated_text;
}

// Initialize or cache a state for a prompt
LlamaWorkerState *LlamaWorker::make_state(const std::string prompt)
{
    auto state = new LlamaWorkerState(model, params);
    state->tokens = ::llama_tokenize(model, prompt, true, true);
    int n_consumed = 0;

    // New context and sampling context
    llama_context *ctx = state->ctx;
    std::vector<llama_token> embd_inp = state->tokens;
    llama_sampling_context *ctx_sampling = llama_sampling_init(params->sparams);

    while ((int)embd_inp.size() > n_consumed)
    {
        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);
        ++n_consumed;
    }

    state->n_consumed = state->n_past = n_consumed;
    return state;
}