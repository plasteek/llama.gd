#include "llama_worker.hpp"
#include "llama_state.hpp"
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
    n_consumed = 0;
    n_past = 0;
}
LlamaWorkerState::LlamaWorkerState(llama_model *model, gpt_params *params)
{
    // Call default overloading
    LlamaWorkerState();
    // Initialize default context
    auto cparams = llama_context_default_params();
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

std::string LlamaWorker::run(std::string prompt)
{
    (*params).prompt = prompt;
    auto tokens = ::llama_tokenize(model, prompt, true, true);
    return predict(tokens);
}

void LlamaWorker::use_state(const LlamaWorkerState *new_state)
{
    if (state != nullptr)
        std::free(state);

    // Copy the state to ensure immutability
    state = (LlamaWorkerState *)malloc(sizeof(new_state));
    memcpy(state, new_state, sizeof(new_state));
}

// This long function is direct implementation from the main.cpp
std::string LlamaWorker::predict(std::vector<llama_token> tokens)
{
    // NOTE: the comments contains my version of what the hell is going on
    // Append the prompt
    std::string generated_text = "";

    // Needed llama_context
    llama_sampling_params &sparams = (*params).sparams;
    llama_context *ctx = state->ctx;
    llama_context *ctx_guidance = NULL;

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log start\n");
#endif // LOG_DISABLE_LOGS

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
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train)
        LOG_TEE(
            "%s: warning: model was trained on only %d context tokens (%d specified)\n",
            __func__, n_ctx_train, n_ctx);

    {
        // print system information
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info((*params)).c_str());
    }

    // Does the model require a bos_token for starting generation?
    const bool add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG("add_bos: %d\n", add_bos);

    // Tokenize the prompt
    std::vector<llama_token> embd_inp = tokens;

    // TODO: because the prompt is optional and we only have token
    // LOG("prompt: \"%s\"\n", log_tostr((*params).prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // If the prompt is empty, add starting token
    if (embd_inp.empty())
    {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    if (ctx_guidance)
    {
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

        guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

        std::vector<llama_token> original_inp = tokens;
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));
    }

    if ((int)embd_inp.size() > n_ctx - 4)
    {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
        throw std::runtime_error(std::string(__func__) + ": error: prompt is too long (" + std::to_string((int)embd_inp.size()) + " tokens, max " + std::to_string(n_ctx - 4) + ")");
    }

    // Number of tokens to keep when resetting context
    if ((*params).n_keep < 0 || (*params).n_keep > (int)embd_inp.size())
    {
        (*params).n_keep = (int)embd_inp.size();
    }
    else
    {
        (*params).n_keep += add_bos; // always keep the BOS token
    }

    // Verbose prompt logging I assume.
    if ((*params).verbose_prompt)
    {
        LOG_TEE("\n");
        // LOG_TEE("%s: prompt: '%s'\n", __func__, (*params).prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int)embd_inp.size(); i++)
        {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (ctx_guidance)
        {
            LOG_TEE("\n");
            LOG_TEE("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
            LOG_TEE("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int)guidance_inp.size(); i++)
            {
                LOG_TEE("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
        }

        if ((*params).n_keep > add_bos)
        {
            LOG_TEE("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < (*params).n_keep; i++)
            {
                LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_TEE("'\n");
        }
        LOG_TEE("\n");
    }

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, (*params).n_batch, (*params).n_predict, (*params).n_keep);

    // "Black magic" grouping attention state
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

    // This is to echo initial or not
    bool display = false;

    // How many tokens have been 'traversed' (from prompt beginning)
    int n_past = state->n_past;
    // Maximum tokens to be predicted
    // Basically sampling budget
    int n_remain = (*params).n_predict;
    // This is the one responsible for "re-building KV cache"
    int n_consumed = state->n_consumed;
    // This is basically n_past but for guidance token
    int n_past_guidance = 0;

    std::vector<int> input_tokens;
    std::vector<int> output_tokens;
    std::ostringstream output_ss;

    std::vector<llama_token> embd = state->tokens;
    std::vector<llama_token> embd_guidance;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve((*params).antiprompt.size());
    for (const std::string &antiprompt : (*params).antiprompt)
    {
        antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
    }

    // Create sampling prompts
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
    if (!ctx_sampling)
    {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        throw std::runtime_error(std::string(__func__) + ": failed to initialize sampling subsystem");
    }

    // prediction loop
    while (!should_yield && (n_remain != 0))
    {
        // Probably initialization
        if (!embd.empty())
        {
            // Note: (n_ctx - 4) here is to match the logic for command line prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size)
            {
                const int skipped_tokens = (int)embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                fflush(stdout);
            }

            // Note: This section probably has something to do with handling the context
            if (ga_n == 1)
            {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int)embd.size() + std::max<int>(0, guidance_offset) >= n_ctx)
                {
                    if ((*params).n_predict == -2)
                    {
                        LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, (*params).n_predict);
                        break;
                    }

                    const int n_left = n_past - (*params).n_keep;
                    const int n_discard = n_left / 2;

                    LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, (*params).n_keep, n_discard);

                    llama_kv_cache_seq_rm(ctx, 0, (*params).n_keep, (*params).n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, (*params).n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    if (ctx_guidance)
                    {
                        n_past_guidance -= n_discard;
                    }

                    LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);
                    LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());
                }
            }
            else
            {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w)
                {
                    const int ib = (ga_n * ga_i) / ga_w;
                    const int bd = (ga_w / ga_n) * (ga_n - 1);
                    const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                    LOG("\n");
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib * bd, ga_i + ib * bd, n_past + ib * bd);
                    LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past + ib * bd, dd, ga_i + ib * bd + ga_w + dd, n_past + ib * bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib * bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);

                    n_past -= bd;

                    ga_i += ga_w / ga_n;

                    LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            // This is basically to "build" the guidance context if not yet
            if (ctx_guidance)
            {
                int input_size = 0;
                llama_token *input_buf = NULL;

                if (n_past_guidance < (int)guidance_inp.size())
                {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end())
                    {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end());
                    }

                    input_buf = embd_guidance.data();
                    input_size = embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
                }
                else
                {
                    input_buf = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += (*params).n_batch)
                {
                    int n_eval = std::min(input_size - i, (*params).n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0)))
                    {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        throw std::runtime_error(std::string(__func__) + ": failed to eval");
                    }

                    n_past_guidance += n_eval;
                }
            }

            // Not sure what this does, but it seems to check all the batched decoded result
            for (int i = 0; i < (int)embd.size(); i += (*params).n_batch)
            {
                int n_eval = (int)embd.size() - i;
                if (n_eval > (*params).n_batch)
                {
                    n_eval = (*params).n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0)))
                {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    throw std::runtime_error(std::string(__func__) + ": failed to eval");
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if ((*params).n_print > 0 && n_past % (*params).n_print == 0)
                {
                    LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // Sample the prediction result here
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            llama_sampling_accept(ctx_sampling, ctx, id, true);

            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            // Push it back into the prompt/embd
            embd.push_back(id);

            // echo this to console
            display = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        }
        else
        {
            // NOTE:
            // So this is what they mean by forwarding, this is the rebuilding part I think

            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int)embd_inp.size(), n_consumed);
            while ((int)embd_inp.size() > n_consumed)
            {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int)embd.size() >= (*params).n_batch)
                {
                    break;
                }
            }
        }

        // We do this to bypass initial prompt
        // If the embd is empty, the if statement would pretty much tokenize
        // the prompt and push it back into the embeddings. Once it do that, then
        // we want to enable this
        if (display)
        {
            for (auto id : embd)
            {
                const std::string token_str = llama_token_to_piece(ctx, id, !(*params).conversation);
                bool is_bos = (id == llama_token_bos(model));
                bool is_eos = (id == llama_token_eos(model));

                // Output or signal eos or bos when ONLY when the user requested
                if ((!is_bos || output_bos) && (!is_eos || output_eos))
                {
                    // Override and append godot trigger word
                    generated_text.append(token_str);
                    on_new_token(token_str);
                }

                if (embd.size() > 1)
                {
                    input_tokens.push_back(id);
                }
                else
                {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
            fflush(stdout);
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back()))
        {
            LOG(" [end of text]\n");
            break;
        }
    }

    // Free all the used context here
    // Apart from one provided by the constructor
    llama_print_timings(ctx);

    // Assume that state is handled immutably
    delete state;

    llama_sampling_free(ctx_sampling);
    if (ctx_guidance)
        llama_free(ctx_guidance);

    // Reset context if needed (ensure previous prompt does not get carried)
    llama_kv_cache_clear(ctx);
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