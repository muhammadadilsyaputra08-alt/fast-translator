#include "engine.h"
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

namespace engine {

std::unique_ptr<FastEngine> FastEngine::load(const std::string& model_path, Backend backend) {
    auto loaded = fllm::read_fllm(model_path);
    if (!loaded.has_value()) {
        std::cerr << "[engine] failed to load .fllm model: " << model_path << "\n";
        return nullptr;
    }
    auto eng = std::unique_ptr<FastEngine>(new FastEngine(backend));
    eng->fllm_model_ = std::move(*loaded);

    if (eng->fllm_model_.has_transformer) {
        eng->quantized_ = std::make_unique<transformer_mixed::MixedPrecisionTransformer>(
            eng->fllm_model_, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
        eng->tokenizer_ = tokenizer::Tokenizer::load_from_bytes(
            eng->fllm_model_.tokenizer_json, eng->fllm_model_.transformer_config.vocab_size);
        if (!eng->tokenizer_) {
            std::cerr << "[engine] .fllm has a transformer section but tokenizer bytes failed to parse\n";
            return nullptr;
        }

        if (eng->fllm_model_.kv_config.enable_rocketkv) {
            size_t threshold = eng->fllm_model_.kv_config.rocketkv_evict_threshold;
            // keep_top_n: 60% of threshold is a conservative default that
            // leaves headroom before the next eviction pass triggers again.
            size_t keep_top_n = std::max<size_t>(8, threshold * 3 / 5);
            eng->evictor_ = std::make_unique<kvcache::RocketKVEvictor>(threshold, keep_top_n);
        }
    }
    return eng;
}

std::unique_ptr<FastEngine> FastEngine::load_raw(
    const std::string& checkpoint_path,
    const std::string& tokenizer_path,
    Backend backend) {

    auto model = transformer::Transformer::load(checkpoint_path);
    if (!model) {
        std::cerr << "[engine] failed to load transformer checkpoint: " << checkpoint_path << "\n";
        return nullptr;
    }
    auto tok = tokenizer::Tokenizer::load(tokenizer_path, model->config().vocab_size);
    if (!tok) {
        std::cerr << "[engine] failed to load tokenizer: " << tokenizer_path << "\n";
        return nullptr;
    }

    auto eng = std::unique_ptr<FastEngine>(new FastEngine(backend));
    eng->transformer_ = std::move(model);
    eng->tokenizer_ = std::move(tok);
    return eng;
}

void FastEngine::generate(const GenerateOptions& opts,
                           const TokenCallback& callback,
                           const std::function<bool()>& should_cancel) {
    if (quantized_ && tokenizer_) {
        generate_quantized(opts, callback, should_cancel);
    } else if (transformer_ && tokenizer_) {
        generate_fp32(opts, callback, should_cancel);
    } else {
        generate_echo(opts, callback, should_cancel);
    }
}

void FastEngine::generate_echo(const GenerateOptions& opts,
                                const TokenCallback& callback,
                                const std::function<bool()>& should_cancel) {
    std::istringstream iss(opts.prompt);
    std::string word;
    int emitted = 0;
    while (iss >> word && emitted < opts.max_tokens) {
        if (should_cancel && should_cancel()) break;
        callback(word);
        ++emitted;
    }
}

namespace {
int sample_argmax(const float* logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    return best;
}
int sample_temperature(const float* logits, int n, float temperature, std::mt19937& rng) {
    std::vector<float> probs(logits, logits + n);
    float max_val = *std::max_element(probs.begin(), probs.end());
    double sum = 0.0;
    for (float& p : probs) { p = std::exp((p - max_val) / std::max(temperature, 1e-4f)); sum += p; }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng), cumulative = 0.0;
    for (int i = 0; i < n; ++i) { cumulative += probs[i]; if (r <= cumulative) return i; }
    return n - 1;
}

// Masks out every vocab token whose decoded string isn't a legal
// continuation of the grammar from `state`, then samples (argmax or
// temperature, per opts.sampling) over what remains. Brute-forces the
// whole vocabulary each call -- straightforward and correct, and fast
// enough at this model's ~32k vocab / 6-layer scale; a production system
// serving a much larger vocab would want to cache per-token feasibility
// across steps instead of recomputing from scratch every time.
int sample_grammar_constrained(
    const float* logits, int vocab_size, const tokenizer::Tokenizer& tok, int prev_token,
    const gbnf::Grammar& grammar, const gbnf::Grammar::State& state,
    const GenerateOptions& opts, std::mt19937& rng) {

    std::vector<float> masked(logits, logits + vocab_size);
    bool any_valid = false;

    for (int t = 0; t < vocab_size; ++t) {
        std::string piece = tok.decode(prev_token, t);
        auto s = state;
        bool ok = true;
        for (unsigned char c : piece) {
            if (!grammar.can_accept(s, c)) { ok = false; break; }
            s = grammar.advance(s, c);
        }
        if (ok) any_valid = true;
        else masked[t] = -std::numeric_limits<float>::infinity();
    }

    if (!any_valid) {
        // Grammar has no legal continuation from here (e.g. an
        // over-constrained grammar, or the model's vocab can't express the
        // needed next character) -- fall back to unconstrained sampling
        // rather than silently producing garbage/crashing.
        return (opts.sampling == SamplingMode::Temperature)
            ? sample_temperature(logits, vocab_size, opts.temperature, rng)
            : sample_argmax(logits, vocab_size);
    }

    return (opts.sampling == SamplingMode::Temperature)
        ? sample_temperature(masked.data(), vocab_size, opts.temperature, rng)
        : sample_argmax(masked.data(), vocab_size);
}
} // namespace

void FastEngine::generate_fp32(const GenerateOptions& opts,
                                const TokenCallback& callback,
                                const std::function<bool()>& should_cancel) {
    auto prompt_tokens = tokenizer_->encode(opts.prompt, true);
    if (prompt_tokens.empty()) return;
    static std::mt19937 rng(std::random_device{}());

    int pos = 0, token = prompt_tokens[0], prev_token = token;
    int vocab_size = transformer_->config().vocab_size;
    int limit = std::min(opts.max_tokens + (int)prompt_tokens.size(), transformer_->config().seq_len);

    while (pos < limit) {
        if (should_cancel && should_cancel()) break;
        float* logits = transformer_->forward(token, pos);
        int next = (pos + 1 < (int)prompt_tokens.size()) ? prompt_tokens[pos + 1]
            : (opts.sampling == SamplingMode::Temperature ? sample_temperature(logits, vocab_size, opts.temperature, rng)
                                                            : sample_argmax(logits, vocab_size));
        if (pos >= (int)prompt_tokens.size() - 1) callback(tokenizer_->decode(prev_token, token));
        if (next == tokenizer_->eos_id()) break;
        prev_token = token; token = next; ++pos;
    }
}

void FastEngine::generate_quantized(const GenerateOptions& opts,
                                     const TokenCallback& callback,
                                     const std::function<bool()>& should_cancel) {
    if (evictor_) quantized_->reset_evictable_cache();
    auto prompt_tokens = tokenizer_->encode(opts.prompt, true);
    if (prompt_tokens.empty()) return;
    static std::mt19937 rng(std::random_device{}());

    int pos = 0, token = prompt_tokens[0], prev_token = token;
    int vocab_size = fllm_model_.transformer_config.vocab_size;
    int limit = std::min(opts.max_tokens + (int)prompt_tokens.size(), fllm_model_.transformer_config.seq_len);

    // Grammar-guided sampling: parsed fresh per generate() call so state
    // never leaks across separate generations. Only active once we're past
    // the prompt (the prompt itself is never grammar-constrained).
    std::unique_ptr<gbnf::Grammar> grammar;
    gbnf::Grammar::State grammar_state;
    if (!opts.grammar_gbnf.empty()) {
        grammar = gbnf::Grammar::parse(opts.grammar_gbnf);
        if (grammar) grammar_state = grammar->initial_state();
        else std::cerr << "[engine] grammar failed to parse; generating unconstrained\n";
    }

    while (pos < limit) {
        if (should_cancel && should_cancel()) break;
        float* logits = evictor_ ? quantized_->forward_evictable(token, pos, *evictor_)
                                  : quantized_->forward(token, pos);

        int next;
        if (pos + 1 < (int)prompt_tokens.size()) {
            next = prompt_tokens[pos + 1];
        } else if (grammar) {
            next = sample_grammar_constrained(logits, vocab_size, *tokenizer_, token, *grammar, grammar_state, opts, rng);
        } else {
            next = (opts.sampling == SamplingMode::Temperature)
                ? sample_temperature(logits, vocab_size, opts.temperature, rng)
                : sample_argmax(logits, vocab_size);
        }

        if (pos >= (int)prompt_tokens.size() - 1) callback(tokenizer_->decode(prev_token, token));

        // Once past the prompt, advance the grammar state by the token
        // that's actually about to be emitted next iteration.
        if (grammar && pos >= (int)prompt_tokens.size() - 1) {
            std::string piece = tokenizer_->decode(token, next);
            for (unsigned char c : piece) {
                if (!grammar->can_accept(grammar_state, c)) break; // shouldn't happen; defensive
                grammar_state = grammar->advance(grammar_state, c);
            }
        }

        if (next == tokenizer_->eos_id()) break;
        prev_token = token; token = next; ++pos;
    }
}

} // namespace engine
