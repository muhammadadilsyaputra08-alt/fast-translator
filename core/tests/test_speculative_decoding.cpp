// Real speculative decoding experiment: Prompt Lookup Decoding (no extra
// trained heads needed, unlike Medusa — drafts come from n-gram repeats in
// the text generated/seen so far, verified in one batched forward pass).
// This is a legitimate technique used in production systems (HF
// transformers' assisted_generation, vLLM). Measures honest end-to-end
// tok/s against plain sequential generation on the SAME model/prompt.
#include "fllm_parser.h"
#include "transformer_mixed.h"
#include "tokenizer.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

using Transformer = transformer_mixed::MixedPrecisionTransformer;

int argmax(const float* logits, int n) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

// Looks for the most recent occurrence of the last `ngram_size` tokens
// earlier in `history` (excluding the match at the very end), and if
// found, returns up to `max_draft` tokens that followed that earlier
// occurrence — the standard Prompt Lookup Decoding draft rule.
std::vector<int> lookup_draft(const std::vector<int>& history, int ngram_size, int max_draft) {
    int h = (int)history.size();
    if (h < ngram_size + 1) return {};
    std::vector<int> needle(history.end() - ngram_size, history.end());

    for (int start = h - ngram_size - 1; start >= 0; --start) {
        bool match = true;
        for (int i = 0; i < ngram_size; ++i) {
            if (history[start + i] != needle[i]) { match = false; break; }
        }
        if (match) {
            std::vector<int> draft;
            int follow_start = start + ngram_size;
            for (int i = 0; i < max_draft && follow_start + i < h; ++i) {
                draft.push_back(history[follow_start + i]);
            }
            return draft;
        }
    }
    return {};
}

std::string generate_baseline(Transformer& model, const tokenizer::Tokenizer& tok,
                               std::vector<int>& history, int max_new_tokens) {
    std::string output;
    int pos = (int)history.size() - 1;
    int token = history.back();
    int prev_token = token;

    for (int i = 0; i < max_new_tokens; ++i) {
        float* logits = model.forward(token, pos);
        int next = argmax(logits, model.config().vocab_size);
        output += tok.decode(prev_token, token);
        if (next == tok.eos_id()) break;
        history.push_back(next);
        prev_token = token; token = next; ++pos;
    }
    return output;
}

// Speculative generation using prompt-lookup drafting + batched verification.
std::string generate_speculative(Transformer& model, const tokenizer::Tokenizer& tok,
                                  std::vector<int>& history, int max_new_tokens,
                                  int ngram_size, int max_draft, int* accepted_out, int* proposed_out) {
    std::string output;
    int pos = (int)history.size() - 1;
    int token = history.back();
    int prev_token = token;
    int generated = 0;
    int accepted_total = 0, proposed_total = 0;

    while (generated < max_new_tokens) {
        auto draft = lookup_draft(history, ngram_size, max_draft);
        proposed_total += (int)draft.size();

        // Verify: batch = [current token, draft tokens...], positions pos..pos+draft.size()
        std::vector<int> batch_tokens;
        batch_tokens.push_back(token);
        for (int d : draft) batch_tokens.push_back(d);

        auto logits_per_pos = model.forward_batch(batch_tokens, pos);

        // logits_per_pos[0] is the prediction made AFTER consuming `token`
        // at position pos — i.e. it predicts what should come next. Walk
        // through draft tokens checking if the model agrees at each step.
        int accepted = 0;
        for (int i = 0; i < (int)draft.size(); ++i) {
            int predicted = argmax(logits_per_pos[i].data(), model.config().vocab_size);
            if (predicted == draft[i]) {
                accepted++;
            } else {
                break;
            }
        }
        accepted_total += accepted;

        // Emit `token`, then the `accepted` confirmed draft tokens, then one
        // bonus token from the model's own prediction at the point of
        // divergence (or after the last accepted draft token) — standard
        // speculative decoding accounting.
        output += tok.decode(prev_token, token);
        generated++;

        for (int i = 0; i < accepted && generated < max_new_tokens; ++i) {
            prev_token = token;
            token = draft[i];
            history.push_back(token);
            output += tok.decode(prev_token, token);
            generated++;
            pos++;
        }

        if (generated >= max_new_tokens) break;

        int bonus = argmax(logits_per_pos[accepted].data(), model.config().vocab_size);
        prev_token = token;
        token = bonus;
        history.push_back(token);
        pos++;
        if (bonus == tok.eos_id()) { generated++; break; }
    }

    if (accepted_out) *accepted_out = accepted_total;
    if (proposed_out) *proposed_out = proposed_total;
    return output;
}

int main() {
    auto loaded = fllm::read_fllm("/tmp/model_production.fllm");
    auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);

    std::string prompt = "Once upon a time, there was a little girl named Lily. Lily loved to play outside. Once upon a time, there was a little";
    auto prompt_tokens = tok->encode(prompt, true);

    const int max_new = 80;

    // --- Baseline ---
    Transformer base_model(*loaded, Transformer::FfnPrecision::Int8);
    std::vector<int> hist1 = prompt_tokens;
    // Pre-fill prompt through sequential forward to establish KV cache.
    for (int i = 0; i + 1 < (int)hist1.size(); ++i) base_model.forward(hist1[i], i);

    auto t0 = std::chrono::steady_clock::now();
    std::string out_baseline = generate_baseline(base_model, *tok, hist1, max_new);
    double baseline_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // --- Speculative ---
    Transformer spec_model(*loaded, Transformer::FfnPrecision::Int8);
    std::vector<int> hist2 = prompt_tokens;
    for (int i = 0; i + 1 < (int)hist2.size(); ++i) spec_model.forward(hist2[i], i);

    int accepted = 0, proposed = 0;
    auto t1 = std::chrono::steady_clock::now();
    std::string out_spec = generate_speculative(spec_model, *tok, hist2, max_new, /*ngram=*/3, /*max_draft=*/4, &accepted, &proposed);
    double spec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    std::cout << "=== BASELINE (sequential) ===\n" << out_baseline << "\n";
    std::cout << "time: " << baseline_ms << " ms (" << (max_new / (baseline_ms/1000.0)) << " tok/s)\n\n";

    std::cout << "=== SPECULATIVE (prompt-lookup + batched verify) ===\n" << out_spec << "\n";
    std::cout << "time: " << spec_ms << " ms (" << (max_new / (spec_ms/1000.0)) << " tok/s)\n";
    std::cout << "draft tokens proposed: " << proposed << ", accepted: " << accepted
               << " (" << (proposed > 0 ? 100.0*accepted/proposed : 0.0) << "% acceptance)\n\n";

    std::cout << "outputs identical: " << (out_baseline == out_spec ? "YES (correctness preserved)" : "NO (mismatch!)") << "\n";
    std::cout << "speedup: " << (baseline_ms / spec_ms) << "x\n";

    return 0;
}
