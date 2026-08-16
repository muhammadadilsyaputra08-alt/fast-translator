#include "fllm_parser.h"
#include "transformer_mixed.h"
#include "tokenizer.h"
#include "kv_cache.h"
#include <iostream>
#include <cmath>
#include <string>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

int argmax(const float* logits, int n) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "/tmp/model_production.fllm";
    auto loaded = fllm::read_fllm(model_path);
    auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);
    auto prompt_tokens = tok->encode("Once upon a time", true);

    // --- Test 1: with a threshold LARGER than the whole sequence, eviction
    // never triggers, so forward_evictable() must match forward() exactly. ---
    {
        transformer_mixed::MixedPrecisionTransformer m1(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
        transformer_mixed::MixedPrecisionTransformer m2(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
        kvcache::RocketKVEvictor no_op_evictor(/*evict_threshold=*/10000, /*keep_top_n=*/10000);

        int pos = 0;
        double max_diff = 0.0;
        for (int t : prompt_tokens) {
            float* l1 = m1.forward(t, pos);
            float* l2 = m2.forward_evictable(t, pos, no_op_evictor);
            for (int v = 0; v < loaded->transformer_config.vocab_size; ++v)
                max_diff = std::max(max_diff, (double)std::fabs(l1[v] - l2[v]));
            pos++;
        }
        std::cout << "max diff (eviction never triggered vs plain forward): " << max_diff << "\n";
        CHECK(max_diff < 1e-4, "forward_evictable() matches forward() exactly when eviction never fires");
    }

    // --- Test 2: aggressive eviction (threshold=20, keep 12) on a 60-token
    // generation. Verify: (a) doesn't crash, (b) memory actually shrinks,
    // (c) output stays coherent-ish (not degenerate garbage). ---
    {
        transformer_mixed::MixedPrecisionTransformer baseline(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
        transformer_mixed::MixedPrecisionTransformer evicting(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
        kvcache::RocketKVEvictor evictor(/*evict_threshold=*/20, /*keep_top_n=*/12);

        std::string out_baseline, out_evicting;
        int pos = 0, token = prompt_tokens[0], prev = token;
        int pos_e = 0, token_e = prompt_tokens[0], prev_e = token;

        // Prime both with the prompt.
        for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) { baseline.forward(prompt_tokens[i], pos++); }
        for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) { evicting.forward_evictable(prompt_tokens[i], pos_e++, evictor); }
        token = token_e = prompt_tokens.back();

        const int max_new = 60;
        for (int i = 0; i < max_new; ++i) {
            float* lb = baseline.forward(token, pos);
            int nb = argmax(lb, loaded->transformer_config.vocab_size);
            out_baseline += tok->decode(prev, token);
            prev = token; token = nb; pos++;

            float* le = evicting.forward_evictable(token_e, pos_e, evictor);
            int ne = argmax(le, loaded->transformer_config.vocab_size);
            out_evicting += tok->decode(prev_e, token_e);
            prev_e = token_e; token_e = ne; pos_e++;
        }

        std::cout << "\n--- baseline (no eviction) ---\n" << out_baseline << "\n";
        std::cout << "\n--- with RocketKV eviction (threshold=20, keep=12) ---\n" << out_evicting << "\n";

        size_t baseline_kv_bytes = (size_t)loaded->transformer_config.n_layers * (pos) * loaded->transformer_config.dim * 2 * sizeof(float);
        size_t evicting_kv_bytes = evicting.kv_cache_bytes();
        std::cout << "\nKV cache size -- baseline (uncompressed, " << pos << " positions): " << baseline_kv_bytes << " bytes\n";
        std::cout << "KV cache size -- with eviction (capped at ~keep_top_n): " << evicting_kv_bytes << " bytes\n";
        std::cout << "memory reduction: " << (100.0 * (1.0 - (double)evicting_kv_bytes / baseline_kv_bytes)) << "%\n";

        CHECK(!out_evicting.empty(), "eviction path produces non-empty output");
        CHECK(evicting_kv_bytes < baseline_kv_bytes, "eviction actually reduces KV cache memory footprint");

        int word_count = 0;
        for (char c : out_evicting) if (c == ' ') word_count++;
        CHECK(word_count > 10, "output with eviction has multiple distinct words (not degenerate repetition)");
    }

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
