#include "fllm_parser.h"
#include "transformer_mixed.h"
#include "tokenizer.h"
#include <iostream>
#include <cmath>
#include <chrono>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

int main() {
    auto loaded = fllm::read_fllm("/tmp/model_production.fllm");
    if (!loaded) { std::cerr << "failed to load model\n"; return 1; }
    auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);

    auto prompt_tokens = tok->encode("Once upon a time, the little", true);
    int n = (int)prompt_tokens.size();
    std::cout << "testing with " << n << " tokens\n";

    // --- Sequential path (ground truth) ---
    transformer_mixed::MixedPrecisionTransformer seq_model(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
    std::vector<std::vector<float>> seq_logits;
    for (int pos = 0; pos < n; ++pos) {
        float* logits = seq_model.forward(prompt_tokens[pos], pos);
        seq_logits.emplace_back(logits, logits + loaded->transformer_config.vocab_size);
    }

    // --- Batched path (fresh model instance, fresh KV cache) ---
    transformer_mixed::MixedPrecisionTransformer batch_model(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
    auto batch_logits = batch_model.forward_batch(prompt_tokens, 0);

    CHECK(batch_logits.size() == seq_logits.size(), "forward_batch returns one logits vector per input token");

    double max_abs_diff = 0.0;
    for (int t = 0; t < n; ++t) {
        for (int v = 0; v < loaded->transformer_config.vocab_size; ++v) {
            max_abs_diff = std::max(max_abs_diff, (double)std::fabs(seq_logits[t][v] - batch_logits[t][v]));
        }
    }
    std::cout << "max abs logit difference (sequential vs batched): " << max_abs_diff << "\n";
    CHECK(max_abs_diff < 1e-3, "batched forward pass produces numerically identical logits to sequential calls");

    // --- Timing comparison: batch of 8 tokens, single call vs 8 sequential calls ---
    auto bench_tokens_prompt = tok->encode("The little dog ran across the yard and", true);
    int bn = (int)bench_tokens_prompt.size();

    transformer_mixed::MixedPrecisionTransformer t1(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
    auto start_seq = std::chrono::steady_clock::now();
    for (int pos = 0; pos < bn; ++pos) t1.forward(bench_tokens_prompt[pos], pos);
    double seq_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_seq).count();

    transformer_mixed::MixedPrecisionTransformer t2(*loaded, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);
    auto start_batch = std::chrono::steady_clock::now();
    t2.forward_batch(bench_tokens_prompt, 0);
    double batch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_batch).count();

    std::cout << "\n--- timing: " << bn << " tokens, prompt pre-fill ---\n";
    std::cout << "sequential (one forward() per token): " << seq_ms << " ms\n";
    std::cout << "batched (one forward_batch() call):    " << batch_ms << " ms\n";
    std::cout << "speedup: " << (seq_ms / batch_ms) << "x\n";

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
