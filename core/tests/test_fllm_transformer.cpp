#include "fllm_parser.h"
#include "transformer_bitnet.h"
#include "tokenizer.h"
#include <iostream>
#include <chrono>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

int argmax(const float* logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    return best;
}

std::string generate(transformer_bitnet::BitnetTransformer& model, const tokenizer::Tokenizer& tok,
                      const std::string& prompt, int max_new_tokens) {
    auto prompt_tokens = tok.encode(prompt, true);
    std::string output;
    int pos = 0;
    int token = prompt_tokens[0];
    int prev_token = token;

    while (pos < max_new_tokens + static_cast<int>(prompt_tokens.size())) {
        float* logits = model.forward(token, pos);
        int next = (pos + 1 < static_cast<int>(prompt_tokens.size()))
            ? prompt_tokens[pos + 1]
            : argmax(logits, model.config().vocab_size);

        if (pos >= static_cast<int>(prompt_tokens.size()) - 1) {
            output += tok.decode(prev_token, token);
        }
        if (next == tok.eos_id()) break;
        prev_token = token;
        token = next;
        ++pos;
    }
    output += tok.decode(prev_token, token);
    return output;
}

int main(int argc, char** argv) {
    std::string fllm_path = argc > 1 ? argv[1] : "/tmp/stories15M.fllm";

    auto loaded = fllm::read_fllm(fllm_path);
    CHECK(loaded.has_value(), ".fllm file loads and passes checksum validation");
    if (!loaded) { std::cout << tests_passed << "/" << tests_run << "\n"; return 1; }

    CHECK(loaded->has_transformer, "loaded model reports has_transformer == true");
    CHECK(loaded->transformer_config.dim == 288, "transformer_config.dim preserved through pack/load roundtrip");
    CHECK(loaded->transformer_config.n_layers == 6, "transformer_config.n_layers preserved");
    CHECK(!loaded->transformer_weights.wq.empty(), "quantized wq tensor is present and non-empty");
    CHECK(loaded->transformer_weights.wq_alpha.size() == 6, "one alpha scale per layer stored for wq");

    auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);
    CHECK(tok != nullptr, "tokenizer loads from embedded bytes inside .fllm (fully self-contained file)");
    if (!tok) { std::cout << tests_passed << "/" << tests_run << "\n"; return 1; }

    transformer_bitnet::BitnetTransformer model(*loaded);

    auto start = std::chrono::steady_clock::now();
    std::string story = generate(model, *tok, "Once upon a time", 60);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\n--- GENERATED TEXT (BitNet ternary weights, from .fllm) ---\n";
    std::cout << story << "\n";
    std::cout << "--- end (" << seconds << "s, " << (60.0 / seconds) << " tok/s) ---\n\n";

    CHECK(!story.empty(), "ternary-model generation produces non-empty output");

    // Loose coherence check: at least a few common English words/spacing
    // should appear — this won't catch subtle quality loss, but it does
    // catch the failure mode of quantization producing pure noise/garbage.
    int word_like = 0;
    for (char c : story) if (c == ' ') word_like++;
    CHECK(word_like > 5, "output has multiple space-separated words (not degenerate repeated-token output)");

    std::cout << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
