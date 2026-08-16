#include "transformer.h"
#include "tokenizer.h"
#include <iostream>
#include <chrono>
#include <cmath>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

// Paths are configurable via argv so this same test runs both in this
// sandbox (/mnt/user-data/uploads/...) and in CI (models/...).
std::string g_model_path = "/mnt/user-data/uploads/stories15M.bin";
std::string g_tok_path = "/mnt/user-data/uploads/tokenizer.bin";

int argmax(const float* logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

std::string generate_story(transformer::Transformer& model, const tokenizer::Tokenizer& tok,
                            const std::string& prompt, int max_new_tokens) {
    auto prompt_tokens = tok.encode(prompt, /*add_bos=*/true);
    std::string output;
    int pos = 0;
    int token = prompt_tokens[0];
    int prev_token = token;

    while (pos < max_new_tokens + static_cast<int>(prompt_tokens.size())) {
        float* logits = model.forward(token, pos);

        int next;
        if (pos + 1 < static_cast<int>(prompt_tokens.size())) {
            next = prompt_tokens[pos + 1];
        } else {
            next = argmax(logits, model.config().vocab_size);
        }

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

void test_config_matches_known_stories15M() {
    auto model = transformer::Transformer::load(g_model_path);
    CHECK(model != nullptr, "stories15M.bin loads successfully");
    if (!model) return;
    const auto& c = model->config();
    CHECK(c.dim == 288, "config.dim == 288");
    CHECK(c.n_layers == 6, "config.n_layers == 6");
    CHECK(c.n_heads == 6, "config.n_heads == 6");
    CHECK(c.vocab_size == 32000, "config.vocab_size == 32000");
    CHECK(c.seq_len == 256, "config.seq_len == 256");
}

void test_tokenizer_loads() {
    auto tok = tokenizer::Tokenizer::load(g_tok_path, 32000);
    CHECK(tok != nullptr, "tokenizer.bin loads successfully");
    if (!tok) return;
    auto ids = tok->encode("Once upon a time", true);
    CHECK(!ids.empty(), "encode() produces at least one token");
    CHECK(ids[0] == 1, "encode() prepends BOS token when requested");
}

void test_generation_deterministic_and_nonempty() {
    auto model = transformer::Transformer::load(g_model_path);
    auto tok = tokenizer::Tokenizer::load(g_tok_path, 32000);
    CHECK(model != nullptr && tok != nullptr, "model and tokenizer both load for generation test");
    if (!model || !tok) return;

    auto start = std::chrono::steady_clock::now();
    std::string story1 = generate_story(*model, *tok, "Once upon a time", 60);
    auto elapsed = std::chrono::steady_clock::now() - start;
    double seconds = std::chrono::duration<double>(elapsed).count();

    std::cout << "\n--- GENERATED TEXT (real stories15M weights, argmax) ---\n";
    std::cout << story1 << "\n";
    std::cout << "--- end (" << seconds << "s, "
               << (60.0 / seconds) << " tok/s) ---\n\n";

    CHECK(!story1.empty(), "generated story is non-empty");
    CHECK(story1.size() > 20, "generated story has substantial length");

    auto model2 = transformer::Transformer::load(g_model_path);
    std::string story2 = generate_story(*model2, *tok, "Once upon a time", 60);
    CHECK(story1 == story2, "generation is deterministic across independent model loads");
}

void test_different_prompts_diverge() {
    auto model = transformer::Transformer::load(g_model_path);
    auto tok = tokenizer::Tokenizer::load(g_tok_path, 32000);
    if (!model || !tok) { CHECK(false, "model/tokenizer load for divergence test"); return; }

    std::string story_a = generate_story(*model, *tok, "Once upon a time", 30);
    auto model2 = transformer::Transformer::load(g_model_path);
    std::string story_b = generate_story(*model2, *tok, "The little dog", 30);

    std::cout << "--- second prompt test ---\n" << story_b << "\n---\n\n";

    CHECK(story_a != story_b, "different prompts produce different continuations");
}

int main(int argc, char** argv) {
    if (argc > 1) g_model_path = argv[1];
    if (argc > 2) g_tok_path = argv[2];

    test_config_matches_known_stories15M();
    test_tokenizer_loads();
    test_generation_deterministic_and_nonempty();
    test_different_prompts_diverge();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
