#include "engine.h"
#include <iostream>
#include <sstream>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

std::string g_model_path = "/mnt/user-data/uploads/stories15M.bin";
std::string g_tok_path = "/mnt/user-data/uploads/tokenizer.bin";

void test_load_raw_real_model() {
    auto eng = engine::FastEngine::load_raw(g_model_path, g_tok_path, engine::Backend::QNN);
    CHECK(eng != nullptr, "load_raw() loads real stories15M checkpoint + tokenizer");
    if (eng) CHECK(eng->has_real_model(), "has_real_model() reports true after load_raw()");
}

void test_generate_via_engine_produces_coherent_stream() {
    auto eng = engine::FastEngine::load_raw(g_model_path, g_tok_path, engine::Backend::QNN);
    CHECK(eng != nullptr, "engine loads for generation test");
    if (!eng) return;

    engine::GenerateOptions opts;
    opts.prompt = "Once upon a time";
    opts.max_tokens = 40;
    opts.sampling = engine::SamplingMode::Argmax;

    std::ostringstream out;
    int token_count = 0;
    eng->generate(opts, [&](const std::string& tok) {
        out << tok;
        token_count++;
    }, nullptr);

    std::cout << "\n--- FastEngine::generate() output (via full engine API) ---\n";
    std::cout << out.str() << "\n---\n\n";

    CHECK(token_count > 0, "generate() via FastEngine emits tokens");
    CHECK(out.str().find("Lily") != std::string::npos ||
          out.str().find("little") != std::string::npos ||
          out.str().find("day") != std::string::npos,
          "output contains plausible TinyStories vocabulary (sanity check)");
}

void test_cancellation_works_with_real_model() {
    auto eng = engine::FastEngine::load_raw(g_model_path, g_tok_path, engine::Backend::QNN);
    if (!eng) { CHECK(false, "engine loads for cancellation test"); return; }

    engine::GenerateOptions opts;
    opts.prompt = "The little dog";
    opts.max_tokens = 100;

    int count = 0;
    auto should_cancel = [&]() { return count >= 5; };
    eng->generate(opts, [&](const std::string&) { ++count; }, should_cancel);

    CHECK(count == 5, "cancellation stops real-model generation exactly at the requested point");
}

int main(int argc, char** argv) {
    if (argc > 1) g_model_path = argv[1];
    if (argc > 2) g_tok_path = argv[2];

    test_load_raw_real_model();
    test_generate_via_engine_produces_coherent_stream();
    test_cancellation_works_with_real_model();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
