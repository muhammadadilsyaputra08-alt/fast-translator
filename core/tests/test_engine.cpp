#include "engine.h"
#include "fllm_parser.h"
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

static void make_dummy_model(const std::string& path) {
    fllm::FllmModel m;
    m.header.model_type = 1;
    m.tokenizer_json = R"({"vocab":{}})";
    m.embeddings = {0.1f, 0.2f};
    m.grammar_gbnf = "";
    m.kv_config = {true, false, 16, 5.0f, 512};
    m.medusa_config = {4, 128, 0.8f, 0.5f};
    m.weights = {1, -1, 0};
    fllm::write_fllm(path, m);
}

void test_load_missing_file_returns_null() {
    auto eng = engine::FastEngine::load("/tmp/does_not_exist.fllm", engine::Backend::QNN);
    CHECK(eng == nullptr, "loading a missing file returns nullptr");
}

void test_load_valid_model() {
    make_dummy_model("/tmp/engine_test.fllm");
    auto eng = engine::FastEngine::load("/tmp/engine_test.fllm", engine::Backend::QNN);
    CHECK(eng != nullptr, "loading a valid model succeeds");
    if (eng) {
        CHECK(eng->backend() == engine::Backend::QNN, "backend recorded correctly");
        CHECK(eng->fllm_model().medusa_config.num_heads == 4, "loaded model config accessible");
    }
}

void test_generate_streams_tokens_in_order() {
    make_dummy_model("/tmp/engine_test2.fllm");
    auto eng = engine::FastEngine::load("/tmp/engine_test2.fllm", engine::Backend::VULKAN);

    engine::GenerateOptions opts;
    opts.prompt = "halo dunia dari fasttranslator";
    opts.max_tokens = 10;

    std::vector<std::string> received;
    eng->generate(opts, [&](const std::string& tok) { received.push_back(tok); }, nullptr);

    std::vector<std::string> expected = {"halo", "dunia", "dari", "fasttranslator"};
    CHECK(received == expected, "tokens streamed in correct order");
}

void test_generate_respects_max_tokens() {
    make_dummy_model("/tmp/engine_test3.fllm");
    auto eng = engine::FastEngine::load("/tmp/engine_test3.fllm", engine::Backend::EXECUTORCH);

    engine::GenerateOptions opts;
    opts.prompt = "satu dua tiga empat lima enam tujuh";
    opts.max_tokens = 3;

    int count = 0;
    eng->generate(opts, [&](const std::string&) { ++count; }, nullptr);
    CHECK(count == 3, "generation stops at max_tokens");
}

void test_generate_cancellation() {
    make_dummy_model("/tmp/engine_test4.fllm");
    auto eng = engine::FastEngine::load("/tmp/engine_test4.fllm", engine::Backend::QNN);

    engine::GenerateOptions opts;
    opts.prompt = "a b c d e f g h i j";
    opts.max_tokens = 100;

    int count = 0;
    auto should_cancel = [&]() { return count >= 2; };
    eng->generate(opts, [&](const std::string&) { ++count; }, should_cancel);
    CHECK(count == 2, "cancellation flag stops generation early");
}

void test_generate_runs_off_main_thread() {
    // Mirrors how the JNI bridge will actually invoke generate(): on a
    // detached worker thread, with results collected via a thread-safe queue.
    make_dummy_model("/tmp/engine_test5.fllm");
    auto eng = engine::FastEngine::load("/tmp/engine_test5.fllm", engine::Backend::QNN);

    engine::GenerateOptions opts;
    opts.prompt = "async streaming test";
    opts.max_tokens = 10;

    std::atomic<int> count{0};
    std::thread worker([&]() {
        eng->generate(opts, [&](const std::string&) { count++; }, nullptr);
    });
    worker.join();

    CHECK(count == 3, "generate() works correctly when run on a worker thread");
}

int main() {
    test_load_missing_file_returns_null();
    test_load_valid_model();
    test_generate_streams_tokens_in_order();
    test_generate_respects_max_tokens();
    test_generate_cancellation();
    test_generate_runs_off_main_thread();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
