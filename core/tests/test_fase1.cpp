#include "fllm_parser.h"
#include "bitnet_kernel.h"
#include "kv_cache.h"
#include "medusa.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

void test_fllm_roundtrip() {
    fllm::FllmModel m;
    m.header.model_type = 1;
    m.tokenizer_json = R"({"vocab":{"hello":1,"world":2}})";
    m.embeddings = {0.1f, 0.2f, 0.3f, 0.4f};
    m.grammar_gbnf = "root ::= \"a\" | \"b\"";
    m.kv_config = {true, true, 16, 5.0f, 512};
    m.medusa_config = {4, 128, 0.8f, 0.5f};
    m.weights = {1, -1, 0, 1, -1, 0, 1, 1};

    bool wrote = fllm::write_fllm("/tmp/test_model.fllm", m);
    CHECK(wrote, "fllm write succeeds");

    auto loaded = fllm::read_fllm("/tmp/test_model.fllm");
    CHECK(loaded.has_value(), "fllm read succeeds");
    if (!loaded) return;

    CHECK(loaded->tokenizer_json == m.tokenizer_json, "tokenizer roundtrip");
    CHECK(loaded->embeddings == m.embeddings, "embeddings roundtrip");
    CHECK(loaded->grammar_gbnf == m.grammar_gbnf, "grammar roundtrip");
    CHECK(loaded->kv_config.compress_every == 16, "kv config roundtrip");
    CHECK(loaded->medusa_config.num_heads == 4, "medusa config roundtrip");
    CHECK(loaded->weights == m.weights, "weights roundtrip");
}

void test_fllm_checksum_corruption_detected() {
    fllm::FllmModel m;
    m.tokenizer_json = "test";
    m.grammar_gbnf = "root ::= \"x\"";
    m.kv_config = {true, false, 16, 5.0f, 512};
    m.medusa_config = {2, 64, 0.8f, 0.5f};
    m.weights = {1, 0, -1};
    fllm::write_fllm("/tmp/test_corrupt.fllm", m);

    // Flip a byte in the payload to simulate corruption.
    std::fstream f("/tmp/test_corrupt.fllm", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(sizeof(fllm::FllmHeader) + 20);
    char c;
    f.seekg(sizeof(fllm::FllmHeader) + 20);
    f.read(&c, 1);
    c = ~c;
    f.seekp(sizeof(fllm::FllmHeader) + 20);
    f.write(&c, 1);
    f.close();

    auto loaded = fllm::read_fllm("/tmp/test_corrupt.fllm");
    CHECK(!loaded.has_value(), "corrupted file rejected by checksum");
}

void test_bitnet_quantize_and_matmul() {
    // Weight matrix: 2x4, values chosen so sign is unambiguous after
    // absmean thresholding.
    std::vector<float> w = {
        1.0f, -1.0f, 0.01f, 0.9f,   // row 0
        -0.8f, 0.02f, 1.2f, -1.1f   // row 1
    };
    float alpha = 0.0f;
    auto tw = bitnet::quantize_ternary(w, &alpha);
    CHECK(tw.size() == 8, "ternary weight count matches input");
    CHECK(tw[0] == 1 && tw[1] == -1, "large positive/negative quantized correctly");
    CHECK(tw[2] == 0, "near-zero weight quantized to 0");

    std::vector<int8_t> act = {2, 3, 4, 1};
    auto out = bitnet::matmul_ternary_int8(tw, 2, 4, act, /*scale=*/1.0f, alpha);
    CHECK(out.size() == 2, "matmul output dim correct");

    // Manual expected calc for row 0: tw = [1,-1,0,1] -> acc = 2 -3 +0 +1 = 0
    float expected_row0 = 0.0f * alpha;
    CHECK(std::fabs(out[0] - expected_row0) < 1e-4f, "matmul row0 numerically correct");
}

void test_minicache_merge() {
    kvcache::MiniCacheCompressor comp(0.9f);
    kvcache::KVEntry a{{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}};
    kvcache::KVEntry b_similar{{0.99f, 0.01f, 0.0f}, {0.51f, 0.49f, 0.5f}};
    kvcache::KVEntry b_different{{0.0f, 1.0f, 0.0f}, {-0.5f, -0.5f, -0.5f}};

    kvcache::KVEntry merged;
    std::vector<float> residual;
    bool merged_ok = comp.try_merge(a, b_similar, &merged, &residual);
    CHECK(merged_ok, "similar KV entries get merged");

    bool merged_bad = comp.try_merge(a, b_different, &merged, &residual);
    CHECK(!merged_bad, "dissimilar KV entries are not merged");
}

void test_rocketkv_eviction() {
    kvcache::RocketKVEvictor evictor(/*evict_threshold=*/5, /*keep_top_n=*/3);
    std::vector<kvcache::KVEntry> entries;
    for (int i = 0; i < 10; ++i) {
        entries.push_back({{static_cast<float>(i), 0.0f}, {0.0f, 0.0f}});
    }
    auto kept = evictor.select_kept_indices(entries);
    CHECK(kept.size() <= 3 + 1, "eviction respects budget (allowing rounding)");
    // Most recent tokens should always survive.
    bool has_last = std::find(kept.begin(), kept.end(), entries.size() - 1) != kept.end();
    CHECK(has_last, "most recent token always kept");
}

void test_medusa_tree_mask() {
    // Tree: root(-1) -> A(0) -> B(1), root -> C(2)
    std::vector<medusa::TreeNode> nodes = {
        {10, -1, -0.1f}, // node 0 = A, child of root
        {11, 0, -0.2f},  // node 1 = B, child of A
        {12, -1, -0.3f}, // node 2 = C, child of root
    };
    auto mask = medusa::build_tree_attention_mask(nodes);
    size_t n = nodes.size();
    CHECK(mask[1 * n + 0] == 1, "B attends to its parent A");
    CHECK(mask[1 * n + 2] == 0, "B does not attend to sibling branch C");
    CHECK(mask[0 * n + 0] == 1, "A attends to itself");
}

void test_medusa_typical_acceptance() {
    medusa::TypicalAcceptanceConfig cfg{0.8f, 0.5f};
    // Peaked distribution (low entropy) -> high-prob token should be accepted
    std::vector<float> peaked = {0.95f, 0.01f, 0.01f, 0.01f, 0.02f};
    CHECK(medusa::typical_acceptance_check(peaked, 0, cfg), "high-confidence token accepted");
    CHECK(!medusa::typical_acceptance_check(peaked, 1, cfg), "low-prob token in peaked dist rejected");
}

void test_medusa_accept_chain() {
    std::vector<medusa::TreeNode> nodes = {
        {1, -1, 0.0f},  // depth 1
        {2, 0, 0.0f},   // depth 2
        {3, 1, 0.0f},   // depth 3
    };
    std::vector<std::vector<float>> probs = {
        {0.9f, 0.1f},       // node0 token=1 -> index 1 must be high prob; adjust below
        {0.9f, 0.1f},
        {0.05f, 0.95f},
    };
    // token ids are used as distribution indices in this simplified test
    std::vector<std::vector<float>> aligned_probs = {
        {0.1f, 0.9f},  // token 1 -> prob 0.9
        {0.1f, 0.1f, 0.9f}, // token 2 -> index2 prob 0.9
        {0.1f, 0.1f, 0.1f, 0.9f}, // token 3 -> index3 prob 0.9
    };
    medusa::TypicalAcceptanceConfig cfg{0.8f, 0.5f};
    auto accepted = medusa::accept_longest_chain(nodes, aligned_probs, cfg);
    CHECK(accepted.size() == 3, "full chain accepted when all steps high-confidence");
}

int main() {
    test_fllm_roundtrip();
    test_fllm_checksum_corruption_detected();
    test_bitnet_quantize_and_matmul();
    test_minicache_merge();
    test_rocketkv_eviction();
    test_medusa_tree_mask();
    test_medusa_typical_acceptance();
    test_medusa_accept_chain();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
