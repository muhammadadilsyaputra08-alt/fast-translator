#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace fllm {

// Magic number "FLLM" as little-endian uint32
constexpr uint32_t kMagic = 0x4D4C4C46; // bytes 'F','L','L','M' little-endian read
constexpr uint16_t kVersion = 0x0200; // v2: adds native quantized-transformer sections

enum class ModelType : uint16_t { Base = 0, Instruct = 1, Code = 2 };

// On-disk header, packed. v2 adds two offsets (transformer config + weights)
// after the original v1 fields, so v1 tooling that only reads the first six
// offsets still finds them at the same byte positions.
#pragma pack(push, 1)
struct FllmHeader {
    uint32_t magic;                  // 0x00
    uint16_t version;                // 0x04
    uint16_t model_type;             // 0x06
    uint64_t weights_offset;         // 0x08  (legacy generic weights blob)
    uint64_t tokenizer_offset;       // 0x10  (raw tokenizer.bin bytes)
    uint64_t embeddings_offset;      // 0x18  (legacy generic embeddings)
    uint64_t grammar_offset;         // 0x20
    uint64_t kv_cache_offset;        // 0x28
    uint64_t medusa_offset;          // 0x30
    uint64_t transformer_config_offset;  // 0x38  (v2)
    uint64_t transformer_weights_offset; // 0x40  (v2)
    uint64_t checksum;               // 0x48 (first 8 bytes of SHA-256 over payload)
};
#pragma pack(pop)

static_assert(sizeof(FllmHeader) == 0x50, "FllmHeader must be exactly 80 bytes (v2)");

struct KVCacheConfig {
    bool enable_minicache;
    bool enable_rocketkv;
    uint32_t compress_every; // tokens
    float minicache_merge_ratio;
    uint32_t rocketkv_evict_threshold; // tokens before eviction kicks in
};

struct MedusaHeadConfig {
    uint32_t num_heads;      // K, typically 2-4
    uint32_t hidden_dim;
    float epsilon;           // typical-acceptance threshold
    float delta;             // typical-acceptance threshold
};

// Transformer architecture config (mirrors transformer::Config so the
// forward pass can be reconstructed purely from what's stored in the file).
struct TransformerConfig {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
};

// The seven big matmul weight tensors are BitNet-ternary quantized (one
// alpha scale per layer per tensor); everything else (embeddings, norms,
// RoPE tables) stays fp32, matching common BitNet practice of leaving the
// embedding/classifier and normalization layers at full precision.
struct QuantizedTransformerWeights {
    std::vector<float> token_embedding_table; // [vocab_size, dim] fp32
    std::vector<float> rms_att_weight;        // [n_layers, dim] fp32
    std::vector<float> rms_ffn_weight;        // [n_layers, dim] fp32
    std::vector<float> rms_final_weight;      // [dim] fp32
    std::vector<float> freq_cis_real;         // [seq_len, head_size/2] fp32
    std::vector<float> freq_cis_imag;         // [seq_len, head_size/2] fp32

    // wq/wo: [n_layers, dim, dim]. wk/wv: [n_layers, kv_dim, dim] where
    // kv_dim = n_kv_heads * (dim/n_heads) -- narrower than dim under GQA
    // (n_kv_heads < n_heads); equal to dim under plain MHA (n_kv_heads ==
    // n_heads, e.g. stories15M).
    std::vector<int8_t> wq, wk, wv, wo;       // ternary/int8, see shapes above
    std::vector<float> wq_alpha, wk_alpha, wv_alpha, wo_alpha; // [n_layers]

    std::vector<int8_t> w1, w3;               // ternary, [n_layers, hidden_dim, dim]
    std::vector<int8_t> w2;                   // ternary, [n_layers, dim, hidden_dim]
    std::vector<float> w1_alpha, w2_alpha, w3_alpha; // [n_layers]

    // Unshared output classifier (e.g. TinyLlama). EMPTY when the model
    // ties its classifier to token_embedding_table (e.g. stories15M) --
    // an empty vector here IS the "shared classifier" signal, no separate
    // flag needed. fp32, matching token_embedding_table's precision.
    std::vector<float> wcls;                  // [vocab_size, dim] fp32, or empty
};

// In-memory representation after a successful load.
struct FllmModel {
    FllmHeader header;
    std::string tokenizer_json;      // v1: JSON; v2: raw tokenizer.bin bytes (binary-safe string)
    std::vector<float> embeddings;   // legacy v1 generic field
    std::string grammar_gbnf;
    KVCacheConfig kv_config;
    MedusaHeadConfig medusa_config;
    std::vector<int8_t> weights;     // legacy v1 generic field

    bool has_transformer = false;
    TransformerConfig transformer_config{};
    QuantizedTransformerWeights transformer_weights;
};

} // namespace fllm

