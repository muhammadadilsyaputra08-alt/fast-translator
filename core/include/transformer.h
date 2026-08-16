#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace transformer {

struct Config {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
};

// All weight tensors, flattened, in the exact order they appear in the
// checkpoint file: token_embedding, per-layer attention weights, per-layer
// FFN weights, final norm, then RoPE frequency tables.
//
// Supports Grouped Query Attention (GQA): when n_kv_heads < n_heads, wk/wv
// are narrower than wq/wo (kv_dim = n_kv_heads * head_size instead of
// dim), and multiple query heads share each KV head during attention.
// n_kv_heads == n_heads (stories15M's case) is just the special case
// group_size == 1, so this is a strict superset of the old MHA-only path.
struct Weights {
    std::vector<float> token_embedding_table; // [vocab_size, dim]
    std::vector<float> rms_att_weight;        // [n_layers, dim]
    std::vector<float> wq;                    // [n_layers, dim, dim]
    std::vector<float> wk;                    // [n_layers, kv_dim, dim]  kv_dim = n_kv_heads * head_size
    std::vector<float> wv;                    // [n_layers, kv_dim, dim]
    std::vector<float> wo;                    // [n_layers, dim, dim]
    std::vector<float> rms_ffn_weight;        // [n_layers, dim]
    std::vector<float> w1;                    // [n_layers, hidden_dim, dim]
    std::vector<float> w2;                    // [n_layers, dim, hidden_dim]
    std::vector<float> w3;                    // [n_layers, hidden_dim, dim]
    std::vector<float> rms_final_weight;      // [dim]
    std::vector<float> freq_cis_real;         // [seq_len, head_size/2]
    std::vector<float> freq_cis_imag;         // [seq_len, head_size/2]

    // Output classifier. Only present (non-empty) when the checkpoint's
    // vocab_size was written NEGATIVE in the header, signaling an
    // UNSHARED classifier (e.g. TinyLlama) — as opposed to models like
    // stories15M that tie the classifier to token_embedding_table
    // (shared_classifier == true, wcls left empty, saves the disk space).
    std::vector<float> wcls;                  // [vocab_size, dim], only if !shared_classifier
};

// Per-step scratch buffers + the KV cache (grows across the whole sequence).
// key_cache/value_cache are sized by kv_dim (== dim when n_kv_heads ==
// n_heads, narrower under GQA), matching Weights::wk/wv above.
struct RunState {
    std::vector<float> x, xb, xb2;
    std::vector<float> hb, hb2;
    std::vector<float> q, k, v;
    std::vector<float> att;
    std::vector<float> logits;
    std::vector<float> key_cache;   // [n_layers, seq_len, kv_dim]
    std::vector<float> value_cache; // [n_layers, seq_len, kv_dim]

    void init(const Config& c);
};

class Transformer {
public:
    // Loads a raw checkpoint in the llama2.c binary layout: a 28-byte
    // (7 x int32) Config header followed immediately by all weight tensors
    // as float32, in the fixed order described in Weights above.
    // Returns nullptr on any read/size mismatch.
    static std::unique_ptr<Transformer> load(const std::string& path);

    // Runs one forward step for `token` at sequence position `pos`,
    // updating the KV cache in place, and returns a pointer to the logits
    // (length = vocab_size), owned by the internal RunState.
    float* forward(int token, int pos);

    const Config& config() const { return config_; }

private:
    Config config_{};
    Weights w_;
    RunState state_;
    bool shared_classifier_ = true;
};

} // namespace transformer
