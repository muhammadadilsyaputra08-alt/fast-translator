#pragma once
#include "fllm_format.h"
#include <vector>
#include <memory>

namespace transformer_bitnet {

// Per-step scratch buffers + KV cache, same shapes as the fp32 version.
struct RunState {
    std::vector<float> x, xb, xb2;
    std::vector<float> hb, hb2;
    std::vector<float> q, k, v;
    std::vector<float> att;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    void init(const fllm::TransformerConfig& c);
};

// Runs forward passes against BitNet-ternary-quantized weights loaded from
// a .fllm file. The seven big matmul tensors (wq/wk/wv/wo/w1/w2/w3) are
// ternary {-1,0,+1} with one alpha scale per layer; everything else
// (embedding table, norms, RoPE tables) is fp32, same precision trade-off
// convention as bitnet_kernel.h documents.
class BitnetTransformer {
public:
    explicit BitnetTransformer(const fllm::FllmModel& model);

    float* forward(int token, int pos);

    const fllm::TransformerConfig& config() const { return config_; }

private:
    const fllm::TransformerConfig& config_;
    const fllm::QuantizedTransformerWeights& w_;
    RunState state_;
};

} // namespace transformer_bitnet
