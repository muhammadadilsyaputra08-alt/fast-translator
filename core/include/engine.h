#pragma once
#include "fllm_format.h"
#include "fllm_parser.h"
#include "transformer.h"
#include "transformer_mixed.h"
#include "tokenizer.h"
#include "kv_cache.h"
#include "gbnf.h"
#include <string>
#include <functional>
#include <memory>

namespace engine {

enum class Backend { QNN = 0, VULKAN = 1, EXECUTORCH = 2 };
enum class SamplingMode { Argmax = 0, Temperature = 1 };

struct GenerateOptions {
    std::string prompt;
    std::string grammar_gbnf; // empty = no grammar constraint
    int max_tokens = 128;
    float temperature = 0.7f;
    SamplingMode sampling = SamplingMode::Argmax;
};

using TokenCallback = std::function<void(const std::string& token)>;

// FastEngine owns a loaded model and exposes the generate() pipeline that
// the JNI bridge calls into. Three loading paths, in order of maturity:
//
//   - load(): reads a .fllm file. If it contains a packed transformer
//     section (has_transformer == true — produced by tools/pack_fllm.cpp),
//     generation runs real INT8-quantized inference (see the Fase 3
//     quantization experiment log in notes/ for why INT8 was chosen over
//     ternary: empirically near-lossless, perplexity 27.01 vs fp32's
//     27.11, at 1/4 the storage). If the .fllm has no transformer section
//     (an older placeholder file), generation falls back to a
//     whitespace-echo loop — kept only so the very first Android demo
//     fixture still runs without needing a real model file.
//   - load_raw(): reads a real llama2.c-format checkpoint + tokenizer.bin
//     directly, fp32, no quantization. Useful for generating a fp32
//     "ground truth" to compare a packed .fllm's quality against.
class FastEngine {
public:
    static std::unique_ptr<FastEngine> load(const std::string& model_path, Backend backend);

    static std::unique_ptr<FastEngine> load_raw(
        const std::string& checkpoint_path,
        const std::string& tokenizer_path,
        Backend backend);

    void generate(const GenerateOptions& opts,
                  const TokenCallback& callback,
                  const std::function<bool()>& should_cancel);

    Backend backend() const { return backend_; }
    bool has_real_model() const { return transformer_ != nullptr || quantized_ != nullptr; }
    const fllm::FllmModel& fllm_model() const { return fllm_model_; }

private:
    FastEngine(Backend backend) : backend_(backend) {}

    void generate_echo(const GenerateOptions& opts, const TokenCallback& callback,
                        const std::function<bool()>& should_cancel);
    void generate_fp32(const GenerateOptions& opts, const TokenCallback& callback,
                        const std::function<bool()>& should_cancel);
    void generate_quantized(const GenerateOptions& opts, const TokenCallback& callback,
                             const std::function<bool()>& should_cancel);

    Backend backend_;
    fllm::FllmModel fllm_model_; // populated by load()

    // fp32 dev/reference path (load_raw())
    std::unique_ptr<transformer::Transformer> transformer_;
    // production quantized path (load(), when fllm_model_.has_transformer)
    std::unique_ptr<transformer_mixed::MixedPrecisionTransformer> quantized_;
    std::unique_ptr<tokenizer::Tokenizer> tokenizer_;

    // Active only when fllm_model_.kv_config.enable_rocketkv is set (see
    // tools/pack_fllm.cpp). When present, generate_quantized() uses
    // forward_evictable() instead of forward() to bound KV-cache memory
    // growth — verified in notes/KV_CACHE_COMPRESSION_FINDINGS.md to give
    // ~70% memory reduction with preserved output coherence.
    std::unique_ptr<kvcache::RocketKVEvictor> evictor_;
};

} // namespace engine
