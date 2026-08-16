// Milestone 5: mixed-precision packer.
// Attention (wq/wk/wv/wo) -> INT8 (real magnitude, 255 levels).
// FFN (w1/w2/w3) -> ternary (naive absmean threshold, same as pack_fllm.cpp).
// Embedding/norms/RoPE -> fp32 (unchanged).
#include "fllm_format.h"
#include "fllm_parser.h"
#include "bitnet_kernel.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace {
struct RawConfig { int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; };

std::vector<float> read_floats(std::ifstream& f, size_t count) {
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    return v;
}

void quantize_layers_int8(const std::vector<float>& src, int n_layers, int64_t per_layer,
                           std::vector<int8_t>& out, std::vector<float>& out_scale) {
    out.resize(src.size());
    out_scale.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        std::vector<float> slice(src.begin() + l * per_layer, src.begin() + (l + 1) * per_layer);
        float scale;
        auto q = bitnet::quantize_int8(slice, &scale);
        std::copy(q.begin(), q.end(), out.begin() + l * per_layer);
        out_scale[l] = scale;
    }
}

void quantize_layers_ternary(const std::vector<float>& src, int n_layers, int64_t per_layer,
                              std::vector<int8_t>& out, std::vector<float>& out_alpha) {
    out.resize(src.size());
    out_alpha.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        std::vector<float> slice(src.begin() + l * per_layer, src.begin() + (l + 1) * per_layer);
        float alpha;
        auto q = bitnet::quantize_ternary(slice, &alpha);
        std::copy(q.begin(), q.end(), out.begin() + l * per_layer);
        out_alpha[l] = alpha;
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: " << argv[0] << " <checkpoint.bin> <tokenizer.bin> <output.fllm>\n"; return 1; }
    std::string checkpoint_path = argv[1], tokenizer_path = argv[2], output_path = argv[3];

    std::ifstream cf(checkpoint_path, std::ios::binary);
    RawConfig rc{};
    cf.read(reinterpret_cast<char*>(&rc), sizeof(rc));
    bool shared = rc.vocab_size > 0;
    if (!shared) rc.vocab_size = -rc.vocab_size;
    int head_size = rc.dim / rc.n_heads;

    fllm::FllmModel model;
    model.header.model_type = 1;
    model.has_transformer = true;
    model.transformer_config = {rc.dim, rc.hidden_dim, rc.n_layers, rc.n_heads, rc.n_kv_heads, rc.vocab_size, rc.seq_len};

    auto& w = model.transformer_weights;
    int64_t dim2 = (int64_t)rc.dim * rc.dim;
    int64_t dim_hidden = (int64_t)rc.dim * rc.hidden_dim;

    w.token_embedding_table = read_floats(cf, (size_t)rc.vocab_size * rc.dim);
    w.rms_att_weight = read_floats(cf, (size_t)rc.n_layers * rc.dim);

    std::vector<float> raw_wq = read_floats(cf, rc.n_layers * dim2);
    std::vector<float> raw_wk = read_floats(cf, rc.n_layers * dim2);
    std::vector<float> raw_wv = read_floats(cf, rc.n_layers * dim2);
    std::vector<float> raw_wo = read_floats(cf, rc.n_layers * dim2);

    w.rms_ffn_weight = read_floats(cf, (size_t)rc.n_layers * rc.dim);

    std::vector<float> raw_w1 = read_floats(cf, rc.n_layers * dim_hidden);
    std::vector<float> raw_w2 = read_floats(cf, rc.n_layers * dim_hidden);
    std::vector<float> raw_w3 = read_floats(cf, rc.n_layers * dim_hidden);

    w.rms_final_weight = read_floats(cf, rc.dim);
    w.freq_cis_real = read_floats(cf, (size_t)rc.seq_len * (head_size / 2));
    w.freq_cis_imag = read_floats(cf, (size_t)rc.seq_len * (head_size / 2));

    std::cout << "quantizing attention to INT8...\n";
    quantize_layers_int8(raw_wq, rc.n_layers, dim2, w.wq, w.wq_alpha);
    quantize_layers_int8(raw_wk, rc.n_layers, dim2, w.wk, w.wk_alpha);
    quantize_layers_int8(raw_wv, rc.n_layers, dim2, w.wv, w.wv_alpha);
    quantize_layers_int8(raw_wo, rc.n_layers, dim2, w.wo, w.wo_alpha);

    std::cout << "quantizing FFN to ternary...\n";
    quantize_layers_ternary(raw_w1, rc.n_layers, dim_hidden, w.w1, w.w1_alpha);
    quantize_layers_ternary(raw_w2, rc.n_layers, dim_hidden, w.w2, w.w2_alpha);
    quantize_layers_ternary(raw_w3, rc.n_layers, dim_hidden, w.w3, w.w3_alpha);

    std::ifstream tf(tokenizer_path, std::ios::binary);
    model.tokenizer_json.assign((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    model.kv_config = {true, false, 16, 5.0f, 512};
    model.medusa_config = {4, (uint32_t)rc.dim, 0.8f, 0.5f};
    model.grammar_gbnf = "";
    model.weights = {};

    if (!fllm::write_fllm(output_path, model)) { std::cerr << "write failed\n"; return 1; }

    std::ifstream check(output_path, std::ios::binary | std::ios::ate);
    auto out_size = check.tellg();
    std::ifstream orig(checkpoint_path, std::ios::binary | std::ios::ate);
    auto orig_size = orig.tellg();
    std::cout << "wrote " << output_path << " (" << (out_size/1024/1024) << " MB, "
              << (100.0*out_size/orig_size) << "% of original " << (orig_size/1024/1024) << " MB)\n";
    return 0;
}
